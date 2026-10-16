// treedriver.cpp - tree-structured prefill, question answering and subtree eviction.
// Build as examples/treedriver/treedriver.cpp (replaces the copy of batched.cpp).
//
// Usage:
//   llama-treedriver -m model.gguf --doc doc.json --ask-seq 1 -q "What is the capital of France?"
//                    [--evict-seq 2,5,6] [--evict-p0 0] [--evict-p1 -1] [-n 32] [-t 4]
//   --evict-seq takes a comma separated list: a subtree is evicted by removing its own id AND its
//   descendants' ids, all with the same position range [p0, p1).  p0 = start position of the subtree.
//
// doc.json (parallel arrays, one entry per token, in document order):
//   tokens[]     int token ids
//   positions[]  position of each token. MUST be consecutive within every seq id.
//   seq_ids[]    list of ints per token. FIRST id = the token's own subtree (used for its mask),
//                extra ids = descendants that are allowed to see this token (shared ancestors).
//   node_type[]  optional, parent[] optional (only used for stats)
//
// Prints lines starting with ANSWER1 / ANSWER2 / KV so tests can diff them.

#include "llama.h"
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

struct Doc {
    std::vector<llama_token>                 tok;
    std::vector<llama_pos>                   pos;
    std::vector<std::vector<llama_seq_id>>   sids;
    std::vector<std::string>                 node_type;
};

static bool load_doc(const std::string & path, Doc & d) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    json j = json::parse(f);
    d.tok  = j.at("tokens").get<std::vector<llama_token>>();
    d.pos  = j.at("positions").get<std::vector<llama_pos>>();
    d.sids = j.at("seq_ids").get<std::vector<std::vector<llama_seq_id>>>();
    if (j.contains("node_type")) d.node_type = j["node_type"].get<std::vector<std::string>>();
    if (d.tok.size() != d.pos.size() || d.tok.size() != d.sids.size()) {
        fprintf(stderr, "tokens/positions/seq_ids length mismatch\n");
        return false;
    }
    return true;
}

static void batch_add(llama_batch & b, llama_token t, llama_pos p,
                      const std::vector<llama_seq_id> & s, bool logits) {
    int i = b.n_tokens;
    b.token[i]    = t;
    b.pos[i]      = p;
    b.n_seq_id[i] = (int32_t) s.size();
    for (size_t k = 0; k < s.size(); k++) b.seq_id[i][k] = s[k];
    b.logits[i]   = logits;
    b.n_tokens++;
}

static std::string piece_of(const llama_vocab * vocab, llama_token t) {
    char buf[256];
    int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    return n > 0 ? std::string(buf, n) : std::string();
}

// Ask a question that can see only `seq` (its own subtree + shared ancestors).
// The question and the answer are removed from the cache afterwards, so you can ask again.
static std::string ask(llama_context * ctx, const llama_vocab * vocab, llama_sampler * smpl,
                       llama_batch & batch, llama_seq_id seq, const std::string & question, int n_predict) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_pos start = llama_memory_seq_pos_max(mem, seq) + 1;

    std::string text = "\n\nQuestion: " + question + "\nAnswer:";
    std::vector<llama_token> q(text.size() + 8);
    int nq = llama_tokenize(vocab, text.c_str(), (int) text.size(), q.data(), (int) q.size(), false, true);
    if (nq < 0) { fprintf(stderr, "tokenize failed\n"); return ""; }
    q.resize(nq);

    std::vector<llama_seq_id> ids = { seq };   // question tokens belong to ONE sequence only
    batch.n_tokens = 0;
    for (int i = 0; i < nq; i++) batch_add(batch, q[i], start + i, ids, i == nq - 1);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed (question)\n"); return ""; }

    std::string out;
    llama_pos p = start + nq;
    for (int i = 0; i < n_predict; i++) {
        llama_token t = llama_sampler_sample(smpl, ctx, -1);   // greedy
        if (llama_vocab_is_eog(vocab, t)) break;
        out += piece_of(vocab, t);
        batch.n_tokens = 0;
        batch_add(batch, t, p++, ids, true);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed (gen)\n"); break; }
    }
    llama_memory_seq_rm(mem, seq, start, -1);   // drop question + answer again
    for (auto & c : out) if (c == '\n') c = ' ';
    return out;
}

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    std::string model_path, doc_path, question = "Summarize this in one sentence.";
    int ask_seq = 1, evict_p0 = 0, evict_p1 = -1, n_predict = 32, n_threads = 4;
    std::vector<int> evict_seqs;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if      (a == "-m")          model_path = next();
        else if (a == "--doc")       doc_path   = next();
        else if (a == "-q")          question   = next();
        else if (a == "--ask-seq")   ask_seq    = atoi(next().c_str());
        else if (a == "--evict-seq") {
            std::stringstream ss(next());
            std::string item;
            while (std::getline(ss, item, ',')) if (!item.empty()) evict_seqs.push_back(atoi(item.c_str()));
        }
        else if (a == "--evict-p0")  evict_p0   = atoi(next().c_str());
        else if (a == "--evict-p1")  evict_p1   = atoi(next().c_str());
        else if (a == "-n")          n_predict  = atoi(next().c_str());
        else if (a == "-t")          n_threads  = atoi(next().c_str());
    }
    if (model_path.empty() || doc_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf --doc doc.json --ask-seq N -q \"question\" [--evict-seq N]\n", argv[0]);
        return 1;
    }

    Doc doc;
    if (!load_doc(doc_path, doc)) return 1;

    std::set<int> uniq;
    for (auto & s : doc.sids) for (int x : s) uniq.insert(x);
    const int n_seq_max = 64;
    if ((int) uniq.size() > n_seq_max) {
        fprintf(stderr, "%zu distinct seq ids > %d: merge small subtrees in the parser\n", uniq.size(), n_seq_max);
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 4096;
    cp.n_batch    = 512;
    cp.n_ubatch   = 512;
    cp.n_seq_max  = n_seq_max;
    cp.kv_unified = true;            // required: shared ancestors carry several seq ids
    cp.n_threads  = n_threads;
    cp.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context creation failed\n"); return 1; }

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_init(cp.n_batch, 0, n_seq_max);

    // ---- prefill, in chunks of n_batch ----
    const int n = (int) doc.tok.size();
    double t0 = now_s();
    for (int i = 0; i < n; i += cp.n_batch) {
        batch.n_tokens = 0;
        for (int k = i; k < n && k < i + (int) cp.n_batch; k++)
            batch_add(batch, doc.tok[k], doc.pos[k], doc.sids[k], false);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "prefill failed at token %d (check: positions consecutive per seq id?)\n", i);
            return 1;
        }
    }
    double t_prefill = now_s() - t0;
    printf("PREFILL tokens=%d seqs=%zu time=%.2fs\n", n, uniq.size(), t_prefill);

    llama_memory_t mem = llama_get_memory(ctx);
    size_t kv_before = llama_state_get_size(ctx);
    printf("KV state_bytes_before=%zu\n", kv_before);

    // ---- first answer ----
    t0 = now_s();
    std::string a1 = ask(ctx, vocab, smpl, batch, ask_seq, question, n_predict);
    printf("ANSWER1: %s\n", a1.c_str());
    printf("TIME answer1=%.2fs\n", now_s() - t0);

    // ---- evict a subtree and ask again ----
    if (!evict_seqs.empty()) {
        for (int s : evict_seqs) llama_memory_seq_rm(mem, s, evict_p0, evict_p1);
        size_t kv_after = llama_state_get_size(ctx);
        printf("KV state_bytes_after=%zu (dropped %zu)\n", kv_after, kv_before - kv_after);

        t0 = now_s();
        std::string a2 = ask(ctx, vocab, smpl, batch, ask_seq, question, n_predict);
        printf("ANSWER2: %s\n", a2.c_str());
        printf("TIME answer2=%.2fs\n", now_s() - t0);
        printf("EVICT_TEST %s\n", a1 == a2 ? "PASS" : "FAIL");
    }

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

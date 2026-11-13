// op_profile.cpp - per-operator time breakdown of llama.cpp on CPU, using the scheduler's
// eval callback (same mechanism as llama.cpp's own examples/eval-callback).
//
// How it works: with cb_eval set, ggml's scheduler computes the graph ONE NODE AT A TIME and calls
//   cb(node, ask=true)   right before the node runs   -> we start a clock
//   cb(node, ask=false)  right after it finished      -> we stop the clock and add the time to
//                                                        the bucket (operator, tensor-name stem)
// Cost: every node becomes a separate thread-pool job, so the total is a bit slower than a normal
// run.  We therefore ALSO time a normal un-instrumented run and report both, so the report can state
// the distortion honestly.
//
//   op_profile -m model.gguf -t 6 --phase decode --depth 4096 --n-gen 64 --tag laptop --out ops.csv
//   op_profile -m model.gguf -t 6 --phase prefill --depth 0 --n-prompt 512
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using clk = std::chrono::steady_clock;
static inline uint64_t now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count();
}

struct Stat { uint64_t ns = 0; uint64_t calls = 0; };

struct Prof {
    bool active = false;
    uint64_t t0 = 0;
    std::map<std::tuple<std::string, std::string, std::string>, Stat> stats;   // (op, stem, src0 type)
};

// "Qcur-12 (reshaped)" -> "Qcur" ; "ffn_up-3" -> "ffn_up" ; "result_output" -> "result_output"
static std::string stem_of(const char * name) {
    std::string s(name);
    size_t sp = s.find(' ');
    if (sp != std::string::npos) s.resize(sp);
    size_t dash = s.rfind('-');
    if (dash != std::string::npos && dash + 1 < s.size() &&
        std::all_of(s.begin() + dash + 1, s.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        s.resize(dash);
    }
    if (s.rfind("node_", 0) == 0) s.clear();      // unnamed graph nodes (e.g. fused flash-attention)
    return s;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * p = (Prof *) ud;
    if (!p->active) return false;                 // not profiling: let the scheduler batch nodes normally
    if (ask) { p->t0 = now_ns(); return true; }   // "yes, run this node alone and tell me when it is done"
    const uint64_t dt = now_ns() - p->t0;
    std::string src0 = (t->src[0] != nullptr) ? ggml_type_name(t->src[0]->type) : "-";
    auto & st = p->stats[std::make_tuple(std::string(ggml_op_desc(t)), stem_of(t->name), src0)];
    st.ns += dt;
    st.calls++;
    return true;
}

static ggml_type parse_kv_type(const std::string & s) {
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    fprintf(stderr, "unsupported KV type '%s' (use f16|f32|q8_0|q4_0)\n", s.c_str());
    exit(2);
}

static void usage(const char * a0) {
    fprintf(stderr,
        "usage: %s -m model.gguf [-t threads] [--phase decode|prefill] [--depth N] [--n-gen N] [--n-prompt N]\n"
        "          [--fa auto|on|off] [--ctk f16|q8_0|q4_0] [--ctv ...] [-ub N] [--tag T] [--out file.csv]\n", a0);
}

int main(int argc, char ** argv) {
    std::string model, phase = "decode", fa = "auto", tag = "run", out, ctk = "f16", ctv = "f16";
    int threads = 4, depth = 0, n_gen = 64, n_prompt = 512, ubatch = 512;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](int n) { if (i + n >= argc) { usage(argv[0]); exit(2); } };
        if      (a == "-m")         { need(1); model = argv[++i]; }
        else if (a == "-t")         { need(1); threads = atoi(argv[++i]); }
        else if (a == "--phase")    { need(1); phase = argv[++i]; }
        else if (a == "--depth")    { need(1); depth = atoi(argv[++i]); }
        else if (a == "--n-gen")    { need(1); n_gen = atoi(argv[++i]); }
        else if (a == "--n-prompt") { need(1); n_prompt = atoi(argv[++i]); }
        else if (a == "--fa")       { need(1); fa = argv[++i]; }
        else if (a == "--ctk")      { need(1); ctk = argv[++i]; }
        else if (a == "--ctv")      { need(1); ctv = argv[++i]; }
        else if (a == "-ub")        { need(1); ubatch = atoi(argv[++i]); }
        else if (a == "--tag")      { need(1); tag = argv[++i]; }
        else if (a == "--out")      { need(1); out = argv[++i]; }
        else { usage(argv[0]); return 2; }
    }
    if (model.empty() || (phase != "decode" && phase != "prefill")) { usage(argv[0]); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;                                    // CPU only
    llama_model * lm = llama_model_load_from_file(model.c_str(), mp);
    if (!lm) { fprintf(stderr, "failed to load %s\n", model.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(lm);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int steps_decode = (phase == "decode") ? n_gen : 0;
    const int n_ctx = depth + std::max(steps_decode + 8, n_prompt) + 16;

    Prof prof;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = std::max(2048, ubatch);
    cp.n_ubatch = ubatch;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                       : fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    cp.type_k = parse_kv_type(ctk);
    cp.type_v = parse_kv_type(ctv);
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &prof;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(lm, cp);
    if (!ctx) { fprintf(stderr, "failed to create context (n_ctx=%d)\n", n_ctx); return 1; }

    std::mt19937 rng(1234);
    auto rand_tok = [&]() {
        for (;;) {                                      // random *normal* tokens (no specials)
            llama_token t = 1000 + (int) (rng() % (uint32_t) std::max(1, std::min(n_vocab, 100000) - 1000));
            if (t < n_vocab && !llama_vocab_is_control(vocab, t)) return t;
        }
    };
    auto decode_tokens = [&](int n) {                   // feed n random tokens in chunks of n_batch
        std::vector<llama_token> toks(n);
        for (auto & t : toks) t = rand_tok();
        for (int i = 0; i < n; i += (int) cp.n_batch) {
            int c = std::min<int>((int) cp.n_batch, n - i);
            llama_batch b = llama_batch_get_one(toks.data() + i, c);
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "llama_decode failed\n"); exit(1); }
        }
    };
    auto reset_to_depth = [&]() {
        llama_memory_clear(llama_get_memory(ctx), true);
        prof.active = false;
        if (depth > 0) decode_tokens(depth);
    };
    auto run_measured_part = [&]() {                    // the part we time
        if (phase == "decode") {
            for (int i = 0; i < n_gen; i++) decode_tokens(1);
        } else {
            decode_tokens(n_prompt);
        }
    };

    // warm-up (page in weights, spin up threads)
    reset_to_depth();
    decode_tokens(4);

    // 1) plain, un-instrumented timing
    reset_to_depth();
    uint64_t a0 = now_ns();
    run_measured_part();
    const double plain_ms = (now_ns() - a0) / 1e6;

    // 2) instrumented
    reset_to_depth();
    prof.stats.clear();
    prof.active = true;
    uint64_t b0 = now_ns();
    run_measured_part();
    const double prof_ms = (now_ns() - b0) / 1e6;
    prof.active = false;

    const int units = (phase == "decode") ? n_gen : n_prompt;   // tokens processed in the measured part
    double sum_ms = 0;
    for (auto & kv : prof.stats) sum_ms += kv.second.ns / 1e6;

    FILE * f = out.empty() ? stdout : fopen(out.c_str(), "a");
    if (!f) { fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }
    if (!out.empty() && (fseek(f, 0, SEEK_END), ftell(f)) == 0)
        fprintf(f, "tag,model,phase,depth,threads,fa,ctk,ctv,units,op,stem,src0_type,calls,ms_total\n");
    const char * mname = strrchr(model.c_str(), '/');
    mname = mname ? mname + 1 : model.c_str();
    auto row = [&](const char * op, const char * stem, const char * ty, uint64_t calls, double ms) {
        fprintf(f, "%s,%s,%s,%d,%d,%s,%s,%s,%d,%s,%s,%s,%llu,%.4f\n", tag.c_str(), mname, phase.c_str(), depth,
                threads, fa.c_str(), ctk.c_str(), ctv.c_str(), units, op, stem, ty, (unsigned long long) calls, ms);
    };
    for (auto & kv : prof.stats)
        row(std::get<0>(kv.first).c_str(), std::get<1>(kv.first).c_str(), std::get<2>(kv.first).c_str(),
            kv.second.calls, kv.second.ns / 1e6);
    row("_WALL_PLAIN", "-", "-", 1, plain_ms);
    row("_WALL_PROFILED", "-", "-", 1, prof_ms);
    if (f != stdout) fclose(f);

    fprintf(stderr, "[op_profile] %s depth=%d threads=%d: plain %.3f ms/token (%.2f tok/s) | instrumented %.3f ms/token | "
                    "sum of operator times %.3f ms/token\n", phase.c_str(), depth, threads, plain_ms / units,
            units * 1000.0 / plain_ms, prof_ms / units, sum_ms / units);

    llama_free(ctx);
    llama_model_free(lm);
    llama_backend_free();
    return 0;
}

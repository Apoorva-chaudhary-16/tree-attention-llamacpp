#!/usr/bin/env python3
"""Build tiny test documents for llama-treedriver (no extra Python packages needed).

Tree:  root (seq 0, visible to everyone)
         |- section A (seq 1)   <- we ask questions about this one
         |- section B (seq 2)   <- we change / evict this one
         |- section C (seq 3)   <- unrelated, extra eviction target

Schema per token (parallel arrays): tokens, positions, seq_ids, node_type, parent.
seq_ids[i][0] is the token's OWN subtree; extra ids list who may also see it.
Root tokens carry [0,1,2,3] so every section can see them. Each section starts at the
same position (right after the root), so changing B's length never shifts A's positions.

Usage:
  python3 make_test_docs.py --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
      --tokenizer build/bin/llama-tokenize --out-dir tests/tree
"""
import argparse, json, os, subprocess


def tokenize(tok_bin, model, text, add_bos):
    cmd = [tok_bin, "-m", model, "--ids", "--log-disable", "-p", text]
    if not add_bos:
        cmd.append("--no-bos")
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in reversed(r.stdout.splitlines()):
        line = line.strip()
        if line.startswith("["):
            return json.loads(line)
    raise RuntimeError("could not parse llama-tokenize output:\n" + r.stdout + r.stderr)


def build(tok_bin, model, root_text, sections):
    """sections: list of (seq_id, text). Returns the schema dict."""
    all_ids = [0] + [s for s, _ in sections]
    tokens, positions, seq_ids, node_type, parent = [], [], [], [], []

    root = tokenize(tok_bin, model, root_text, add_bos=True)
    for p, t in enumerate(root):
        tokens.append(t); positions.append(p); seq_ids.append(list(all_ids))
        node_type.append("document"); parent.append(-1)
    start = len(root)

    for sid, text in sections:
        toks = tokenize(tok_bin, model, text, add_bos=False)
        for k, t in enumerate(toks):
            tokens.append(t); positions.append(start + k); seq_ids.append([sid])
            node_type.append("section"); parent.append(0)

    return dict(tokens=tokens, positions=positions, seq_ids=seq_ids,
                node_type=node_type, parent=parent)


def stats(name, d):
    per = {}
    for ids in d["seq_ids"]:
        per[ids[0]] = per.get(ids[0], 0) + 1
    print(f"{name}: {len(d['tokens'])} tokens, tokens per own-subtree = {per}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="build/bin/llama-tokenize")
    ap.add_argument("--out-dir", default="tests/tree")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    root = "Document: short facts about countries.\n\n"
    sec_a = "Section A. The capital of France is Paris. France is a country in Europe.\n\n"
    sec_b = "Section B. The capital of Japan is Tokyo. Japan is an island country in Asia.\n\n"
    sec_b2 = ("Section B. Brazil is the largest country in South America, and its capital "
              "city is Brasilia, which was built in the 1950s.\n\n")          # different text AND length
    sec_c = "Section C. The Nile is a long river in Africa that flows north into the sea.\n\n"

    docs = {
        "doc_orig.json":     build(a.tokenizer, a.model, root, [(1, sec_a), (2, sec_b),  (3, sec_c)]),
        "doc_changedB.json": build(a.tokenizer, a.model, root, [(1, sec_a), (2, sec_b2), (3, sec_c)]),
    }
    for fn, d in docs.items():
        with open(os.path.join(a.out_dir, fn), "w") as f:
            json.dump(d, f)
        stats(fn, d)
    print("written to", a.out_dir)


if __name__ == "__main__":
    main()

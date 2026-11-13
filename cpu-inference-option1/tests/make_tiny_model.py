#!/usr/bin/env python3
"""
Create a tiny random-weight Llama GGUF (Q8_0 matrices) for smoke-testing the toolchain.
Uses the vocab-only GGUF that ships inside the llama.cpp checkout (models/ggml-vocab-llama-spm.gguf).
The model is gibberish by design: it only has to run, not to be good.

    python tests/make_tiny_model.py
    CPU_PROJECT_TEST=1 python scripts/06_bench.py --quick
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType  # noqa: E402
from gguf.quants import quantize  # noqa: E402

VOCAB = config.LLAMA_DIR / "models" / "ggml-vocab-llama-spm.gguf"
OUT = config.MODELS_DIR / "tiny-test-Q8_0.gguf"
N_EMBD, N_LAYER, N_HEAD, N_KV, N_FF = 128, 4, 8, 4, 256
HEAD = N_EMBD // N_HEAD


def main():
    if not VOCAB.exists():
        sys.exit(f"{VOCAB} not found - build/clone llama.cpp first (scripts/00_build_llamacpp.py)")
    rv = GGUFReader(str(VOCAB))
    n_vocab = len(rv.fields["tokenizer.ggml.tokens"].contents())
    rng = np.random.default_rng(0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(OUT), "llama")
    w.add_name("tiny-test")
    w.add_context_length(2048)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_KV)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(HEAD)
    w.add_vocab_size(n_vocab)
    w.add_file_type(7)                                   # MOSTLY_Q8_0
    for key, f in rv.fields.items():
        if key.startswith("tokenizer."):
            vals = f.contents()
            if f.types[0] == GGUFValueType.ARRAY:
                w.add_key_value(key, list(vals), GGUFValueType.ARRAY, sub_type=f.types[-1])
            else:
                w.add_key_value(key, vals, f.types[0])

    def mat(name, rows, cols):
        a = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
        w.add_tensor(name, quantize(a, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)

    def vec(name, n):
        w.add_tensor(name, np.ones(n, dtype=np.float32))

    mat("token_embd.weight", n_vocab, N_EMBD)            # tied: no output.weight
    vec("output_norm.weight", N_EMBD)
    for i in range(N_LAYER):
        vec(f"blk.{i}.attn_norm.weight", N_EMBD)
        mat(f"blk.{i}.attn_q.weight", N_EMBD, N_EMBD)
        mat(f"blk.{i}.attn_k.weight", N_KV * HEAD, N_EMBD)
        mat(f"blk.{i}.attn_v.weight", N_KV * HEAD, N_EMBD)
        mat(f"blk.{i}.attn_output.weight", N_EMBD, N_EMBD)
        vec(f"blk.{i}.ffn_norm.weight", N_EMBD)
        mat(f"blk.{i}.ffn_gate.weight", N_FF, N_EMBD)
        mat(f"blk.{i}.ffn_up.weight", N_FF, N_EMBD)
        mat(f"blk.{i}.ffn_down.weight", N_EMBD, N_FF)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

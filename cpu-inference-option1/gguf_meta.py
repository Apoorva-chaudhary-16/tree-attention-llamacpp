"""Read the numbers the roofline model needs straight from a GGUF file (no llama.cpp required)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict


def _val(field):
    if field is None:
        return None
    try:
        v = field.contents() if hasattr(field, "contents") else field.parts[field.data[0]][0]
        return v.item() if hasattr(v, "item") else v
    except Exception:
        return None


def read_meta(path: Path) -> Dict:
    from gguf import GGUFReader  # lazy: only roofline/get_models need it

    r = GGUFReader(str(path))
    arch = _val(r.fields.get("general.architecture"))
    g = lambda k: _val(r.fields.get(f"{arch}.{k}"))
    n_embd, n_head = g("embedding_length"), g("attention.head_count")
    n_head_kv = g("attention.head_count_kv") or n_head
    head_dim = g("attention.key_length") or (n_embd // n_head)
    val_dim = g("attention.value_length") or head_dim
    tensors = {t.name: dict(n_bytes=int(t.n_bytes), shape=[int(x) for x in t.shape],
                            type=str(t.tensor_type.name)) for t in r.tensors}
    emb = tensors.get("token_embd.weight")
    n_vocab = emb["shape"][-1] if emb else None            # gguf shape = [n_embd, n_vocab]
    if emb and emb["shape"][0] != n_embd and emb["shape"][-1] == n_embd:
        n_vocab = emb["shape"][0]
    tied = "output.weight" not in tensors
    return dict(arch=arch, n_layer=g("block_count"), n_head=n_head, n_head_kv=n_head_kv, n_embd=n_embd,
                head_dim_k=head_dim, head_dim_v=val_dim, ctx_train=g("context_length"), n_vocab=n_vocab,
                tied_embeddings=tied, tensors=tensors,
                file_bytes=Path(path).stat().st_size)


def bytes_read_per_token(meta: Dict) -> Dict:
    """
    Bytes of WEIGHTS that one decode step must stream from RAM.
      * every matrix/vector except the token-embedding table is read once per token
      * the embedding table is read in full only when it doubles as the output projection (tied
        embeddings, true for Llama-3.2); otherwise only one row is gathered
    """
    t = meta["tensors"]
    total = sum(v["n_bytes"] for v in t.values())
    emb = t.get("token_embd.weight")
    if emb and not meta["tied_embeddings"]:
        total = total - emb["n_bytes"] + emb["n_bytes"] // meta["n_vocab"]
    return dict(weights_total_bytes=sum(v["n_bytes"] for v in t.values()), weights_read_bytes=total,
                embedding_bytes=emb["n_bytes"] if emb else 0)


KV_BYTES_PER_ELEM = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
                     "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}


def kv_bytes_per_token(meta: Dict, k_type="f16", v_type="f16") -> float:
    """KV-cache bytes added per context token (== bytes read per token of context at each decode step)."""
    return meta["n_layer"] * meta["n_head_kv"] * (meta["head_dim_k"] * KV_BYTES_PER_ELEM[k_type]
                                                  + meta["head_dim_v"] * KV_BYTES_PER_ELEM[v_type])

# Tree-structured sequence ids: JSON schema (v1)

One JSON object per document. The C++ driver reads the first five arrays; everything else is for tests and statistics.

| field | type | meaning |
|---|---|---|
| `tokens` | int[n] | token ids (from `llama-tokenize`, includes BOS as the first token) |
| `positions` | int[n] | position of each token (RoPE position and KV position) |
| `seq_ids` | int[n][k] | sequence ids of each token. **First id = the token's own subtree.** Further ids = every descendant subtree that may also see this token |
| `node_type` | string[n] | block type of the token: `document`, `heading`, `paragraph`, `list`, `code`, `table`, `quote`, `rule`, `blank` |
| `parent` | int[n] | sequence id of the parent subtree (`-1` for the root) |
| `is_markup` | 0/1[n] | 1 if the token is pure syntax (`#`, bullets, code fences, table separators, blank lines) |
| `nodes` | object[] | one entry per sequence: `sid, title, level, parent_sid, depth, start, own_tokens, subtree_tokens, contains` |
| `meta` | object | `doc_title, min_tokens_used, max_heading_depth, n_sections_before_merge` |

## Rules

1. **Visibility.** Token *i* attends to cell *j* iff the first id of *i* appears in `seq_ids[j]` and `pos[j] <= pos[i]`
   (this is how llama.cpp builds its mask). A section therefore sees its ancestors and itself, never its siblings.
2. **Positions.** A node starts right after its parent's own tokens, so siblings start at the same position.
   Within every sequence id the positions are exactly `0..m-1` (no gaps, no duplicates). The parser checks this.
3. **Sequence ids.** Root = 0. Other kept nodes are numbered 1, 2, ... in document order. At most 64 ids (`n_seq_max`).
4. **Merge rule.** A subtree with fewer than `min_tokens` tokens (default 24) is appended to the nearest kept ancestor.
   If more than 63 sequences remain, `min_tokens` is raised by 1.5x until they fit. The merged section titles are
   listed in the host node's `contains`.
5. **KV cache.** The driver must create the context with `kv_unified = true` because shared ancestor tokens carry several ids.
6. **Eviction.** To evict a subtree, call `seq_rm` on the subtree's own id **and all its descendants' ids** with
   `p0 = start` of the subtree and `p1 = -1`. Ancestor tokens (positions below `start`) stay.

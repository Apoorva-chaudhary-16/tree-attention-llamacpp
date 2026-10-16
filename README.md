# Tree-Structured Attention on llama.cpp (CPU) - Week 1 Report

**Domain:** Markdown documents (parser: `markdown-it-py`)
**Model:** Llama-3.2-1B-Instruct, Q4_K_M, CPU only (4 threads)
**llama.cpp commit used:** `ace20f679e30a39c09a7b17bec14199855469633`
**Author:** Apoorva Chaudhary (MT25016), IIIT Delhi. Individual submission; parts of the design were discussed with my team.

---

## 1. Idea

A long structured document is mostly layout, and a question usually depends on one part of it. The idea of this project is to
parse the document into a tree, give each subtree its own **sequence id**, and let a token attend only to tokens that share its
sequence id. A section then sees its ancestors and itself, never its siblings. Because llama.cpp already supports sequence ids
(for batching several prompts), this needs no change to the model code: only the way we label tokens and the KV cache calls.

Two properties follow, and they are what the tests check:

1. **Isolation:** changing an unrelated section B must not change the answer about section A.
2. **Eviction:** an unrelated subtree can be removed from the KV cache (by sequence id and position range) without changing the
   answer, and the KV memory shrinks.

## 2. What I did

1. **Environment.** Built llama.cpp for CPU in WSL (Ubuntu), downloaded `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, and ran it with
   `llama-cli`. Pinned the llama.cpp commit (above).
2. **Batched example.** Ran `llama-batched` with two parallel sequences (`-np 2`). With the default per-sequence KV cache it failed with
   `split_equal: sequential split is not supported when there are coupled sequences`. A "coupled" token belongs to several
   sequences at once, which is exactly what shared ancestors need, so the fix was the **unified KV cache** (`-kvu`, and
   `kv_unified = true` in the context params). My driver always sets it.
3. **Mask rule.** Token *i* may attend to cell *j* if the sequence id of *i* is among the ids of *j* and `pos(j) <= pos(i)`.
   I rely on this rule when I assign ids: the first id of a token is its own subtree, extra ids list the descendants that may see it.
4. **Schema** (`schema/SCHEMA.md`): `tokens`, `positions`, `seq_ids`, `node_type`, `parent` (plus `is_markup`, `nodes`, `meta`
   for tests and statistics), with rules for positions, visibility, merging and eviction.
5. **Parser** (`parsers/markdown/parse_markdown.py`): heading sections become tree nodes. Token ids come from `llama-tokenize`,
   so they match the model exactly. Positions are validated to be gap-free for every sequence id.
6. **C++ driver** (`examples/treedriver/treedriver.cpp`, based on `examples/batched`): loads the JSON, prefills with the
   structured sequence ids in chunks, answers a question with greedy decoding using only the asked subtree's id, and evicts
   subtrees with `llama_memory_seq_rm` over a position range.
7. **Tests and statistics** (`tests/tree/run_tests.py`, `tests/tree/stats.py`) on five Markdown documents under 400 tokens each.
8. **One-slide report** (tree statistics, test results, CPU times), submitted separately.

## 3. Design decisions

- **One sequence id per heading section.** Root = 0, other kept sections are numbered 1, 2, ... in document order.
  A token of section N carries `[N, ids of all descendants of N]`, so children can see their parent but not the other way round.
- **Positions.** A section starts right after its parent's own tokens, so sibling sections start at the same position.
  This matters: if B's length changed the positions of A, A's computation would change even without any attention to B.
  Positions are consecutive within every sequence id (checked by the parser).
- **Merge rule for small subtrees.** llama.cpp allows a limited number of sequences (the driver uses `n_seq_max = 64`).
  A subtree with fewer than 24 tokens is merged into the nearest kept ancestor. If more than 63 sequences remain, the threshold is
  raised by 1.5x until they fit. Merged titles are listed in the host node's `contains` field.
- **Question and answer tokens** get only the asked section's id and are removed from the cache after answering, so the same
  document can be queried repeatedly on a clean state.
- **Eviction of a subtree** removes the section's id and all descendant ids over `[start, end)` of the section. Ancestor tokens (below `start`)
  stay in the cache.
- **Greedy decoding** (temperature 0) so "answer unchanged" is an exact comparison.

## 4. Results

### 4.1 Tree statistics (Markdown, 5 documents)

| document | tokens | depth | sequences | subtree tokens min / median / max | pure markup tokens | attention pairs saved |
|---|---|---|---|---|---|---|
| fluxcache_install.md | 397 | 3 | 13 | 27 / 52 / 389 | 16.6% | 66.5% |
| orbit_api.md | 347 | 3 | 10 | 31 / 51 / 340 | 17.0% | 65.9% |
| meeting_notes.md | 295 | 3 | 8 | 24 / 42 / 284 | 15.3% | 46.4% |
| lisbon_guide.md | 300 | 3 | 9 | 26 / 52 / 292 | 14.0% | 61.3% |
| textutils_module.md | 279 | 3 | 8 | 38 / 52 / 272 | 15.1% | 38.3% |
| **average** | 324 | 3 | 9.6 | | **15.6%** | **55.7%** |

- *Depth* = number of heading levels below the document root.
- *Pure markup tokens* = tokens of syntax with no content of their own: heading `#` marks, list bullets, quote marks, code fences,
  table separator rows and blank lines.
- *Attention pairs saved* = 1 - (query-key pairs allowed by the tree mask) / (pairs under ordinary full causal attention).
  This is a **count**, not a measured speed-up (see limitations).

### 4.2 Tests

For each document I asked a question about section A and used a different, unrelated section B for the changes.

- **Test 1 (change B, ask about A):** B's body was replaced by different, longer text. The greedy answer about A had to stay identical.
- **Control:** asking about B itself on the original and changed document had to give *different* answers (so the test can fail).
- **Test 2 (evict B):** B's whole subtree was removed with `seq_rm`; the answer about A had to stay the same and the KV state had to shrink.

| document | A (asked) | B (changed / evicted) | answer correct | Test 1 | control | Test 2 | tokens evicted | KV state before -> after | KV freed |
|---|---|---|---|---|---|---|---|---|---|
| fluxcache_install | Configuration | Using the package manager | yes | PASS (exact) | differs | PASS (answer line) | 53 | 13,016,245 -> 11,278,905 B | 13.3% |
| orbit_api | Getting a token | Create a ground station | yes | PASS (exact) | differs | PASS (exact) | 66 | 11,376,421 -> 9,212,941 B | 19.0% |
| meeting_notes | Budget review | Action items | yes | PASS (exact) | differs | PASS (answer line) | 70 | 9,672,277 -> 7,377,677 B | 23.7% |
| lisbon_guide | By air | Where to stay | yes | PASS (exact) | differs | PASS (exact) | 54 | 9,835,673 -> 8,065,553 B | 18.0% |
| textutils_module | truncate | slugify | yes | PASS (exact) | differs | PASS (exact) | 67 | 9,147,921 -> 6,951,661 B | 24.0% |

- Test 1 passes exactly on 5 of 5 documents, and the control differs on 5 of 5.
- Test 2: the answer line is unchanged on 5 of 5 documents; the full 16-token output is identical on 3 of 5.
  On `fluxcache_install` and `meeting_notes` the answer itself (e.g. `7421`, `1,800 dollars per month.`) was the same, but the text
  generated after it differed once the section was evicted. I have not investigated the cause. A plausible explanation is small numerical
  differences from a different KV cache layout, which can flip a late token, but I did not verify this.
- The KV figure is the size of the saved context state (`llama_state_get_size`), which is dominated by the KV cache; it is not an exact
  byte count of the cache.

### 4.3 CPU run times (4 threads)

| document | prefill | one 16-token answer |
|---|---|---|
| fluxcache_install | 2.05 s | 0.55 s |
| orbit_api | 2.56 s | 0.63 s |
| meeting_notes | 2.38 s | 0.65 s |
| lisbon_guide | 2.51 s | 0.63 s |
| textutils_module | 2.14 s | 0.70 s |

## 5. Problems met and how they were solved

| problem | cause | fix |
|---|---|---|
| `apt install cmake` returned 403 | blocked Ubuntu mirror | switched mirror / alternative install |
| `pip install` failed with `externally-managed-environment` | Ubuntu protects system Python | used a virtual environment |
| `split_equal: sequential split is not supported when there are coupled sequences` | tokens with several sequence ids need one shared KV buffer | `kv_unified = true` (`-kvu`) |
| batched example output differs between sequences | default sampler is random | greedy sampler in the driver |
| risk of A's answer changing when B's length changes | positions of later sections would shift | siblings start at the same position |
| two evict tests differed only after the answer line | see 4.2 | compare the answer line and show both full outputs |

## 6. Limitations

- The five documents are small (279 to 397 tokens) and the model is a 1B model with greedy decoding and short answers, so the results show
  that the mechanism works, not how it behaves on long documents.
- "Attention pairs saved" is a count. llama.cpp still computes the masked entries, so run time does not improve yet; the saving in practice is the
  KV memory that eviction frees.
- The tree-structured mask is checked through answers, not by inspecting the mask tensor itself.
- Cross-references and non-tree structure (tables as grids, citations) are not modelled; Markdown tables are treated as ordinary text blocks.
- Sections under 24 tokens are merged into their parent, so very fine structure is lost.
- Only one domain (Markdown) is implemented here; other domains need their own parser that writes the same schema.

## 7. Team collaboration

[Fill in: team members, which domain each member took, what we agreed on together (schema fields, merge rule), and what I did myself.]

---

# Repository

## Layout
| path | what |
|---|---|
| `examples/treedriver/treedriver.cpp` | C++ driver (copy of `batched`): load JSON, prefill with structured seq ids, answer a question, evict subtrees |
| `examples/treedriver/CMakeLists.txt` | build file for the driver |
| `schema/SCHEMA.md` | JSON schema and the visibility / position / merge / eviction rules |
| `parsers/markdown/parse_markdown.py` | Markdown -> JSON parser |
| `parsers/markdown/samples/` | 5 sample documents and `index.json` (one question per document) |
| `tests/tree/run_tests.py` | runs Test 1, control and Test 2 on all samples, writes `tests/tree/out/results.md` and `results.csv` |
| `tests/tree/stats.py` | tree statistics |
| `tests/tree/out/` | generated JSON files and result tables from my run |

Other domains can be added as `parsers/<domain>/parse_<domain>.py` that write the same JSON schema; the driver and the runner stay unchanged.

## Setup
```bash
# 1. get llama.cpp and check out the commit above
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp && git checkout ace20f679e30a39c09a7b17bec14199855469633

# 2. add the driver: copy this repo's examples/treedriver into llama.cpp/examples
cp -r <this-repo>/examples/treedriver examples/
sed -i 's/^\(\s*\)add_subdirectory(batched)$/&\n\1add_subdirectory(treedriver)/' examples/CMakeLists.txt

# 3. build (CPU)
cmake -B build
cmake --build build --config Release -j 4 --target llama-treedriver llama-tokenize

# 4. python dependency (inside a venv) and the model
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install markdown-it-py huggingface_hub
hf download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir models

# 5. copy parsers/, schema/ and tests/ from this repo into ~/llama.cpp
```

## Run everything
```bash
cd ~/llama.cpp
python3 tests/tree/run_tests.py --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
python3 tests/tree/stats.py tests/tree/out/fluxcache_install.json tests/tree/out/orbit_api.json \
    tests/tree/out/meeting_notes.json tests/tree/out/lisbon_guide.json tests/tree/out/textutils_module.json
```

## Run one document by hand
```bash
python3 parsers/markdown/parse_markdown.py parsers/markdown/samples/orbit_api.md -o /tmp/orbit.json \
    --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
./build/bin/llama-treedriver -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --doc /tmp/orbit.json \
    --ask-seq 3 -q "How long is an Orbit API token valid?" --evict-seq 7 --evict-p0 <start of that subtree>
```
Section ids and start positions are listed in the JSON under `nodes`. `--evict-seq` takes a comma-separated list (a section and its descendants).

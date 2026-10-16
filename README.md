# Tree-Structured Attention on llama.cpp (CPU) - Week 1 Report

| | |
|---|---|
| **GitHub repository** | https://github.com/Apoorva-chaudhary-16/tree-attention-llamacpp |
| **Domain** | Markdown documents (parser: `markdown-it-py`) |
| **Model** | Llama-3.2-1B-Instruct, Q4_K_M (GGUF), CPU only, 4 threads |
| **llama.cpp commit** | `ace20f679e30a39c09a7b17bec14199855469633` |
| **Team members** | Apoorva Chaudhary (MT25016), Harsh Sharma (MT25023), Vedika Jain (MT25051), IIIT Delhi. |

---

## Table of contents

1. [Idea](#1-idea)
2. [How it works, step by step](#2-how-it-works-step-by-step)
3. [Technology used, why it was chosen, and the alternatives](#3-technology-used-why-it-was-chosen-and-the-alternatives)
4. [What was done (week-1 checklist)](#4-what-was-done-week-1-checklist)
5. [Design decisions](#5-design-decisions)
6. [Results](#6-results)
7. [Problems met and how they were solved](#7-problems-met-and-how-they-were-solved)
8. [Limitations and future work](#8-limitations-and-future-work)
9. [Team collaboration](#9-team-collaboration)
10. [Repository layout](#10-repository-layout)
11. [Setup](#11-setup)
12. [How to run](#12-how-to-run)
13. [Troubleshooting](#13-troubleshooting)
14. [Adding another domain](#14-adding-another-domain)

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

---

## 2. How it works, step by step

```
 doc.md --(1) split into heading sections--> tree of nodes
        --(2) tokenize every line with llama-tokenize--> tokens + is_markup + block type
        --(3) fold tiny subtrees, assign sequence ids and positions--> doc.json
        --(4) llama-treedriver: prefill with structured seq ids--> KV cache
        --(5) ask a question with ONE seq id (greedy)--> ANSWER1
        --(6) seq_rm the subtree of B (own id + descendant ids, from its start position)--> smaller KV
        --(7) ask again--> ANSWER2   (must equal ANSWER1)
```

### Step 1 - Parse the Markdown into a section tree (`parse_markdown.py`)
`markdown-it-py` parses the text and reports every heading with its level and line range. A stack of open headings turns
the flat list into a tree: `ROOT -> # h1 -> ## h2 -> ### h3`. The text before the first heading, plus an injected line
`Document: <title>`, belongs to the root. Using a real Markdown parser (and not a regex on `#`) means a `#` inside a code
block is never mistaken for a heading.

### Step 2 - Tokenize each line and mark pure syntax
Each section is cut into small segments (`segment_lines`): heading marks (`# `), list bullets, quote marks, code fences,
table separator rows and blank lines are marked `is_markup = 1`; everything else is content. Every segment gets a block type
(`heading`, `paragraph`, `list`, `code`, `table`, `quote`, `rule`, `blank`, `document`). Token ids come from the real
`llama-tokenize` binary of the same model, so they match what the model would see. BOS (`128000`) is added once at the
very start.

### Step 3 - Merge small subtrees, assign sequence ids and positions
- A subtree with fewer than **24 tokens** is merged into the nearest kept ancestor (its text is appended to that node, and its title is
  listed in the host's `contains`). If more than 63 sequences remain, the threshold is multiplied by 1.5 until they fit
  (llama.cpp context is created with `n_seq_max = 64`).
- Sequence ids: root = 0, the other kept sections 1, 2, ... in document order.
- A token of node N gets `seq_ids = [N, ids of all kept descendants of N]`. The **first id is the token's own subtree**
  (this is the id used when the token acts as a query); the extra ids say "these descendant subtrees may also look at me".
- Positions: a node starts right after its parent's own tokens, so **siblings start at the same position**.
  The parser checks that positions are exactly `0..m-1` for every sequence id.

Worked example (`lisbon_guide.md`, 300 tokens, 9 sequences):

```
seq 0  (document)            pos   0..7     ids [0..8]   every section can see it
seq 1  Travel Guide: Lisbon  pos   8..35    ids [1..8]
 |- seq 2  Getting there     pos  36..61    ids [2,3]    (also holds the merged "By train")
 |   '- seq 3  By air        pos  62..102   ids [3]
 |- seq 4  Where to stay     pos  36..89    ids [4]
 |- seq 5  Food              pos  36..85    ids [5]
 |- seq 6  Day trips         pos  36..63    ids [6,7]    (also holds the merged "Cascais")
 |   '- seq 7  Sintra        pos  64..89    ids [7]
 '- seq 8  Practical tips    pos  36..74    ids [8]
```

A question asked with seq id 3 ("By air") can attend to cells that carry id 3: the document, the title section, the
"Getting there" intro and the "By air" text. It cannot attend to "Where to stay", "Food", etc., because those cells do not
carry id 3 (and their positions overlap with the siblings, so the position-order rule alone would not be enough to hide them).

### Step 4 - Prefill in C++ (`treedriver.cpp`)
The driver loads the JSON (`tokens`, `positions`, `seq_ids`), creates a context with `kv_unified = true` and
`n_seq_max = 64`, and feeds the tokens to `llama_decode` in chunks of 512, giving every token its list of sequence ids and its
explicit position. llama.cpp builds the attention mask itself from these ids (token *i* sees cell *j* if the first id of *i* is in the id list
of *j* and `pos(j) <= pos(i)`).

### Step 5 - Ask a question about one subtree
The question `"\n\nQuestion: ...\nAnswer:"` is tokenized and decoded with **only** the asked section's id, then up to `-n` tokens are
generated with the greedy sampler. Afterwards the question and answer cells are removed again (`llama_memory_seq_rm(seq, start, -1)`),
so the same cache can be queried many times.

### Step 6 - Evict a subtree
`--evict-seq` takes the subtree's own id **and all its descendants' ids**, and `--evict-p0` its start position. The driver calls
`llama_memory_seq_rm(mem, id, p0, -1)` for each of them. Shared ancestor cells (positions below `p0`) are untouched. A cell is freed
when the last sequence id on it is removed, which is why all descendant ids must be removed too.

### Step 7 - Compare
`tests/tree/run_tests.py` automates the whole chain on all five sample documents (section 6 explains Test 1, the control and Test 2).

---

## 3. Technology used, why it was chosen, and the alternatives

| Part | Chosen | Why | Other options (and why not used) |
|---|---|---|---|
| Inference engine | **llama.cpp** (pinned commit) | Required by the task, runs well on CPU, and already has sequence ids, a unified KV cache and `seq_rm` by position range, so the idea needs no change to model code. Small C++ code base that can be traced. | *HF Transformers* with a custom 4-D attention mask: very flexible but slow on CPU and no cheap per-subtree KV removal. *vLLM / SGLang (RadixAttention)*: closest ideas (prefix trees, paged KV) but GPU-oriented. *PyTorch FlexAttention*: needs a GPU for speed. *MLC / ONNX Runtime*: harder to change the cache. |
| Model | **Llama-3.2-1B-Instruct, Q4_K_M** | Small enough for a laptop CPU (about 0.8 GB), instruct-tuned so it answers short questions, 4-bit K-quant keeps good quality at small size. | *Qwen2.5-0.5B / Phi-3-mini*: also possible. *Q8_0 or F16*: more accurate but 2-4x bigger and slower. *Q4_0*: slightly smaller but lower quality than Q4_K_M. |
| How to express the tree | **Sequence ids + explicit positions** | Uses a feature that already exists in llama.cpp, so we only produce labels. Shared ancestors are stored once. | *Edit the mask-building code in llama.cpp*: most general (DAGs, grids, cross-references) but means changing C++ internals and rebuilding. *Copy the shared prefix into every sequence (`seq_cp`)*: simpler idea, but one logical copy per sequence. *One sequence per subtree and re-prefill ancestors*: wastes compute. |
| KV cache mode | **Unified KV cache** (`kv_unified = true`, `-kvu`) | A token with several sequence ids ("coupled" sequences) needs one shared buffer. The default per-sequence cache fails with `split_equal: sequential split is not supported when there are coupled sequences`. | There is no real alternative if shared ancestor cells are wanted. |
| Positions | **Siblings start at the same position** | The answer about A must not depend on how long B is. Same start position means A's RoPE positions are the same in the original and the changed document. | *Continuous document-order positions*: natural, but changing B's length would shift every later section and change A's computation even without any attention to B. (Tree-attention for speculative decoding also gives sibling branches the same depth-based position.) |
| Markdown parser | **markdown-it-py** (CommonMark + tables) | Listed in the task, spec-compliant, gives each heading with its source line range, never confuses `#` in code blocks. | *mistune, python-markdown, commonmark.py*: also fine. *Regex on `^#+`*: simplest but breaks on code fences. *pandoc AST / remark (JS)*: more complete but extra dependency. |
| Tokenizer | **`llama-tokenize`** from the same build | Exactly the vocabulary and special tokens of the GGUF model; no mismatch between Python and C++. | *HF `tokenizers` / `transformers`*: needs the original tokenizer files and may differ in details. *llama-cpp-python*: extra package, and it would not follow the pinned commit. |
| Interface parser <-> driver | **JSON file** (parallel arrays), read with `nlohmann/json` | Human-readable, every language can write it, so every team's parser (HTML, LaTeX, code, ...) feeds the same driver. `nlohmann/json` ships with llama.cpp. | *Protobuf / MessagePack*: smaller and faster but not needed for documents under 2k tokens and harder to inspect. *A custom text format*: easy to get wrong. *Parsing inside C++*: would tie the driver to one domain. |
| Decoding | **Greedy (temperature 0)** | Deterministic, so "answer unchanged" is an exact string comparison. | *Sampling with a fixed seed*: also reproducible but less direct. *Compare logits* (max absolute difference): stronger test, listed under future work. |
| Test oracle | **Answer text + control** | Cheap, shows the end-to-end effect, and the control proves the test can fail. | *Inspect the mask tensor or hidden states*: more direct, needs changes in llama.cpp. |
| Glue / tests | **Python 3 scripts** calling the binaries with `subprocess` | Fast to write and easy to read. | *C++ tests / CTest*: heavier. *pytest*: a good next step to wrap the same functions. |
| Build / OS | **CMake in WSL2 Ubuntu** | Official llama.cpp build system; Linux toolchain is the least painful. | *Native Windows (MSVC)*, *Docker*, *macOS*: also work. |

---

## 4. What was done (week-1 checklist)

| Task from the project statement | Status | Where |
|---|---|---|
| Build CPU llama.cpp, run Llama-3.2-1B Q4_K_M | done (WSL Ubuntu, commit pinned above) | section 11 |
| Read `llama.h`, trace the batched example | done; found the coupled-sequence / `kv_unified` issue | section 5 and 7 |
| Trace mask construction from sequence ids | done; rule written in section 5 and `schema/SCHEMA.md` | `schema/SCHEMA.md` |
| Two prompts in two sequences vs separate runs | `llama-batched -np 2` run with `-kvu`, greedy sampler used for comparable output | section 7 |
| JSON schema (`tokens, positions, seq_ids, node_type, parent`) + merge rules | done, plus `is_markup`, `nodes`, `meta` | `schema/SCHEMA.md` |
| Parser, 5 documents under 2k tokens | done for Markdown (279-397 tokens each) | `parsers/markdown/` |
| Shared C++ driver (load JSON, prefill, answer, evict by position range) | done, based on `examples/batched` | `examples/treedriver/` |
| Test: change B, ask about A, output unchanged | done (Test 1 + control) | `tests/tree/run_tests.py` |
| Test: remove an unrelated subtree, answer holds, KV drops | done (Test 2) | `tests/tree/run_tests.py` |
| One slide: domain, tree statistics, test results, CPU times | submitted separately; numbers in section 6 | `tests/tree/stats.py`, `out/results.*` |

---

## 5. Design decisions

- **One sequence id per heading section.** Root = 0, other kept sections are numbered 1, 2, ... in document order.
  A token of section N carries `[N, ids of all descendants of N]`, so children can see their parent but not the other way round.
- **Mask rule.** Token *i* may attend to cell *j* if the sequence id of *i* is among the ids of *j* and `pos(j) <= pos(i)`.
  The first id of a token is its own subtree; extra ids list the descendants that may see it.
- **Positions.** A section starts right after its parent's own tokens, so sibling sections start at the same position.
  This matters: if B's length changed the positions of A, A's computation would change even without any attention to B.
  Positions are consecutive within every sequence id (checked by the parser).
- **Merge rule for small subtrees.** llama.cpp allows a limited number of sequences (the driver uses `n_seq_max = 64`).
  A subtree with fewer than 24 tokens is merged into the nearest kept ancestor. If more than 63 sequences remain, the threshold is
  raised by 1.5x until they fit. Merged titles are listed in the host node's `contains` field.
- **Question and answer tokens** get only the asked section's id and are removed from the cache after answering, so the same
  document can be queried repeatedly on a clean state.
- **Eviction of a subtree** removes the section's id and all descendant ids from the section's start position to the end (`p1 = -1`).
  Ancestor tokens (below `start`) stay in the cache.
- **Greedy decoding** (temperature 0) so "answer unchanged" is an exact comparison.

---

## 6. Results

### 6.1 Tree statistics (Markdown, 5 documents)

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

### 6.2 Tests

For each document one question about section A is asked (`parsers/markdown/samples/index.json`), and a different section B
that is not an ancestor or descendant of A is used for the changes (the largest such section).

- **Test 1 (change B, ask about A):** B's body is replaced by scrambled text (ROT13) plus an extra filler sentence, so both its content and its length change.
  The greedy answer about A had to stay identical (the parser is re-run, so all positions and ids are recomputed).
- **Control:** ask `"What is the last sentence of this section?"` about B itself on the original and the changed document; the answers had to be
  *different* (this proves the change is visible to B and the test can fail).
- **Test 2 (evict B):** B's whole subtree is removed with `seq_rm`; the answer about A had to stay the same and the KV state had to shrink.

| document | A (asked) | B (changed / evicted) | answer correct | Test 1 | control | Test 2 | tokens evicted | KV state before -> after | KV freed |
|---|---|---|---|---|---|---|---|---|---|
| fluxcache_install | Configuration | Using the package manager | yes | PASS (exact) | differs | PASS (answer line) | 53 | 13,016,245 -> 11,278,905 B | 13.3% |
| orbit_api | Getting a token | Create a ground station | yes | PASS (exact) | differs | PASS (exact) | 66 | 11,376,421 -> 9,212,941 B | 19.0% |
| meeting_notes | Budget review | Action items | yes | PASS (exact) | differs | PASS (answer line) | 70 | 9,672,277 -> 7,377,677 B | 23.7% |
| lisbon_guide | By air | Where to stay | yes | PASS (exact) | differs | PASS (exact) | 54 | 9,835,673 -> 8,065,553 B | 18.0% |
| textutils_module | truncate | slugify | yes | PASS (exact) | differs | PASS (exact) | 67 | 9,147,921 -> 6,951,661 B | 24.0% |

"PASS (answer line)" is the label stored in `results.md`; the current `run_tests.py` prints the same verdict as "PASS (through answer)":
the generated text is identical up to and including the expected value.

- Test 1 passes exactly on 5 of 5 documents, and the control differs on 5 of 5.
- Test 2: the answer is unchanged on 5 of 5 documents; the full 16-token output is identical on 3 of 5.
  On `fluxcache_install` and `meeting_notes` the answer itself (e.g. `7421`, `1,800 dollars per month.`) was the same, but the text
  generated after it differed once the section was evicted. I have not investigated the cause. A plausible explanation is small numerical
  differences from a different KV cache layout, which can flip a late token, but I did not verify this.
- The KV figure is the size of the saved context state (`llama_state_get_size`), which is dominated by the KV cache; it is not an exact
  byte count of the cache. As a sanity check: Llama-3.2-1B stores 16 layers x 8 KV heads x 64 dims x 2 (K and V) x 2 bytes (fp16) = 32,768 B per token,
  and the observed drops match "tokens evicted x 32,768 B" closely (e.g. lisbon: 54 x 32,768 = 1,769,472 B vs 1,770,120 B measured).

### 6.3 CPU run times (4 threads)

| document | prefill | one 16-token answer |
|---|---|---|
| fluxcache_install | 2.05 s | 0.55 s |
| orbit_api | 2.56 s | 0.63 s |
| meeting_notes | 2.38 s | 0.65 s |
| lisbon_guide | 2.51 s | 0.63 s |
| textutils_module | 2.14 s | 0.70 s |

---

## 7. Problems met and how they were solved

| problem | cause | fix |
|---|---|---|
| `apt install cmake` returned 403 | blocked Ubuntu mirror | switched mirror / alternative install |
| `pip install` failed with `externally-managed-environment` | Ubuntu protects system Python | used a virtual environment |
| `split_equal: sequential split is not supported when there are coupled sequences` | tokens with several sequence ids need one shared KV buffer | `kv_unified = true` (`-kvu`) |
| batched example output differs between sequences | default sampler is random | greedy sampler in the driver |
| risk of A's answer changing when B's length changes | positions of later sections would shift | siblings start at the same position |
| two evict tests differed only after the answer line | see 6.2 | compare up to the answer and show both full outputs |

---

## 8. Limitations and future work

- The five documents are small (279 to 397 tokens) and the model is a 1B model with greedy decoding and short answers, so the results show
  that the mechanism works, not how it behaves on long documents.
- "Attention pairs saved" is a count. llama.cpp still computes the masked entries, so run time does not improve yet; the saving in practice is the
  KV memory that eviction frees.
- The tree-structured mask is checked through answers, not by inspecting the mask tensor or the logits. Comparing logits would be a stronger test.
- Each line is tokenized separately (for example `#` and the following space are two tokens), so the token stream is slightly different from tokenizing the
  whole file in one go. The model still answers correctly, but quality on other documents may differ.
- Cross-references and non-tree structure (tables as grids, citations) are not modelled; Markdown tables are treated as ordinary text blocks.
- Sections under 24 tokens are merged into their parent, so very fine structure is lost.
- Only one domain (Markdown) is implemented here; other domains need their own parser that writes the same schema.
- Not yet done: longer documents (up to 2k tokens), more questions per document, a larger model, measuring real speed-ups.

---

## 9. Team collaboration

The project was developed around a shared approach to tree-structured attention: representing documents with a common JSON schema, assigning sequence IDs to tree nodes, and testing whether changes to or eviction of an unrelated subtree affect answers about another subtree. During development, a broader working prototype explored Markdown, JSON/API responses, and agent chat logs through a common tree-processing pipeline. The public repository linked from this report presents the final Markdown-domain implementation and its measured results.

| member | primary areas of contribution |
|---|---|
| Apoorva Chaudhary (MT25016) | Led the final Markdown-domain implementation and maintained the report repository, bringing together the Markdown parser, shared schema, driver integration, sample documents, tests, statistics, and reported results. |
| Harsh Sharma (MT25023) | Build and end-to-end validation, including the Windows/PowerShell driver and the five-document test workflow. |
| Vedika Jain (MT25051) | Developed the multi-domain prototype used during early exploration, including the shared tree-processing and layout pipeline, JSON schema, JSON/API and chat-log parsers, C++ driver integration, automated tests, sample documents, and supporting documentation. Its common interface allowed domain-specific parsers to use the same driver and test workflow. |

These contributions complemented one another: the common schema and tree-processing approach supported domain-specific parsers, while build and end-to-end testing informed the final Markdown results. As with most shared engineering work, the design and integration involved iteration and discussion across the team; the table records primary areas of contribution rather than exclusive ownership of every decision.

---

## 10. Repository layout

| path | what |
|---|---|
| `examples/treedriver/treedriver.cpp` | C++ driver (based on `batched`): load JSON, prefill with structured seq ids, answer a question, evict subtrees |
| `examples/treedriver/CMakeLists.txt` | build file for the driver (target `llama-treedriver`) |
| `schema/SCHEMA.md` | JSON schema and the visibility / position / merge / eviction rules |
| `parsers/markdown/parse_markdown.py` | Markdown -> JSON parser (also contains the test helper `perturb`) |
| `parsers/markdown/samples/` | 5 sample documents and `index.json` (question, target section and expected answer for each) |
| `tests/tree/run_tests.py` | runs Test 1, control and Test 2 on all samples, writes `tests/tree/out/results.md` and `results.csv` |
| `tests/tree/stats.py` | tree statistics (depth, subtree sizes, markup share, attention pairs saved) |
| `tests/tree/make_test_docs.py` | builds a tiny hand-made 3-section document (`doc_orig.json`, `doc_changedB.json`) for a quick driver smoke test |
| `tests/tree/out/` | generated JSON files (`<doc>.json`, `<doc>_changedB.json`) and result tables from my run |
| `.gitignore` | ignores `__pycache__`, `*.pyc`, `*.gguf` and `venv/` (the model file is not committed) |

---

## 11. Setup

Tested on Ubuntu (WSL2). You need `git`, `g++`, `cmake` (3.14 or newer), `python3` (3.9 or newer), about 1 GB of free disk space and 4 GB of RAM.

```bash
# 0. clone this project repository (the parser, schema, tests and driver source)
git clone https://github.com/Apoorva-chaudhary-16/tree-attention-llamacpp.git ~/tree-attention-llamacpp

# 1. get llama.cpp and check out the tested baseline commit
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
git checkout ace20f679e30a39c09a7b17bec14199855469633

# 2. copy the custom driver into llama.cpp and register it as an example
cp -r ~/tree-attention-llamacpp/examples/treedriver examples/
sed -i 's/^\(\s*\)add_subdirectory(batched)$/&\n\1add_subdirectory(treedriver)/' examples/CMakeLists.txt

# 3. build (CPU only)
cmake -B build -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF
cmake --build build --config Release -j 4 --target llama-treedriver llama-tokenize

# 4. copy the parser, schema and tests into the llama.cpp working tree
cp -r ~/tree-attention-llamacpp/parsers ~/tree-attention-llamacpp/schema ~/tree-attention-llamacpp/tests ~/llama.cpp/

# 5. install Python dependencies in an isolated environment
python3 -m venv ~/venv && source ~/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install markdown-it-py huggingface_hub

# 6. download the model (it is not stored in the GitHub repository)
mkdir -p models
hf download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir models
```

Check: `./build/bin/llama-tokenize -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --ids -p "hello"` should print a list of token ids.

---

## 12. How to run

All commands are run from the `~/llama.cpp` folder with the venv active.

### 12.1 Everything (parse, Test 1, control, Test 2, statistics, tables)
```bash
python3 tests/tree/run_tests.py --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```
This writes `tests/tree/out/<doc>.json`, `<doc>_changedB.json`, `results.csv` and `results.md`, prints `ALL TESTS PASS` or `SOME TESTS FAILED`,
and exits with code 0 only if every test passed.

Useful options: `--driver`, `--tokenizer`, `--samples`, `--out`, `--min-tokens 24`, `-n 16` (answer length), `-t 4` (threads).

### 12.2 Parser + statistics only (no model run)
```bash
# offline check with a fake word tokenizer (no model needed)
python3 tests/tree/run_tests.py --dry-run --fake-tokenizer

# statistics from already generated JSON files
python3 tests/tree/stats.py tests/tree/out/fluxcache_install.json tests/tree/out/orbit_api.json \
    tests/tree/out/meeting_notes.json tests/tree/out/lisbon_guide.json tests/tree/out/textutils_module.json
```
(With `--dry-run` and the fake tokenizer the numbers are not the real model token counts.)

### 12.3 Run one document by hand
```bash
# parse
python3 parsers/markdown/parse_markdown.py parsers/markdown/samples/orbit_api.md -o /tmp/orbit.json \
    --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# ask about "Getting a token" (seq 3), then evict "Create a ground station" (seq 6, starts at position 39) and ask again
./build/bin/llama-treedriver -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --doc /tmp/orbit.json \
    --ask-seq 3 -q "How long is an Orbit API token valid?" --evict-seq 6 --evict-p0 39
```
Section ids and start positions are listed in the JSON under `nodes`. `--evict-seq` takes a comma-separated list (a section and its descendants).

Driver output lines:

| line | meaning |
|---|---|
| `PREFILL tokens=... seqs=... time=...s` | prefill size and CPU time |
| `KV state_bytes_before=...` / `state_bytes_after=...` | context state size before and after eviction |
| `ANSWER1:` / `ANSWER2:` | greedy answer before / after eviction (newlines shown as spaces) |
| `TIME answer1=...s` | time for the question + generation |
| `EVICT_TEST PASS/FAIL` | `PASS` only if both answers are byte-identical |

Driver flags: `-m`, `--doc`, `--ask-seq`, `-q`, `--evict-seq`, `--evict-p0`, `--evict-p1` (default -1 = to the end), `-n` (default 32), `-t` (default 4).

### 12.4 Quick smoke test with the tiny hand-made document
```bash
python3 tests/tree/make_test_docs.py --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
./build/bin/llama-treedriver -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --doc tests/tree/doc_orig.json \
    --ask-seq 1 -q "What is the capital of France?" --evict-seq 2 --evict-p0 8
```

---

## 13. Troubleshooting

| symptom | likely cause | what to do |
|---|---|---|
| `llama-tokenize failed` / file not found | target not built or wrong `--tokenizer` path | build with `--target llama-tokenize`, or pass `--tokenizer` |
| `split_equal: sequential split is not supported ...` | context not created with unified KV | use the provided driver (sets `kv_unified = true`); for `llama-batched` add `-kvu` |
| `prefill failed at token N (check: positions consecutive per seq id?)` | JSON positions are not gap-free for some id | re-run the parser; it validates this and raises an error |
| `N distinct seq ids > 64` | too many sections | raise `--min-tokens` in the parser (it also does this automatically) |
| `ModuleNotFoundError: markdown_it` | venv not active or package missing | `source ~/venv/bin/activate && pip install markdown-it-py` |
| `externally-managed-environment` | system Python is protected | use a virtual environment |
| answers differ slightly after eviction (text after the answer) | see 6.2 | compare through the answer, as `run_tests.py` does |

---

## 14. Adding another domain

Write `parsers/<domain>/parse_<domain>.py` that outputs the same JSON (see `schema/SCHEMA.md`): `tokens`, `positions`, `seq_ids`
(first id = own subtree), `node_type`, `parent`, and optionally `is_markup`, `nodes`, `meta`. Keep documents under 2k tokens, use at most
64 sequence ids, make positions gap-free for every id, and let siblings start at the same position. The driver and the test runner
stay unchanged.

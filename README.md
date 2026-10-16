# Tree-structured attention on llama.cpp (CPU) - week 1

Domain: **Markdown documents** (parser: `markdown-it-py`).

## Layout
| path | what |
|---|---|
| `examples/treedriver/treedriver.cpp` | C++ driver (copy of `batched`): load JSON, prefill with structured seq ids, answer a question, evict subtrees |
| `schema/SCHEMA.md` | JSON schema, visibility / position / merge / eviction rules |
| `parsers/markdown/parse_markdown.py` | Markdown -> JSON parser |
| `parsers/markdown/samples/` | 5 sample documents (< 1k tokens each) and `index.json` (question per document) |
| `tests/tree/run_tests.py` | runs Test 1, control and Test 2 on all samples, writes `tests/tree/out/results.md` |
| `tests/tree/stats.py` | tree statistics for the slide |

Other domains: add `parsers/<domain>/parse_<domain>.py` that writes the same JSON schema; the driver and runner stay unchanged.

## Setup (once)
```bash
cd ~/llama.cpp
git rev-parse HEAD                     # llama.cpp commit used: <paste here>
pip install markdown-it-py             # inside your venv
cmake --build build -j 4 --target llama-treedriver llama-tokenize
```

## Run everything
```bash
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
Section ids and start positions are listed in the JSON under `nodes`.

## Tests
* **Test 1** - change an unrelated section B, ask about section A: the greedy answer must be identical.
  Control: asking about B itself must give a different answer on the changed document.
* **Test 2** - evict B's whole subtree by position range: the answer about A is unchanged and the KV state shrinks.


llama.cpp commit used: ace20f679e30a39c09a7b17bec14199855469633

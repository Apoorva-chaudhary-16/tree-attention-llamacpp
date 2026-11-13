"""
Central configuration for the Option-1 project (dense transformer baseline).
Edit this file, not the scripts.
"""
from pathlib import Path

ROOT        = Path(__file__).resolve().parent
LLAMA_DIR   = ROOT / "third_party" / "llama.cpp"
BUILD_DIR   = LLAMA_DIR / "build"
MODELS_DIR  = ROOT / "models"
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"

# Pin llama.cpp so every team member and the PARAM Rudra node build the SAME code.
# Do not change unless you re-test: the profiler uses llama.cpp C APIs that change between versions.
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_REF  = "2bc46baa9207d098d03741ac394e7faaf1a955b6"   # commit these scripts were tested against (5 Nov 2026)

# ---------------------------------------------------------------------------
# Models.  Project rules: Q4_K_M for every model; Q8_0 only if <= 1.5B params;
# no F16.  Context limits: 1B -> 16k, 3B -> 8k.
# GGUF files come from public (non-gated) community repos.  If a filename is not
# found, scripts/01_get_models.py lists what the repo really contains.
# ---------------------------------------------------------------------------
MODELS = {
    "llama-3.2-1b": {
        "repo":    "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "files":   {"Q4_K_M": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                    "Q8_0":   "Llama-3.2-1B-Instruct-Q8_0.gguf"},
        "max_ctx": 16384,
    },
    "llama-3.2-3b": {
        "repo":    "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "files":   {"Q4_K_M": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"},
        "max_ctx": 8192,
    },
}

# ---------------------------------------------------------------------------
# Benchmark sweeps (project table: prefill 512 / 4k / max, decode ctx 1k / 8k / max)
# ---------------------------------------------------------------------------
N_GEN        = 128                       # tokens generated per decode measurement
REPS         = 3                         # project rule: repeat every measurement 3x
PREFILL_LENS = [128, 512, 1024, 2048, 4096, 8192, 16384]   # trimmed to <= max_ctx
DECODE_DEPTHS = [0, 512, 1024, 2048, 4096, 8192, 16384]    # trimmed to <= max_ctx - N_GEN
THREAD_SWEEP_TESTS = {"pp": 512, "tg": N_GEN}              # tests used for thread scaling

# Memory rule from the project: model + KV + buffers < ~4.5 GB
MEM_BUDGET_GB = 4.5

# Perplexity (correctness) -- fixed small sample of WikiText-2 test set
PPL_CTX    = 512
PPL_CHUNKS = 40          # 40 x 512 tokens ~ 20k tokens; same sample for every model
# Q4_K_M is expected to be within this relative PPL increase of Q8_0
PPL_MAX_REL_INCREASE = 0.05
PPL_SANITY_RANGE = (1.0, 100.0)   # a working 1-3B LLM on WikiText-2 lands well inside this

# Reference test for PARAM Rudra (fixed by the project)
REFERENCE = {"model": "llama-3.2-1b", "quant": "Q4_K_M", "pp": 512, "tg": 128, "reps": 3}

# Fixed prompts for the "read a few sample outputs" check (greedy decoding)
SAMPLE_PROMPTS = [
    "The capital of France is",
    "Explain in two sentences why the sky is blue.",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "List three differences between a CPU and a GPU:",
]


# ---------------------------------------------------------------------------
# Smoke-test mode:  CPU_PROJECT_TEST=1 python scripts/06_bench.py --quick
# Uses a ~10 MB random-weight Llama (tests/make_tiny_model.py) so you can verify the whole pipeline
# in minutes before downloading 2 GB.  Results go to results_test/, never mixed with real data.
# ---------------------------------------------------------------------------
import os as _os
if _os.environ.get("CPU_PROJECT_TEST") == "1":
    MODELS = {"tiny-test": {"repo": "", "files": {"Q8_0": "tiny-test-Q8_0.gguf"}, "max_ctx": 2048}}
    RESULTS_DIR = ROOT / "results_test"
    PLOTS_DIR = RESULTS_DIR / "plots"
    PREFILL_LENS = [64, 128, 256, 512]
    DECODE_DEPTHS = [0, 128, 512]
    N_GEN = 32
    PPL_MAX_REL_INCREASE = 10.0
    PPL_CHUNKS = 2
    PPL_SANITY_RANGE = (0.0, 1e9)

# CPU Inference Project - Option 1: Dense transformer baseline (Llama-3.2-1B / 3B)

Everything runs through `llama.cpp` built **CPU-only**. All scripts are Python 3 (no shell needed), work on
Linux / macOS / Windows (WSL2 recommended on Windows), and write CSV files you can plot and cite.

Tested against llama.cpp commit `2bc46baa9207d098d03741ac394e7faaf1a955b6` (pinned in `config.py`).
Smoke-tested end to end with a tiny random model (see "Smoke test"). The real 1B/3B models, the correctness
script on real models, macOS/Windows, and the Rudra node were **not** available to me when writing this, so run
the smoke test first and report any error to me verbatim.

## Prerequisites
* Python >= 3.9, `git`, `cmake >= 3.14`, a C/C++ compiler (gcc >= 11 or clang). Windows: use WSL2 Ubuntu.
* `pip install -r requirements.txt`
* 8 GB RAM laptop: close browser/IDE during runs, plug in the charger, set OS "performance" power mode.

## Run order (maps to the project's 10 steps)

| Step | Command | Output |
|---|---|---|
| 1 build | `python scripts/00_build_llamacpp.py` | llama.cpp binaries + `profiler/build/op_profile` |
| 1 hardware | `python scripts/02_stream.py --tag LAPTOP` then `python scripts/01_hwinfo.py --tag LAPTOP --note "plugged in, performance mode"` | `results/stream_*.json`, `hardware_*.md/json` |
| 2 models | `python scripts/03_get_models.py` | `models/*.gguf`, `data/wikitext-2-raw/` |
| 3 correctness | `python scripts/04_check_correctness.py --tag LAPTOP` | `perplexity_*.csv`, `samples_*.txt` |
| 4 predict | `python scripts/05_roofline.py --tag LAPTOP` | `roofline_predict_*.csv` (do this BEFORE benchmarking, and write the numbers down) |
| 5 sweeps | `python scripts/06_bench.py --tag LAPTOP` (hours; resumable) | `bench_*.csv` |
| 6 profile | `python scripts/07_profile_ops.py --tag LAPTOP [--perf]` | `ops_summary_*.csv`, `ops_top3_*.txt` |
| 7 gap | `python scripts/05_roofline.py --tag LAPTOP --compare` | `roofline_compare_*.csv` |
| 8 experiment | `python scripts/08_experiments.py --tag LAPTOP --which kv_q8,fa_off` | rows with `exp=<name>` in `bench_*.csv` |
| 9 Rudra | see below | rows with `exp=reference` |
| 10 plots | `python scripts/09_plots.py --tag LAPTOP [--compare rudra]` | `results/plots/*.png` |

Tip: first run `python scripts/06_bench.py --quick --tag LAPTOP` (a few minutes) to check the pipeline on the real model.
Use the same `--tag` everywhere. Commit `results/*.csv` to the repo (raw results are a deliverable).

### Laptop hygiene already handled by the code
* Intel hybrid CPU (Linux): performance cores detected and pinned automatically (`-C <mask>`); threads = P-cores. On Windows/macOS pass `--max-threads N` (and `--cpu-mask` if hybrid).
* Apple Silicon: Metal is off in the build; use `--max-threads <performance cores>`.
* 3 repetitions per measurement (`-r 3`), mean and std stored; `samples` column keeps the 3 raw values.
* Peak RSS recorded per run; `over_budget` flags > 4.5 GB; `swap_flag` flags swapping and the run is repeated automatically.
* `throttle_suspect` flags a >15 % CPU-frequency drop during a run (cheap thermal check).
* Context rules enforced from `config.py` (1B: 16k, 3B: 8k; "largest decode context" = max_ctx - 128 generated tokens).

## PARAM Rudra (from the user manual you uploaded)
CPU nodes: 2x Intel Xeon Gold 6240R (Cascade Lake, 2.4 GHz, **48 cores**, AVX-512 + VNNI), 192 GB DDR4-2933,
SLURM partition `cpu`, default walltime 2 h (set `--time` explicitly), compilers via spack
(`module load spack`, `. /home/apps/SPACK/spack/share/spack/setup-env.sh`, `spack load gcc@13.2.0`), python via `module load miniconda`.
The login node has the same CPU model, so a `-march=native` build there also runs on compute nodes.

1. On your laptop: `python scripts/03_get_models.py --model llama-3.2-1b --quant Q4_K_M`, then
   `rsync -av --exclude third_party --exclude results_test ./ <user>@<rudra>:~/cpu-inference/`
   (copy `models/` too; the manual does not say whether compute nodes have internet, so do not rely on it).
2. On the **login node** (one time; keep `--jobs` small - login nodes have CPU-time limits):
   ```
   module load spack; . /home/apps/SPACK/spack/share/spack/setup-env.sh; spack load gcc@13.2.0
   module load miniconda; pip install --user cmake psutil numpy pandas matplotlib gguf py-cpuinfo   # cmake via pip if no cmake module
   export PATH=$HOME/.local/bin:$PATH
   cd ~/cpu-inference && python scripts/00_build_llamacpp.py --jobs 8 --no-profiler
   ```
   (If `git clone` is blocked, clone llama.cpp on your laptop, copy it to `third_party/llama.cpp`, and run the same command; fetch failures are tolerated.)
3. `sbatch slurm/reference_run.sbatch` - runs hardware record, STREAM, and the fixed test
   (1B Q4_K_M, pp512 + tg128, 48 threads, 3 reps, `--numa distribute`). Check `squeue -u $USER`, results in `results/bench_rudra.csv`.
4. Copy `results/` back and plot with `--compare rudra`.
Agree with your team/instructors on the thread count (48 = all cores of the node) so the shared number is comparable.

## Smoke test (2 minutes, no downloads)
```
python scripts/00_build_llamacpp.py
python tests/make_tiny_model.py
export CPU_PROJECT_TEST=1          # Windows PowerShell: $env:CPU_PROJECT_TEST=1
python scripts/06_bench.py --tag t --quick
python scripts/07_profile_ops.py --tag t --reps 1 --depths 0,256 --n-gen 16 --n-prompt 128
python scripts/02_stream.py --tag t --array-elems 10000000 --ntimes 5 --repeats 1
python scripts/05_roofline.py --tag t --compare
python scripts/09_plots.py --tag t
```
Test output goes to `results_test/`, never into your real results.

## Why these choices, and the alternatives
| Decision | Why | Alternatives |
|---|---|---|
| llama.cpp, CPU-only | Required by the project; hand-tuned SIMD kernels (AVX2/AVX-512/NEON) and the `ggml` operator set make results explainable | PyTorch/ONNX Runtime (slower, harder to attribute time to operators), MLC-LLM, ExecuTorch, ik_llama.cpp (tuned fork, option 6 stretch) |
| GGUF Q4_K_M (+Q8_0 for 1B) | Project rule; Q4_K_M ~4.8 bits/weight keeps quality while cutting bytes read per token 3-4x - the whole story of decode speed | Q4_0 (faster, worse quality), Q5_K_M/Q6_K (more bytes), F16 (forbidden, too big) |
| `llama-bench` for timing | Reference tool; handles warm-up, repetitions, context depth (`-d`) | Own timer around `llama-cli` (noisy), Python bindings (extra overhead) |
| Wrapper measures RSS/swap/frequency | llama-bench reports none of them; the project requires peak RSS and flags swapping | `/usr/bin/time -v` (Linux only), `psutil` only |
| STREAM Triad | The accepted bandwidth yardstick; one number gives the roofline ceiling | Intel MLC, sysbench, own memcpy loop |
| Roofline = bandwidth / bytes per token | Decode reads every weight once per token, so it is memory-bound; the model is simple and explains the gap | Full performance model with compute roof (needs peak GFLOPS; useful for prefill discussion) |
| Callback profiler (`op_profile`) | Times real ggml graph nodes by operator and tensor name on any OS; also reports instrumentation overhead | `perf` (Linux only, symbols not operators; kept as `--perf` cross-check), VTune/uProf/Instruments (platform specific), llama.cpp `GGML_PERF` builds |
| WikiText-2 perplexity | Standard small sample; same text for every model | C4/PTB, task benchmarks (slower, not needed for a sanity check) |
| Experiments: KV-cache quantisation, flash-attn on/off, pinning, batch | Each changes one thing and ties back to the roofline/profile story | Thread count, `mlock`, huge pages, different quant per tensor |

## Reading the results (hints for the report)
* Decode at context 0 is bandwidth-bound: expect measured/predicted of roughly 0.5-0.9; the gap comes from the non-matmul operators, thread sync (see `ops_*` instrumentation overhead), cache misses and frequency drop.
* The context where KV bytes equal weight bytes is printed by `05_roofline.py`; compare it with where `attention core` passes ~50 % in `ops_summary_*.csv` - attention often dominates earlier than bytes suggest because it is less efficient than the weight matmuls.
* Thread scaling flattens when the memory bus saturates (compare with the STREAM per-thread numbers).
* The instrumented profile is slower than a plain run; use shares, not absolute milliseconds, and quote the overhead printed in `ops_top3_*.txt`.

## Files
`config.py` settings (models, sweeps, rules) - `common.py` helpers - `gguf_meta.py` byte counting - `scripts/` steps 0-9 -
`profiler/` C++ operator profiler - `slurm/` Rudra job - `tests/make_tiny_model.py` smoke-test model.

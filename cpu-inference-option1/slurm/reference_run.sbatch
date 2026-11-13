#!/bin/bash
# Shared reference run on PARAM Rudra CPU nodes (2x Intel Xeon Gold 6240R = 48 cores, AVX-512, 192 GB DDR4-2933).
# Fixed test: Llama-3.2-1B Q4_K_M, 512-token prompt, 128-token decode, 3 repetitions, all 48 cores.
# Submit from the project folder:   sbatch slurm/reference_run.sbatch
# (Rudra manual: default walltime is 2 h and jobs die at the limit, so it is set explicitly here.)
#SBATCH --job-name=ref_llama1b
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=job.%J.out
#SBATCH --error=job.%J.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results

# --- toolchain (Rudra uses spack for compilers; see manual "Compilation and execution") ---------
module load spack 2>/dev/null || true
. /home/apps/SPACK/spack/share/spack/setup-env.sh 2>/dev/null || true
spack load gcc@13.2.0 2>/dev/null || echo "[warn] could not spack-load gcc@13.2.0 (use the hash shown in the manual: spack load gcc@13.2.0 /3wdooxp)"
module load miniconda 2>/dev/null || true       # provides python3 (and pip for numpy/psutil if needed)
export PATH="$HOME/.local/bin:$PATH"            # where `pip install --user cmake` puts cmake

TAG="${TAG:-rudra}"
export BENCH_MAX_THREADS=48
export OMP_NUM_THREADS=48

[ -x third_party/llama.cpp/build/bin/llama-bench ] || {
  echo "llama.cpp is not built. Build it ONCE on the login node (needs internet), see README section 'PARAM Rudra'."; exit 1; }
for m in models/Llama-3.2-1B-Instruct-Q4_K_M.gguf; do [ -f "$m" ] || { echo "missing $m (scp it from your laptop)"; exit 1; }; done

python3 scripts/01_hwinfo.py --tag "$TAG" --note "SLURM job ${SLURM_JOB_ID:-local}, node ${SLURMD_NODENAME:-?}"
python3 scripts/02_stream.py --tag "$TAG" --max-threads 48 --proc-bind spread

# Two sockets => two NUMA nodes: let llama.cpp spread threads/memory across both.
python3 scripts/06_bench.py --tag "$TAG" --model llama-3.2-1b --quant Q4_K_M --sweep reference \
        --exp-name reference --max-threads 48 --delay 0 --numa distribute

echo "Reference result in results/bench_${TAG}.csv (exp=reference)."
echo "Optional extras (run as a separate job if you want them):"
echo "  python3 scripts/06_bench.py --tag $TAG --sweep all --max-threads 48 --numa distribute"

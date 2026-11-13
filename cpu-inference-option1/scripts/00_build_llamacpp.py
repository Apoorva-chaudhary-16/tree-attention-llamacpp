#!/usr/bin/env python3
"""
Step 1a - clone and build llama.cpp for CPU ONLY (no CUDA / Metal / Vulkan),
then build the per-operator profiler against that build tree.

    python scripts/00_build_llamacpp.py                # normal
    python scripts/00_build_llamacpp.py --ref b1234    # pin a tag / commit
    python scripts/00_build_llamacpp.py --no-profiler  # skip op_profile
    python scripts/00_build_llamacpp.py --jobs 4

Works on Linux, macOS, Windows (needs git, cmake, and a C++ compiler:
gcc/clang, or Visual Studio Build Tools; WSL2 is the easiest on Windows).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from common import IS_WIN, die  # noqa: E402


def run(cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if r.returncode != 0:
        die(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")


def need(tool):
    from shutil import which
    if not which(tool):
        die(f"'{tool}' not found on PATH. Install it first (see README, prerequisites).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=config.LLAMA_CPP_REF, help="git branch/tag/commit (default: config.LLAMA_CPP_REF)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--no-profiler", action="store_true")
    ap.add_argument("--native", choices=["on", "off"], default="on",
                    help="-march=native. Keep ON: it enables AVX2/AVX-512/NEON paths for this CPU.")
    a = ap.parse_args()

    need("git")
    need("cmake")

    src, build = config.LLAMA_DIR, config.BUILD_DIR
    if not (src / ".git").exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", config.LLAMA_CPP_REPO, src])
    # Offline-tolerant (compute nodes often have no internet): a failed fetch is only a warning
    # as long as the wanted commit is already in the local clone.
    if subprocess.run(["git", "fetch", "--tags", "origin"], cwd=src).returncode != 0:
        print("[warn] git fetch failed (offline?) - using the local clone as is")
    run(["git", "checkout", a.ref], cwd=src)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True).stdout.strip()

    cfg = ["cmake", "-S", src, "-B", build,
           "-DCMAKE_BUILD_TYPE=Release",
           f"-DGGML_NATIVE={'ON' if a.native == 'on' else 'OFF'}",
           # CPU ONLY: the project forbids GPU backends (your RTX GPU must not be used)
           "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF", "-DGGML_HIP=OFF",
           "-DGGML_SYCL=OFF", "-DGGML_OPENCL=OFF", "-DGGML_BLAS=OFF",
           "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF",
           ]
    run(cfg)
    targets = ["llama-bench", "llama-perplexity", "llama", "ggml"]
    bcmd = ["cmake", "--build", build, "--config", "Release", "-j", a.jobs]
    for t in targets:
        bcmd += ["--target", t]
    run(bcmd)
    # Non-interactive text generation: 'llama-completion' in new versions, 'llama-cli' in old ones.
    for t in ("llama-completion", "llama-cli"):
        r = subprocess.run([str(c) for c in ["cmake", "--build", build, "--config", "Release",
                                             "-j", a.jobs, "--target", t]])
        if r.returncode == 0:
            targets.append(t)
            break
    if not a.no_profiler:
        pb = config.ROOT / "profiler" / "build"
        run(["cmake", "-S", config.ROOT / "profiler", "-B", pb, "-DCMAKE_BUILD_TYPE=Release",
             f"-DLLAMA_SRC_DIR={src}", f"-DLLAMA_BUILD_DIR={build}"])
        run(["cmake", "--build", pb, "--config", "Release", "-j", a.jobs])

    (config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "llama_cpp_commit.txt").write_text(commit + "\n")
    print("\n=====================================================")
    print(f"llama.cpp commit : {commit}")
    print("To make results reproducible, set in config.py:")
    print(f'    LLAMA_CPP_REF = "{commit}"')
    print("=====================================================")
    ok = [t for t in targets]
    print("Built:", ", ".join(ok))


if __name__ == "__main__":
    main()

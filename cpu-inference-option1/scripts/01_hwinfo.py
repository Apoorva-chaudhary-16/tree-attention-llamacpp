#!/usr/bin/env python3
"""
Step 1b - record the hardware (project "Hardware record").
Writes results/hardware_<tag>.json and results/hardware_<tag>.md (paste into the report).
Memory bandwidth is measured separately by 02_stream.py and merged here if present.

    python scripts/01_hwinfo.py --tag mylaptop
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import common as C  # noqa: E402

SIMD_OF_INTEREST = ["sse4_2", "avx", "avx2", "fma", "f16c", "avx_vnni", "avx512f", "avx512bw", "avx512vl",
                    "avx512_vnni", "avx512vnni", "avx512_bf16", "avx512bf16", "amx_tile", "amx_int8", "amx_bf16",
                    "neon", "asimd", "asimddp", "i8mm", "bf16", "sve", "sve2", "fphp", "asimdhp"]


def sh(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=isinstance(cmd, str))
        return r.stdout.strip()
    except Exception:
        return ""


def linux_info():
    d = {}
    txt = Path("/proc/cpuinfo").read_text(errors="replace")
    m = re.search(r"model name\s*:\s*(.+)", txt)
    d["cpu_model"] = m.group(1).strip() if m else ""
    if not d["cpu_model"]:                       # ARM Linux
        d["cpu_model"] = sh("lscpu | grep -i 'model name' | head -1 | cut -d: -f2").strip()
    m = re.search(r"^(flags|Features)\s*:\s*(.+)$", txt, re.M)
    flags = set(m.group(2).split()) if m else set()
    d["simd"] = sorted(f for f in SIMD_OF_INTEREST if f in flags)
    ls = sh("lscpu")
    for key, field in (("L1d cache", "l1d"), ("L1i cache", "l1i"), ("L2 cache", "l2"), ("L3 cache", "l3"),
                       ("Socket(s)", "sockets"), ("Thread(s) per core", "threads_per_core"),
                       ("Core(s) per socket", "cores_per_socket"), ("NUMA node(s)", "numa_nodes"),
                       ("CPU max MHz", "max_mhz"), ("Model name", "lscpu_model")):
        m = re.search(rf"^{re.escape(key)}:\s*(.+)$", ls, re.M)
        if m:
            d[field] = m.group(1).strip()
    if "cpu_model" in d and not d["cpu_model"] and d.get("lscpu_model"):
        d["cpu_model"] = d["lscpu_model"]
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov.exists():
        d["scaling_governor"] = gov.read_text().strip()
    pc = C.intel_pcore_list()
    if pc:
        d["hybrid_pcore_logical_ids"] = pc
        d["hybrid_pcore_mask"] = C.pcore_mask_hex()
        ac = Path("/sys/devices/cpu_atom/cpus")
        if ac.exists():
            d["hybrid_ecore_logical_ids"] = ac.read_text().strip()
    d["wsl"] = "microsoft" in platform.release().lower()
    return d


def mac_info():
    d = {"cpu_model": sh("sysctl -n machdep.cpu.brand_string")}
    for key, field in (("hw.perflevel0.physicalcpu", "performance_cores"), ("hw.perflevel1.physicalcpu", "efficiency_cores"),
                       ("hw.l1dcachesize", "l1d"), ("hw.l2cachesize", "l2"), ("hw.l3cachesize", "l3"),
                       ("hw.cachelinesize", "cacheline"), ("hw.memsize", "memsize_bytes")):
        v = sh(f"sysctl -n {key}")
        if v:
            d[field] = v
    d["simd"] = ["neon"] if platform.machine() == "arm64" else []
    d["power_mode"] = sh("pmset -g | grep -i 'powermode\\|lowpowermode'")
    return d


def win_info():
    ps = ("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,"
          "L2CacheSize,L3CacheSize,MaxClockSpeed | ConvertTo-Json")
    raw = sh(["powershell", "-NoProfile", "-Command", ps])
    d = {}
    try:
        j = json.loads(raw)
        j = j[0] if isinstance(j, list) else j
        d = {"cpu_model": j.get("Name", "").strip(), "cores": j.get("NumberOfCores"),
             "logical": j.get("NumberOfLogicalProcessors"), "l2": f"{j.get('L2CacheSize')} KB (total)",
             "l3": f"{j.get('L3CacheSize')} KB", "max_mhz": j.get("MaxClockSpeed")}
    except Exception:
        d = {"cpu_model": platform.processor()}
    d["power_scheme"] = sh("powercfg /getactivescheme")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--note", default="", help="free text, e.g. 'plugged in, Best Performance'")
    a = ap.parse_args()
    tag = C.host_tag(a.tag)

    info = {"tag": tag, "note": a.note, "os": platform.platform(), "machine": platform.machine(),
            "python": sys.version.split()[0], "ram_total_gb": round(C.total_gb(), 2),
            "ram_available_gb": round(C.available_gb(), 2), "logical_cpus_allowed": C.allowed_cpus(),
            "default_max_threads": C.default_max_threads(),
            "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "slurm_node": os.environ.get("SLURMD_NODENAME", "")}
    if C.psutil:
        info["physical_cores_psutil"] = C.psutil.cpu_count(logical=False)
        info["logical_cpus_psutil"] = C.psutil.cpu_count(logical=True)
    if C.IS_LINUX:
        info.update(linux_info())
    elif C.IS_MAC:
        info.update(mac_info())
    elif C.IS_WIN:
        info.update(win_info())

    # py-cpuinfo gives flags/caches on Windows & macOS (and a second opinion elsewhere)
    try:
        import cpuinfo  # type: ignore
        ci = cpuinfo.get_cpu_info()
        info.setdefault("cpu_model", ci.get("brand_raw", ""))
        if not info.get("simd"):
            info["simd"] = sorted(f for f in SIMD_OF_INTEREST if f in set(ci.get("flags", [])))
        for k_src, k_dst in (("l2_cache_size", "l2"), ("l3_cache_size", "l3"), ("l1_data_cache_size", "l1d")):
            if k_src in ci and k_dst not in info:
                info[k_dst] = str(ci[k_src])
    except Exception:
        pass

    simd = set(info.get("simd", []))
    info["simd_summary"] = ("AVX-512" if simd & {"avx512f"} else "AVX2" if "avx2" in simd else
                            "NEON" if simd & {"neon", "asimd"} else "unknown")
    info["llama_cpp_commit"] = C.llama_commit()
    info["compiler"] = sh("gcc --version | head -1") or sh("clang --version | head -1")

    sj = config.RESULTS_DIR / f"stream_{tag}.json"
    if sj.exists():
        info["stream"] = json.loads(sj.read_text())

    outj = config.RESULTS_DIR / f"hardware_{tag}.json"
    C.json_dump(outj, info)

    lines = [f"### Hardware record - {tag}", "",
             "| Item | Value |", "|---|---|",
             f"| CPU | {info.get('cpu_model', '?')} |",
             f"| Cores / threads used | physical(psutil)={info.get('physical_cores_psutil', '?')}, "
             f"benchmark max threads={info['default_max_threads']} |",
             f"| Hybrid P-core mask | {info.get('hybrid_pcore_mask', 'n/a (not hybrid or not Linux)')} |",
             f"| Cache L1d / L2 / L3 | {info.get('l1d', '?')} / {info.get('l2', '?')} / {info.get('l3', '?')} |",
             f"| SIMD | {info['simd_summary']} ({', '.join(info.get('simd', []))}) |",
             f"| RAM | {info['ram_total_gb']} GB total |",
             f"| OS | {info['os']} |",
             f"| Governor / power | {info.get('scaling_governor') or info.get('power_scheme') or info.get('power_mode') or '?'} |",
             f"| llama.cpp commit | {info['llama_cpp_commit']} |"]
    if "stream" in info:
        s = info["stream"]
        lines.append(f"| STREAM Triad (best, {s.get('best_threads')} thr) | {s.get('peak_triad_gbs', '?'):.1f} GB/s |")
    else:
        lines.append("| STREAM bandwidth | not measured yet - run scripts/02_stream.py |")
    md = "\n".join(lines) + "\n"
    (config.RESULTS_DIR / f"hardware_{tag}.md").write_text(md)
    print(md)
    print(f"saved {outj}")
    if C.IS_LINUX and info.get("scaling_governor") not in (None, "performance"):
        print("[hint] governor is not 'performance' (laptops often use powersave/schedutil). "
              "Plug in the charger and set the OS performance mode before benchmarking.")
    if info["ram_available_gb"] < 5:
        print("[warn] less than 5 GB RAM available: close browser/IDE before benchmark sweeps.")


if __name__ == "__main__":
    main()

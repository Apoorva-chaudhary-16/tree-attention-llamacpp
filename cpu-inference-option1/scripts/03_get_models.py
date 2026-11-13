#!/usr/bin/env python3
"""
Step 2 - download the GGUF models (+ the WikiText-2 sample for perplexity) and sanity-check them.

    python scripts/03_get_models.py                  # 1B Q4_K_M, 1B Q8_0, 3B Q4_K_M, wikitext
    python scripts/03_get_models.py --model llama-3.2-1b --quant Q4_K_M    # only what the Rudra reference run needs
    python scripts/03_get_models.py --list           # show what the HF repos contain

Project rule: "On the first day, confirm that a GGUF file exists for your model."  This script does
exactly that: it asks the Hugging Face API for the file list before downloading and, if the
configured filename is missing, prints the closest matches so you can edit config.MODELS.

Downloads resume after interruption. No account/token is needed for the default repos.
The PARAM Rudra node may have no internet: run this on your laptop and copy models/ with scp/rsync.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import common as C  # noqa: E402

HF = "https://huggingface.co"
WIKI_URL = f"{HF}/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"


def http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "cpu-inference-project"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def repo_files(repo):
    try:
        return [s["rfilename"] for s in http_json(f"{HF}/api/models/{repo}")["siblings"]]
    except Exception as e:
        print(f"  (could not list {repo}: {e})")
        return None


def download(url, dst: Path, expect_min_bytes=1_000_000):
    """Resumable download (HTTP Range)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, 6):
        start = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "cpu-inference-project",
                                                   **({"Range": f"bytes={start}-"} if start else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = r.headers.get("Content-Range", "").split("/")[-1] or r.headers.get("Content-Length", "0")
                total = int(total) if str(total).isdigit() else 0
                if start and r.status != 206:      # server ignored Range -> restart
                    start = 0
                    part.unlink(missing_ok=True)
                mode = "ab" if start else "wb"
                done = start
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {dst.name}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            print()
            if part.stat().st_size < expect_min_bytes:
                C.die(f"download of {url} is suspiciously small ({part.stat().st_size} bytes)")
            part.replace(dst)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            print(f"\n  download interrupted ({e}); retry {attempt}/5")
    C.die(f"could not download {url}")


def describe_gguf(path: Path):
    try:
        from gguf import GGUFReader  # type: ignore
    except Exception:
        print("  (install 'gguf' to print metadata)")
        return
    r = GGUFReader(str(path))
    def fv(k):
        f = r.fields.get(k)
        if f is None:
            return None
        try:
            v = f.contents() if hasattr(f, "contents") else f.parts[f.data[0]][0]
            return v.item() if hasattr(v, "item") else v
        except Exception:
            return None
    arch = fv("general.architecture")
    print(f"  arch={arch} layers={fv(f'{arch}.block_count')} heads={fv(f'{arch}.attention.head_count')} "
          f"kv_heads={fv(f'{arch}.attention.head_count_kv')} embd={fv(f'{arch}.embedding_length')} "
          f"ctx_train={fv(f'{arch}.context_length')} tensors={len(r.tensors)}")
    if arch != "llama":
        print("  [warn] architecture is not 'llama' - is this the right file?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--quant", default="all")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-wikitext", action="store_true")
    a = ap.parse_args()

    for key, m in config.MODELS.items():
        if a.model not in ("all", key):
            continue
        listing = repo_files(m["repo"])
        if a.list and listing:
            print(f"{m['repo']}:\n  " + "\n  ".join(f for f in listing if f.endswith('.gguf')))
            continue
        for quant, fname in m["files"].items():
            if a.quant not in ("all", quant):
                continue
            dst = config.MODELS_DIR / fname
            if dst.exists():
                print(f"[ok] {dst.name} already present ({dst.stat().st_size / 1e9:.2f} GB)")
            else:
                if listing is not None and fname not in listing:
                    close = [f for f in listing if quant.lower() in f.lower() and f.endswith(".gguf")]
                    C.die(f"{fname} not in {m['repo']}. Similar files: {close}. "
                          f"Edit config.MODELS (project rule: use the nearest sibling).")
                print(f"[get] {m['repo']}/{fname}")
                download(f"{HF}/{m['repo']}/resolve/main/{fname}", dst)
            describe_gguf(dst)

    if not a.no_wikitext and not a.list:
        raw = config.DATA_DIR / "wikitext-2-raw" / "wiki.test.raw"
        if raw.exists():
            print(f"[ok] {raw}")
        else:
            z = config.DATA_DIR / "wikitext-2-raw-v1.zip"
            print("[get] wikitext-2 test set")
            download(WIKI_URL, z, expect_min_bytes=100_000)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(config.DATA_DIR)
            if not raw.exists():
                C.die(f"expected {raw} after unzip")
            print(f"[ok] {raw} ({raw.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

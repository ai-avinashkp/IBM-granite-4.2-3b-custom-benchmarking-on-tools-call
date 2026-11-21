"""Phase 2 (laptop serving prep, runs as an HPC task): merged model -> GGUF bf16 -> Q8_0 and Q4_K_M with llama.cpp.

llama.cpp is a clone at $LLAMA_CPP (default ./llama.cpp; commit recorded in the output JSON). The bf16 GGUF is
temporary and deleted after quantising. Only the small Q4_K_M file is meant to be copied to the laptop.
  python make_gguf.py --model <merged folder> --out <folder>
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LLAMA = Path(os.environ.get("LLAMA_CPP", "llama.cpp")).resolve()


def sh(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    t0 = time.monotonic()
    subprocess.run(list(map(str, cmd)), check=True, **kw)
    return round(time.monotonic() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    times = {}
    quant = LLAMA / "build" / "bin" / "llama-quantize"
    if not quant.exists():
        times["build_s"] = sh(["cmake", "-S", LLAMA, "-B", LLAMA / "build", "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF",
                               "-DCMAKE_BUILD_TYPE=Release"])
        times["build_s"] += sh(["cmake", "--build", LLAMA / "build", "--target", "llama-quantize", "-j", "4"])
    bf16 = out / "sft4-bf16.gguf"
    times["convert_s"] = sh([sys.executable, LLAMA / "convert_hf_to_gguf.py", a.model, "--outtype", "bf16",
                             "--outfile", bf16])
    sizes = {"bf16_gib": round(bf16.stat().st_size / 2**30, 2)}
    for q in ("Q8_0", "Q4_K_M"):
        f = out / f"sft4-{q}.gguf"
        times[f"quantize_{q}_s"] = sh([quant, bf16, f, q, "4"])
        sizes[f"{q}_gib"] = round(f.stat().st_size / 2**30, 2)
    bf16.unlink()
    commit = subprocess.run(["git", "-C", LLAMA, "log", "-1", "--format=%h %ci"], capture_output=True, text=True).stdout.strip()
    res = {"llama_cpp_commit": commit, "model": a.model, "sizes": sizes, "times": times}
    (out / "gguf.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

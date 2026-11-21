"""Phase 2 (laptop): GGUF serving speed on the RTX 3070 Ti 8 GB with llama.cpp (llama-cpp-python, CUDA build, envServe).

Two steps, so the serving env needs no transformers:
  1) envPytorch:  python laptop_bench.py render --out prompts.json    (same val prompts as serve_bench.py, 4.2 template)
  2) envServe:    python laptop_bench.py bench --model <file.gguf> --prompts prompts.json --out result.json
Measures time to first token (streaming), decode tokens/s after the first token (up to 128 new tokens) and GPU memory
used after loading (nvidia-smi), all with full GPU offload (n_gpu_layers = -1).
"""
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

VAL = Path(__file__).resolve().parent.parent / "data" / "v3" / "val.jsonl"


def render(out, n=20):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-3b")
    prompts = []
    for line in open(VAL, encoding="utf-8"):
        r = json.loads(line)
        k = next((i for i, m in enumerate(r["messages"]) if m["role"] == "user"), None)
        if k is not None:
            prompts.append(tok.apply_chat_template(r["messages"][:k + 1], tools=r["tools"], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False))
        if len(prompts) == n:
            break
    Path(out).write_text(json.dumps(prompts), encoding="utf-8")
    print(f"{len(prompts)} prompts -> {out}")


def gpu_mem_mib():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    return int(q[0]) if q else None


def bench(model, prompts_path, out):
    import os
    import llama_cpp as _pkg_probe  # noqa: F401  (import fails without the DLL folders below; done after adding them)
    from llama_cpp import Llama
    return _bench(Llama, model, prompts_path, out)


def add_dll_dirs():
    """Windows: the CUDA wheel needs CUDA runtime DLLs (e.g. a PyTorch install's torch/lib, CUDA 12.x) and its own folder.
    Set CUDA_DLL_DIR to that folder."""
    import os
    import importlib.util
    torch_lib = os.environ["CUDA_DLL_DIR"]
    spec = importlib.util.find_spec("llama_cpp")
    own = str(Path(spec.origin).parent / "lib")
    os.environ["PATH"] = torch_lib + os.pathsep + own + os.pathsep + os.environ["PATH"]   # llama_cpp loads with
    os.add_dll_directory(torch_lib)                                                       # winmode=0 (uses PATH)
    os.add_dll_directory(own)


def _bench(Llama, model, prompts_path, out):
    prompts = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
    mem0 = gpu_mem_mib()
    t0 = time.perf_counter()
    llm = Llama(model_path=model, n_gpu_layers=-1, n_ctx=8192, verbose=False, seed=0)
    load_s = time.perf_counter() - t0
    mem1 = gpu_mem_mib()
    llm.create_completion(prompts[0], max_tokens=4, temperature=0)                      # warm-up
    ttft, decode = [], []
    for p in prompts[:10]:
        llm.reset()
        t0 = time.perf_counter()
        first, n = None, 0
        for _ in llm.create_completion(p, max_tokens=128, temperature=0, stream=True):
            n += 1
            if first is None:
                first = time.perf_counter()
        end = time.perf_counter()
        ttft.append(first - t0)
        if n > 1:
            decode.append((n - 1) / (end - first))
    res = {"model": model, "gpu": "RTX 3070 Ti Laptop 8 GB", "n_prompts": len(ttft),
           "ttft_ms": round(1000 * statistics.median(ttft), 1),
           "decode_tok_s": round(statistics.median(decode), 1) if decode else None,
           "load_s": round(load_s, 1), "gpu_mem_used_mib_after_load": mem1,
           "gpu_mem_model_mib": (mem1 - mem0) if mem0 is not None and mem1 is not None else None}
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["render", "bench"])
    ap.add_argument("--model")
    ap.add_argument("--prompts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.step == "render":
        render(a.out)
    else:
        add_dll_dirs()
        bench(a.model, a.prompts, a.out)

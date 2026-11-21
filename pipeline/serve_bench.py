"""Phase 2: serving speed of one model folder with vLLM's offline engine (runs as an HPC task in the bfcl env).

Measures, on prompts built from our own validation chats (4.2 template, tools included):
  ttft_ms         latency of a single request with max_tokens=1 (prefill + first token), median of 20
  decode_tok_s    batch-1 decode speed: 128 extra tokens / (time(max_tokens=129) - ttft), median of 10
  throughput_tok_s  generated tokens per second for a batch of N prompts, 256 tokens each (ignore_eos)
  load_gib        "Model loading took X GiB" from vLLM's log (weights in GPU memory)
  python serve_bench.py --model <merged folder> --tag bf16 --out <json> --val <val.jsonl> [--quant fp8]
"""
import argparse
import json
import logging
import re
import statistics
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class LoadSize(logging.Handler):
    gib = None

    def emit(self, record):
        m = re.search(r"Model loading took ([0-9.]+) ?GiB", record.getMessage())
        if m:
            LoadSize.gib = float(m.group(1))


def prompts_from_val(path, tok, n):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        first_user = next((i for i, m in enumerate(r["messages"]) if m["role"] == "user"), None)
        if first_user is None:
            continue
        out.append(tok.apply_chat_template(r["messages"][:first_user + 1], tools=r["tools"], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False))
        if len(out) == n:
            break
    return out


def timed(llm, prompts, params):
    t0 = time.perf_counter()
    res = llm.generate(prompts, params, use_tqdm=False)
    return time.perf_counter() - t0, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--quant", default=None)          # None (bf16) or "fp8"
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    logging.getLogger("vllm").addHandler(LoadSize())
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = prompts_from_val(a.val, tok, max(a.batch, 20))
    extra = {"load_format": "bitsandbytes"} if a.quant == "bitsandbytes" else {}     # in-flight 4-bit (NF4)
    llm = LLM(model=a.model, dtype="bfloat16", quantization=a.quant, max_model_len=8192, gpu_memory_utilization=0.85,
              seed=0, **extra)
    timed(llm, prompts[:2], SamplingParams(max_tokens=8, temperature=0))                      # warm-up
    ttft = [timed(llm, [p], SamplingParams(max_tokens=1, temperature=0))[0] for p in prompts[:20]]
    t129 = [timed(llm, [p], SamplingParams(max_tokens=129, temperature=0, ignore_eos=True))[0] for p in prompts[:10]]
    decode = [128 / (t - statistics.median(ttft)) for t in t129]
    t_batch, res = timed(llm, prompts[:a.batch], SamplingParams(max_tokens=256, temperature=0, ignore_eos=True))
    gen = sum(len(r.outputs[0].token_ids) for r in res)
    out = {"tag": a.tag, "model": a.model, "quant": a.quant or "bf16",
           "prompt_tokens_median": statistics.median(len(tok(p)["input_ids"]) for p in prompts[:20]),
           "ttft_ms": round(1000 * statistics.median(ttft), 1), "decode_tok_s": round(statistics.median(decode), 1),
           f"throughput_tok_s_batch{a.batch}": round(gen / t_batch, 1), "load_gib": LoadSize.gib}
    print(json.dumps(out), flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

"""Development measurement for the call-or-answer gate, on OUR validation data only (BFCL stays a clean test).

For each case: p_call = P(<tool_call> starts the reply) at the decision point right after the generation prompt
("<|im_start|>assistant\\n<think></think>"); the reply starts either "\\n<tool_call>" or with text, so
p_call = P(t1=<tool_call>) + P(t1="\\n") * P(t2=<tool_call> | "\\n").
Cases from data/v2/val.jsonl:
  call      first assistant turn of a chat that calls a tool (correct decision: call)
  hard_neg  the same request with the called tool(s) removed from the menu (correct: do not call)
  easy_neg  data-v2 unrelated-tools chats (correct: do not call)
Then sweeps the threshold tau and reports call recall vs rejection rates.

  python pipeline/gate_dev.py --adapter runs/sft3b/adapter_step350 --tag sft3b
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import load_base  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOOL_CALL, NL = 100270, 198


def dev_set(path, rng):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    pool = {t["function"]["name"]: t for r in rows for t in (r["tools"] or [])}
    items = []
    for r in rows:
        m = r["messages"]
        k = next((i for i, x in enumerate(m) if x["role"] == "assistant"), None)
        if k is None or not r["tools"]:
            continue
        prefix, first = m[:k], m[k]
        names = [c["function"]["name"] for c in first.get("tool_calls", [])]
        if names:
            items.append({"id": r["id"], "kind": "call", "messages": prefix, "tools": r["tools"]})
            rest = [t for t in r["tools"] if t["function"]["name"] not in names]
            if not rest:                                      # keep a non-empty menu of tools that cannot help
                rest = rng.sample([t for n, t in sorted(pool.items()) if n not in names], 2)
            items.append({"id": r["id"], "kind": "hard_neg", "messages": prefix, "tools": rest})
        else:
            items.append({"id": r["id"], "kind": "easy_neg", "messages": prefix, "tools": r["tools"]})
    return items


@torch.no_grad()
def p_call(model, tok, messages, tools):
    prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p1 = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float().softmax(-1)
        ids2 = torch.cat([ids, torch.tensor([[NL]], device="cuda")], dim=1)
        p2 = model(input_ids=ids2, logits_to_keep=1).logits[0, -1].float().softmax(-1)
    return (p1[TOOL_CALL] + p1[NL] * p2[TOOL_CALL]).item()


def sweep(items):
    out = []
    for tau in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        rate = {k: sum((x["p_call"] >= tau) == (k == "call") for x in items if x["kind"] == k) /
                max(1, sum(x["kind"] == k for x in items)) for k in ("call", "hard_neg", "easy_neg")}
        score = 0.5 * rate["call"] + 0.25 * rate["hard_neg"] + 0.25 * rate["easy_neg"]
        out.append({"tau": tau, "call_recall": rate["call"], "hard_reject": rate["hard_neg"],
                    "easy_reject": rate["easy_neg"], "score": score})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)        # comma-separated adapter folders
    ap.add_argument("--tags", required=True)            # comma-separated names, one per adapter
    ap.add_argument("--val", default=str(ROOT / "data" / "v2" / "val.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-3b")
    items = dev_set(a.val, random.Random(0))
    for adapter, tag in zip(a.adapters.split(","), a.tags.split(",")):
        if adapter.endswith("/best"):                   # "<run folder>/best" -> the adapter named in best.json
            run_dir = Path(adapter).parent
            adapter = str(run_dir / f"adapter_step{json.loads((run_dir / 'best.json').read_text())['step']}")
            print(f"{tag}: best adapter {adapter}", flush=True)
        model = PeftModel.from_pretrained(load_base(), adapter).eval()
        t0 = time.monotonic()
        for i, x in enumerate(items):
            x["p_call"] = p_call(model, tok, x["messages"], x["tools"])
            if i % 50 == 0:
                print(f"{tag}: {i}/{len(items)} ({time.monotonic() - t0:.0f}s)", flush=True)
        res = sweep(items)
        best = max(res, key=lambda r: r["score"])
        Path(a.out).mkdir(parents=True, exist_ok=True)
        Path(a.out, f"gate_{tag}.json").write_text(json.dumps({"adapter": adapter, "sweep": res, "best": best,
            "p_call": [{k: x[k] for k in ("id", "kind", "p_call")} for x in items]}, indent=1))
        print(f"\n{tag}: {len(items)} cases ({sum(x['kind'] == 'call' for x in items)} call / "
              f"{sum(x['kind'] == 'hard_neg' for x in items)} hard / {sum(x['kind'] == 'easy_neg' for x in items)} easy); "
              f"{time.monotonic() - t0:.0f}s")
        print(f"{'tau':>5s} {'call_recall':>11s} {'hard_reject':>11s} {'easy_reject':>11s} {'score':>6s}")
        for r in res:
            print(f"{r['tau']:5.2f} {r['call_recall']:11.3f} {r['hard_reject']:11.3f} {r['easy_reject']:11.3f} "
                  f"{r['score']:6.3f}{' <- best' if r is best else ''}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

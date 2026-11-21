"""Phase 4: where the GPU time goes in the merged model (transformers, bf16, sdpa attention), with torch.profiler.

Every RMSNorm, every residual update "h = residual + out * 0.22", the attention and the MLP blocks run inside
record_function ranges, so the profiler sums the GPU kernels of each. The fused Triton kernel (triton_rmsnorm.py)
replaces exactly the RMSNorm + residual ops, so their share bounds what the kernel can save end to end.
Two cases: prefill (1 x 512 tokens) and decode (1 new token per step, KV cache of 512+ tokens).
  python profile_model.py --model <merged folder> --val <val.jsonl> --out <json>
"""
import argparse
import json
import time

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.granite import modeling_granite as mg

PARTS = ["rmsnorm", "residual", "attention", "mlp", "lm_head"]


def wrap(cls, name):
    orig = cls.forward

    def forward(self, *args, **kwargs):
        with record_function(name):
            return orig(self, *args, **kwargs)
    cls.forward = forward


def layer_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False,
                  position_embeddings=None, **kwargs):
    """GraniteDecoderLayer.forward (transformers 5.14) with the two residual updates marked."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask,
                                      position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache,
                                      position_embeddings=position_embeddings, **kwargs)
    with record_function("residual"):
        hidden_states = residual + hidden_states * self.residual_multiplier
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    with record_function("residual"):
        hidden_states = residual + hidden_states * self.residual_multiplier
    return hidden_states


def prompt_ids(tok, val, n_tok):
    ids = []
    for line in open(val, encoding="utf-8"):
        r = json.loads(line)
        ids += tok.apply_chat_template(r["messages"], tools=r["tools"], tokenize=True, enable_thinking=False,
                                       return_dict=False)
        if len(ids) >= n_tok:
            return torch.tensor([ids[:n_tok]], device="cuda")
    raise SystemExit("not enough validation text")


def gpu_ms(e):
    return getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)


def summarise(prof, n_steps):
    by_name = {}
    for e in prof.key_averages():
        if e.key in PARTS + ["step"] and e.device_type == DeviceType.CPU:
            by_name[e.key] = gpu_ms(e) / 1000 / n_steps                   # ms of GPU kernels per step
    step = by_name["step"]
    parts = {k: round(by_name.get(k, 0.0), 3) for k in PARTS}
    parts["other"] = round(step - sum(parts.values()), 3)
    share = {k: round(100 * v / step, 1) for k, v in parts.items()}
    kernels = sorted((e for e in prof.key_averages()                  # GPU-side copies of our ranges are not kernels
                      if e.device_type == DeviceType.CUDA and e.key not in PARTS + ["step"]),
                     key=lambda e: -e.self_device_time_total if hasattr(e, "self_device_time_total") else -e.self_cuda_time_total)
    top = [{"kernel": e.key[:90], "ms_per_step": round((getattr(e, "self_device_time_total", None)
                                                        or e.self_cuda_time_total) / 1000 / n_steps, 3)}
           for e in kernels[:10]]
    return {"gpu_ms_per_step": round(step, 3), "ms": parts, "share_pct": share, "top_kernels": top}


def wall_ms(fn, n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return round(1000 * (time.perf_counter() - t0) / n, 3)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    a = ap.parse_args()
    for cls, name in [(mg.GraniteRMSNorm, "rmsnorm"), (mg.GraniteAttention, "attention"), (mg.GraniteMLP, "mlp")]:
        wrap(cls, name)
    mg.GraniteDecoderLayer.forward = layer_forward
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    head = model.lm_head.forward                  # instance-level: patching nn.Linear would mark every projection

    def head_forward(x):
        with record_function("lm_head"):
            return head(x)
    model.lm_head.forward = head_forward
    ids = prompt_ids(tok, a.val, a.tokens)

    def prefill():
        with record_function("step"):
            return model(ids, use_cache=False, logits_to_keep=1)

    cache = {}

    def start_decode():
        out = model(ids, use_cache=True, logits_to_keep=1)
        cache["past"], cache["next"] = out.past_key_values, out.logits[:, -1:].argmax(-1)

    def decode():
        with record_function("step"):
            out = model(cache["next"], past_key_values=cache["past"], use_cache=True)
        cache["past"], cache["next"] = out.past_key_values, out.logits[:, -1:].argmax(-1)

    res = {"device": torch.cuda.get_device_name(), "model": a.model, "prompt_tokens": a.tokens,
           "torch": torch.__version__}
    for name, fn, n, setup in [("prefill", prefill, 5, None), ("decode", decode, 32, start_decode)]:
        if setup:
            setup()
        for _ in range(3):
            fn()                                                              # warm-up
        res[f"{name}_wall_ms"] = wall_ms(fn, n)
        if setup:
            setup()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
        res[name] = summarise(prof, n)
        print(name, json.dumps(res[name]), flush=True)
    for name in ("prefill", "decode"):
        s = res[name]["share_pct"]
        res[name]["fusable_share_pct"] = round(s["rmsnorm"] + s["residual"], 1)
        res[name]["gpu_busy_pct_of_wall"] = round(100 * res[name]["gpu_ms_per_step"] / res[f"{name}_wall_ms"], 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("prefill_wall_ms", "decode_wall_ms")}),
          {n: (res[n]["fusable_share_pct"], res[n]["gpu_busy_pct_of_wall"]) for n in ("prefill", "decode")})


if __name__ == "__main__":
    main()

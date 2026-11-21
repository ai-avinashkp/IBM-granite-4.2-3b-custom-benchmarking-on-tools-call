"""Phase 4: the Granite decoder written from scratch in plain PyTorch (no transformers model code), run on our merged
sft4 weights, checked against transformers, and used for greedy generation with its own KV cache.

Architecture (from config.json): token embedding x 12 (embedding_multiplier); 40 pre-norm blocks of
  RMSNorm -> GQA attention (40 query heads, 8 KV heads, head_dim 64, RoPE theta 1e7, scores x 1/64) -> h += out x 0.22
  RMSNorm -> SwiGLU MLP (2560 -> 8192 -> 2560)                                            -> h += out x 0.22
then RMSNorm, tied output head, logits / 10 (logits_scaling).
With --fused, each "residual update + next RMSNorm" pair runs as one Triton kernel (triton_rmsnorm.fused).

Checks: (1) logits on validation prompts vs transformers (eager attention, same op order);
        (2) greedy generations, token for token, vs transformers generate();
        (3) speed: prefill of 512 tokens and batch-1 decode, eager vs fused vs transformers.
  python decoder_scratch.py --model <merged folder> --val <val.jsonl> --out <json>
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from triton_rmsnorm import fused as fused_residual_norm

EOS = 100257                                            # <|im_end|>


def rms_norm(x, w, eps):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class GraniteFromScratch:
    def __init__(self, folder, max_len=4096, device="cuda", dtype=torch.bfloat16):
        cfg = json.loads((Path(folder) / "config.json").read_text())
        self.n_layers, self.n_heads, self.n_kv = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.dim = cfg["hidden_size"]
        self.head_dim = self.dim // self.n_heads
        self.eps, self.att_mult = cfg["rms_norm_eps"], cfg["attention_multiplier"]
        self.emb_mult, self.res_mult, self.logit_div = cfg["embedding_multiplier"], cfg["residual_multiplier"], cfg["logits_scaling"]
        theta = cfg.get("rope_theta") or cfg["rope_parameters"]["rope_theta"]
        w = {}
        for f in sorted(Path(folder).glob("*.safetensors")):
            w.update(load_file(str(f), device=device))
        w = {k: v.to(dtype) for k, v in w.items()}
        self.embed, self.final_norm = w["model.embed_tokens.weight"], w["model.norm.weight"]
        self.head = self.embed if cfg.get("tie_word_embeddings", True) else w["lm_head.weight"]
        names = {"ln1": "input_layernorm", "ln2": "post_attention_layernorm", "q": "self_attn.q_proj",
                 "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj", "gate": "mlp.gate_proj",
                 "up": "mlp.up_proj", "down": "mlp.down_proj"}
        self.layers = [{k: w[f"model.layers.{i}.{n}.weight"] for k, n in names.items()} for i in range(self.n_layers)]
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        self.k_cache = torch.empty(self.n_layers, 1, self.n_kv, max_len, self.head_dim, device=device, dtype=dtype)
        self.v_cache = torch.empty_like(self.k_cache)
        self.device, self.dtype = device, dtype

    def attention(self, x, l, i, pos0, cos, sin):
        b, t, _ = x.shape
        q = (x @ l["q"].T).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = (x @ l["k"].T).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = (x @ l["v"].T).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        self.k_cache[i, :, :, pos0:pos0 + t], self.v_cache[i, :, :, pos0:pos0 + t] = k, v
        k, v = self.k_cache[i, :, :, :pos0 + t], self.v_cache[i, :, :, :pos0 + t]
        rep = self.n_heads // self.n_kv                                    # grouped-query attention: share KV heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        scores = (q @ k.transpose(2, 3)) * self.att_mult
        if t > 1:                                                          # causal mask (query i sees keys <= pos0+i)
            allowed = torch.arange(pos0 + t, device=x.device)[None, :] <= torch.arange(pos0, pos0 + t, device=x.device)[:, None]
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        att = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = (att @ v).transpose(1, 2).reshape(b, t, self.dim)
        return out @ l["o"].T

    def mlp(self, x, l):
        return (F.silu(x @ l["gate"].T) * (x @ l["up"].T)) @ l["down"].T

    def forward(self, ids, pos0, fused=False, last_only=True):
        b, t = ids.shape
        x = self.embed[ids] * self.emb_mult
        pos = torch.arange(pos0, pos0 + t, device=ids.device).float()
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(self.dtype), emb.sin().to(self.dtype)
        if not fused:
            for i, l in enumerate(self.layers):
                x = x + self.attention(rms_norm(x, l["ln1"], self.eps), l, i, pos0, cos, sin) * self.res_mult
                x = x + self.mlp(rms_norm(x, l["ln2"], self.eps), l) * self.res_mult
            y = rms_norm(x, self.final_norm, self.eps)
        else:                                     # fused: residual update and the NEXT norm in one kernel
            y = rms_norm(x, self.layers[0]["ln1"], self.eps)
            for i, l in enumerate(self.layers):
                a = self.attention(y, l, i, pos0, cos, sin)
                x, y = (z.view(b, t, -1) for z in fused_residual_norm(x.reshape(-1, self.dim), a.reshape(-1, self.dim),
                                                                     l["ln2"], self.res_mult, self.eps))
                m = self.mlp(y, l)
                nxt = self.layers[i + 1]["ln1"] if i + 1 < self.n_layers else self.final_norm
                x, y = (z.view(b, t, -1) for z in fused_residual_norm(x.reshape(-1, self.dim), m.reshape(-1, self.dim),
                                                                     nxt, self.res_mult, self.eps))
        if last_only:
            y = y[:, -1:]
        return (y @ self.head.T) / self.logit_div

    @torch.no_grad()
    def generate(self, ids, max_new, fused=False, stop=True):
        logits = self.forward(ids, 0, fused)
        out, pos = [], ids.shape[1]
        for _ in range(max_new):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out.append(nxt.item())
            if stop and out[-1] == EOS:
                break
            logits = self.forward(nxt, pos, fused)
            pos += 1
        return out


def prompts(tok, val, n):
    res = []
    for line in open(val, encoding="utf-8"):
        r = json.loads(line)
        k = next((i for i, m in enumerate(r["messages"]) if m["role"] == "user"), None)
        if k is not None:
            res.append(tok.apply_chat_template(r["messages"][:k + 1], tools=r["tools"], add_generation_prompt=True,
                                               enable_thinking=False, tokenize=True, return_dict=False))
        if len(res) == n:
            return res
    return res


def timed(fn, n):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=96)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer      # reference and tokenizer only
    tok = AutoTokenizer.from_pretrained(a.model)
    ps = prompts(tok, a.val, a.n_prompts)
    long_ids = torch.tensor([sum(ps, [])[:512]], device="cuda")
    one = torch.tensor([ps[0]], device="cuda")
    res = {"device": torch.cuda.get_device_name(), "model": a.model, "n_prompts": len(ps)}
    speed = {}

    # A) transformers reference first (eager attention); its outputs are kept on the CPU, then it is freed, so the
    #    two bf16 copies of the model are never in GPU memory together (a 1g.24gb MIG slice holds ~21.6 GiB)
    ref = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0},
                                               attn_implementation="eager").eval()
    ref_logits = [ref(torch.tensor([p], device="cuda")).logits.float()[0].cpu() for p in ps]
    ref_gen = [ref.generate(torch.tensor([p], device="cuda"), max_new_tokens=a.max_new, do_sample=False,
                            eos_token_id=EOS, pad_token_id=EOS)[0, len(p):].tolist() for p in ps]
    speed["transformers_eager_prefill512_ms"] = round(1000 * timed(lambda: ref(long_ids, logits_to_keep=1), 5), 2)
    t = timed(lambda: ref.generate(one, max_new_tokens=128, min_new_tokens=128, do_sample=False, pad_token_id=EOS), 2)
    t0 = timed(lambda: ref.generate(one, max_new_tokens=1, do_sample=False, pad_token_id=EOS), 2)
    speed["transformers_eager_decode_tok_s"] = round(127 / (t - t0), 1)
    print("reference done", speed, flush=True)
    del ref
    gc.collect()
    torch.cuda.empty_cache()

    # B) our decoder
    ours = GraniteFromScratch(a.model)
    # (1) logits, all positions of each prompt
    max_diff, agree, total = 0.0, 0, 0
    for p, lr in zip(ps, ref_logits):
        lo = ours.forward(torch.tensor([p], device="cuda"), 0, last_only=False).float()[0].cpu()
        max_diff = max(max_diff, (lo - lr).abs().max().item())
        agree += (lo.argmax(-1) == lr.argmax(-1)).sum().item()
        total += len(p)
    res["logits"] = {"max_abs_diff": round(max_diff, 4), "top1_agreement": round(agree / total, 4), "positions": total}
    print("logits", res["logits"], flush=True)

    # (2) greedy generations, token for token
    gen = {"eager": [], "fused": []}
    same = {"eager": 0, "fused": 0}
    for p, r in zip(ps, ref_gen):
        ids = torch.tensor([p], device="cuda")
        for mode in gen:
            o = ours.generate(ids, a.max_new, fused=(mode == "fused"))
            gen[mode].append(o)
            same[mode] += o == r
        print("prompt", len(p), "tokens: identical eager/fused =", gen["eager"][-1] == r, gen["fused"][-1] == r, flush=True)
    res["generation"] = {"identical_to_transformers": same, "of": len(ps), "max_new_tokens": a.max_new,
                         "example": tok.decode(gen["eager"][0])}

    # (3) speed: prefill of 512 tokens, decode of 128 tokens (batch 1, no early stop)
    for mode in ("eager", "fused"):
        f = mode == "fused"
        speed[f"ours_{mode}_prefill512_ms"] = round(1000 * timed(lambda: ours.forward(long_ids, 0, f), 5), 2)
        t = timed(lambda: ours.generate(one, 128, fused=f, stop=False), 2)
        t0 = timed(lambda: ours.generate(one, 1, fused=f, stop=False), 2)
        speed[f"ours_{mode}_decode_tok_s"] = round(127 / (t - t0), 1)
    res["speed"] = speed
    print("speed", speed, flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

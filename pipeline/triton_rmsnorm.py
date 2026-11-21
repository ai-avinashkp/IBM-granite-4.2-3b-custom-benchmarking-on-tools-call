"""Phase 4: fused Granite "scaled residual add + RMSNorm" in Triton, checked against PyTorch and benchmarked.

Granite's decoder computes  h = residual + out * residual_multiplier ;  y = RMSNorm(h) * weight  (eps 1e-5, hidden 2560).
In PyTorch that is several memory passes (mul, add, pow, mean, rsqrt, mul, mul); the kernel reads residual and out once and
writes h and y once, one program per row.

  python triton_rmsnorm.py --out <json>      (GPU; runs as an HPC task)
"""
import argparse
import json

import torch
import triton
import triton.language as tl

HIDDEN, EPS, RES_MULT = 2560, 1e-5, 0.22             # granite-4.1-3b-base config values


@triton.jit
def _fused_kernel(res_ptr, out_ptr, w_ptr, h_ptr, y_ptr, n_cols, res_mult, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = row * n_cols
    r = tl.load(res_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(out_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    h = r + o * res_mult
    var = tl.sum(h * h, axis=0) / n_cols
    y = h * tl.rsqrt(var + eps) * tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(h_ptr + base + cols, h.to(h_ptr.dtype.element_ty), mask=mask)
    tl.store(y_ptr + base + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def fused(res, out, w, res_mult=RES_MULT, eps=EPS):
    n_rows, n_cols = res.shape
    h, y = torch.empty_like(res), torch.empty_like(res)
    _fused_kernel[(n_rows,)](res, out, w, h, y, n_cols, res_mult, eps, BLOCK=triton.next_power_of_2(n_cols),
                             num_warps=8)
    return h, y


def reference(res, out, w, res_mult=RES_MULT, eps=EPS):
    h = res + out * res_mult                                            # as in the model (bf16 math)
    hf = h.float()
    y = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps)).to(h.dtype) * w
    return h, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    rows = []
    compiled = torch.compile(reference, dynamic=False)       # fairer baseline: PyTorch's own fusion
    for n_tok in (1, 64, 512, 2048, 8192, 32768):
        res = torch.randn(n_tok, HIDDEN, device=dev, dtype=dt)
        out = torch.randn(n_tok, HIDDEN, device=dev, dtype=dt)
        w = (1 + 0.1 * torch.randn(HIDDEN, device=dev)).to(dt)
        h1, y1 = fused(res, out, w)
        h0, y0 = reference(res, out, w)
        err_h = (h1.float() - h0.float()).abs().max().item()
        err_y = (y1.float() - y0.float()).abs().max().item()
        t_ref = triton.testing.do_bench(lambda: reference(res, out, w))
        t_fus = triton.testing.do_bench(lambda: fused(res, out, w))
        compiled(res, out, w)                                     # compile outside the timing
        t_cmp = triton.testing.do_bench(lambda: compiled(res, out, w))
        moved = 4 * n_tok * HIDDEN * 2 + HIDDEN * 2                     # bytes: read res, out; write h, y (bf16) + w
        rows.append({"tokens": n_tok, "max_err_h": err_h, "max_err_y": err_y, "torch_ms": round(t_ref, 4),
                     "triton_ms": round(t_fus, 4), "speedup": round(t_ref / t_fus, 2),
                     "compile_ms": round(t_cmp, 4), "speedup_vs_compile": round(t_cmp / t_fus, 2),
                     "triton_GBps": round(moved / (t_fus * 1e-3) / 1e9, 1)})
        print(json.dumps(rows[-1]), flush=True)
    ok = all(r["max_err_h"] <= 0.0625 and r["max_err_y"] <= 0.0625 for r in rows)      # bf16 rounding level
    result = {"device": torch.cuda.get_device_name(), "hidden": HIDDEN, "correct_within_bf16": ok, "rows": rows}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("correct within bf16 tolerance:", ok)


if __name__ == "__main__":
    main()

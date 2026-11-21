"""QLoRA training of Granite 4.1-3B-base on tool-calling chats (laptop dry run or HPC job)."""
import argparse
import functools
import json
import math
import os
import signal

import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from chat_format import IGNORE
from model import add_lora, load_base
from train_data import ChatDataset, collate


ROOT = Path(__file__).resolve().parent.parent


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)                  # output folder: runs/<run>/
    p.add_argument("--data", default=str(ROOT / "data"))
    p.add_argument("--max-len", type=int, default=4096)     # longer chats are skipped
    p.add_argument("--limit", type=int, default=None)       # use only N training chats (data-size ablation)
    p.add_argument("--steps", type=int, default=1000)       # optimizer steps
    p.add_argument("--batch", type=int, default=1)          # chats per forward pass
    p.add_argument("--accum", type=int, default=16)         # forward passes per optimizer step
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--save-every", type=int, default=50)    # checkpoint every N optimizer steps
    p.add_argument("--stop-after", type=int, default=None)  # testing only: act as if STOP appeared after step N
    p.add_argument("--out", default=None)                    # output folder (the HPC job folder); default runs/<run>/
    p.add_argument("--eval-every", type=int, default=50)     # validation loss every N steps (0 = never)


    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cudnn-sdp", action="store_true")    # avoid the cuDNN attention kernel (crashed in par3)
    p.add_argument("--bf16-base", action="store_true")       # ablation: 16-bit LoRA on a bf16 base (default 4-bit QLoRA)
    return p.parse_args(argv)


def lr_at(step, args):
    """Linear warm-up, then cosine decay down to 10% of the peak."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


@functools.lru_cache(maxsize=4)
def epoch_batches(epoch, lengths, batch, seed):
    """All micro-batches of one epoch: shuffle, sort by length inside chunks of 16 batches (little padding), shuffle
    the batches. Depends only on (epoch, seed), so a resumed run sees exactly the same batches."""
    g = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(lengths), generator=g).tolist()
    chunk, batches = batch * 16, []
    for i in range(0, len(order), chunk):
        part = sorted(order[i:i + chunk], key=lambda j: lengths[j])
        batches += [part[k:k + batch] for k in range(0, len(part) - batch + 1, batch)]
    perm = torch.randperm(len(batches), generator=g).tolist()
    return tuple(tuple(batches[p]) for p in perm)


def batch_indices(step, micro, lengths, args):
    """Positions (in the kept-chat list) of micro-batch `micro` of optimizer step `step`; same on every rerun."""
    per_epoch = len(epoch_batches(0, lengths, args.batch, args.seed))     # equal for every epoch (n is fixed)
    epoch, k = divmod(step * args.accum + micro, per_epoch)
    return list(epoch_batches(epoch, lengths, args.batch, args.seed)[k])


STOP_REQUESTED = False


def request_stop(signum, frame):
    """Ctrl+C on the laptop, or the HPC wrapper's SIGINT 15 min before the time limit."""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"signal {signum}: will save and stop after this step", flush=True)


def save_ckpt(path, step, model, opt):
    """Everything needed to continue exactly: trained weights, optimizer, step, random states."""
    state = {
        "step": step,
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "opt": opt.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)             # atomic: after a crash there is the old file or the new one, never half


def load_ckpt(path, model, opt):
    """Put a saved state back; returns the number of optimizer steps already done."""
    state = torch.load(path, map_location="cpu", weights_only=False)   # RNG states must stay on the CPU
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in state["trainable"].items():
            params[n].copy_(t)
    opt.load_state_dict(state["opt"])
    torch.set_rng_state(state["rng_cpu"])
    torch.cuda.set_rng_state(state["rng_cuda"])
    return state["step"]

def evaluate(model, data, idx, pad_id):
    """Token-weighted mean loss on held-out chats (no dropout, no gradients, no random numbers used)."""
    model.eval()
    loss_sum, tok_sum = 0.0, 0
    with torch.no_grad():
        for i in idx:
            b = {k: v.cuda() for k, v in collate([data[i]], pad_id).items()}
            n_tok = (b["labels"][:, 1:] != IGNORE).sum().item()
            if n_tok == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_sum += model(**b).loss.item() * n_tok
            tok_sum += n_tok
    model.train()
    return loss_sum / max(tok_sum, 1)




def main():
    args = get_args()
    if args.no_cudnn_sdp:                         # par3 crash: cuDNN SDPA "mha_graph.execute ... is_good()" with 3 runs on
        torch.backends.cuda.enable_cudnn_sdp(False)   # one slice; flash / mem-efficient SDPA kernels are used instead
    torch.manual_seed(args.seed)
    out = Path(args.out) if args.out else ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    ckpt, log_path = out / "last.pt", out / "train_log.jsonl"
    if (out / "DONE").exists():
        print("DONE already exists; nothing to do", flush=True)
        return
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    tok = AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-3b")
    train = ChatDataset(Path(args.data) / "train.jsonl", tok, limit=args.limit, seed=args.seed)
    keep = [i for i, ex in enumerate(train.examples) if len(ex["input_ids"]) <= args.max_len]
    lengths = tuple(len(train.examples[i]["input_ids"]) for i in keep)    # for length-grouped batches
    print(f"training chats: {len(keep)} of {len(train)} (max_len {args.max_len})", flush=True)

    val = ChatDataset(Path(args.data) / "val.jsonl", tok)
    val_keep = [i for i, ex in enumerate(val.examples) if len(ex["input_ids"]) <= args.max_len]
    val_path = out / "val_log.jsonl"


    four_bit = not args.bf16_base
    model = add_lora(load_base(four_bit=four_bit), r=args.rank, alpha=2 * args.rank, four_bit=four_bit)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    start = 0
    if ckpt.exists():
        start = load_ckpt(ckpt, model, opt)
        old = log_path.read_text(encoding="utf-8").splitlines()[:start] if log_path.exists() else []
        log_path.write_text("".join(line + "\n" for line in old), encoding="utf-8")  # forget steps after the ckpt
        if val_path.exists():
            kept = [line for line in val_path.read_text(encoding="utf-8").splitlines() if json.loads(line)["step"] <= start]
            val_path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")

        print(f"resumed from step {start}", flush=True)

    model.train()
    t0 = time.monotonic()
    nan_skips = 0
    for step in range(start, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        loss_sum, tok_sum = 0.0, 0
        for micro in range(args.accum):
            idx = [keep[i] for i in batch_indices(step, micro, lengths, args)]
            b = {k: v.cuda() for k, v in collate([train[i] for i in idx], tok.pad_token_id).items()}
            n_tok = (b["labels"][:, 1:] != IGNORE).sum().item()   # tokens the model's loss averages over
            if n_tok == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**b).loss
            (loss * n_tok).backward()                  # a sum, so every trained token counts equally
            loss_sum += loss.item() * n_tok
            tok_sum += n_tok
        for p in params:
            if p.grad is not None:
                p.grad /= max(tok_sum, 1)              # now the mean over all trained tokens of this step
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        skipped = not math.isfinite(gnorm)            # a NaN/inf gradient would poison every later step (sft3)
        if skipped:
            nan_skips += 1
            print(f"step {step + 1}: non-finite gradient, update skipped ({nan_skips} so far)", flush=True)
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        rec = {"step": step + 1, "loss": round(loss_sum / max(tok_sum, 1), 4), "lr": lr_at(step, args),
               "gnorm": round(gnorm, 3), "skipped": skipped, "tokens": tok_sum,
               "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "sec": round(time.monotonic() - t0, 1)}
        print(json.dumps(rec), flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        done = step + 1 == args.steps
        if args.eval_every and ((step + 1) % args.eval_every == 0 or done):
            vrec = {"step": step + 1, "val_loss": round(evaluate(model, val, val_keep, tok.pad_token_id), 4),
                    "sec": round(time.monotonic() - t0, 1)}
            print(json.dumps(vrec), flush=True)
            with open(val_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(vrec) + "\n")
            model.save_pretrained(out / f"adapter_step{step + 1}")      # keep every evaluated adapter (B5)
            best_path = out / "best.json"
            best = json.loads(best_path.read_text()) if best_path.exists() else None
            if best is None or vrec["val_loss"] < best["val_loss"]:
                best_path.write_text(json.dumps(vrec))                  # the step whose adapter goes to BFCL

        stop = STOP_REQUESTED or (out / "STOP").exists() or step + 1 == args.stop_after
        if nan_skips > 10:
            print("more than 10 non-finite gradients: stopping, check the run", flush=True)
            stop = True
        if done or stop or (step + 1) % args.save_every == 0:
            save_ckpt(ckpt, step + 1, model, opt)
        if done:
            model.save_pretrained(out / "adapter")     # the small final adapter: LoRA + token rows
            (out / "DONE").touch()
            print("finished: adapter saved, DONE written", flush=True)
        elif stop:
            print(f"stopped after step {step + 1}; run the same command again to resume", flush=True)
            return


if __name__ == "__main__":
    main()

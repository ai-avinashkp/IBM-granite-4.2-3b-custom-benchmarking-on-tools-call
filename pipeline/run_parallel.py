"""Run several training experiments at the same time on the session's ONE slice (user rule: pack runs into one slice
when memory allows; one 4-bit run needs ~9 GB, a 47 GB slice holds 3). Each run writes to <jobs>/ibm_<name>/ exactly as
a single submitted run would (train_log, val_log, adapters, best.json, DONE), so fetch and eval work unchanged.

  python run_parallel.py --exp sft4s1,sft4s2 --jobs <jobs folder> --data <data folder>
Exits 0 only if every run wrote DONE. SIGINT (the wrapper's stop before walltime) reaches all runs; they checkpoint.
"""
import argparse
import subprocess
import time
import sys
from pathlib import Path

COMMON = ["--steps", "460", "--accum", "16", "--batch", "1", "--max-len", "4096", "--lr", "2e-4", "--warmup", "30",
          "--rank", "16", "--seed", "0", "--eval-every", "50", "--save-every", "50"]
EXP = {                                             # data-v3 recipe (sft4) unless stated; later flags override COMMON
    "sft4s1": ["--seed", "1"],
    "sft4s2": ["--seed", "2"],
    "sft4r8": ["--rank", "8"],
    "sft4r64": ["--rank", "64"],
    "sft4bf16": ["--bf16-base"],
    "sft4d30": ["--limit", "2318", "--steps", "145", "--warmup", "10", "--eval-every", "25"],   # 30% of data, 1 epoch
    "sft4d10": ["--limit", "773", "--steps", "48", "--warmup", "5", "--eval-every", "8"],      # 10% of data, 1 epoch
}
# Reruns after the par3 cuDNN-attention crash (bf16 and d10 failed, d30 finished): same settings + --no-cudnn-sdp.
EXP["sft4bf16"] = EXP["sft4bf16"] + ["--no-cudnn-sdp"]
EXP["sft4d10"] = EXP["sft4d10"] + ["--no-cudnn-sdp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    procs = []
    for name in a.exp.split(","):
        out = Path(a.jobs) / f"ibm_{name}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "train.py", "--run", name, "--out", str(out), "--data", a.data] + COMMON + EXP[name]
        (out / "cmd.sh").write_text(" ".join(cmd) + "\n")
        log = open(out / "out.log", "a")
        procs.append((name, out, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"started {name} -> {out}", flush=True)
    running = {p: (name, out, log) for name, out, p, log in procs}
    while running:                                  # reap every child AS SOON AS it exits: waiting in order left an
        for p in list(running):                     # early-exiting run as a <defunct> zombie, which the HPC monitor
            rc = p.poll()                           # reported as a login-node process (admin e-mail, 2026-11-20)
            if rc is not None:
                name, out, log = running.pop(p)
                log.close()
                print(f"{name}: exit {rc}, DONE={(out / 'DONE').exists()}", flush=True)
        time.sleep(5)
    sys.exit(0 if all((out / "DONE").exists() for _, out, _, _ in procs) else 1)


if __name__ == "__main__":
    main()

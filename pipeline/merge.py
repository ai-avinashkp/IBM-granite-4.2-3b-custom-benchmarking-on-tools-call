"""Build a folder that vLLM can serve: Granite base weights (bf16), optionally with our adapter merged in,
plus the original config and the Granite 4.2 tokenizer files."""
import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM

BASE = "ibm-granite/granite-4.1-3b-base"
TOKENIZER = "ibm-granite/granite-4.2-3b"          # chat template + renamed special tokens (Blocks 1-3)
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "chat_template.jinja", "vocab.json", "merges.txt"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None)       # adapter folder from train.py; omit = the plain base model
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()   # adds (alpha/r)·B·A and the token rows
    model.save_pretrained(out)

    # transformers 5 writes RoPE as "rope_parameters"; vLLM's transformers 4.51 would not find rope_theta and would
    # silently use 10,000 instead of 10,000,000. The merge does not change the architecture, so the base's original
    # config files are exactly right.
    base_dir = Path(snapshot_download(BASE, allow_patterns=["config.json", "generation_config.json"]))
    tok_dir = Path(snapshot_download(TOKENIZER, allow_patterns=TOKENIZER_FILES))
    for f in ("config.json", "generation_config.json"):
        shutil.copy(base_dir / f, out / f)
    for f in TOKENIZER_FILES:
        shutil.copy(tok_dir / f, out / f)
    print(f"saved {out}: {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()

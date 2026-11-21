"""Phase 5 (release): runs the model-card usage snippet as written (4-bit base + LoRA adapter + 4.2 tokenizer) on
two prompts, one that needs the tool and one that does not, and saves the replies.
  python card_check.py --adapter <adapter folder> --out <txt>
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tokenizer", default="ibm-granite/granite-4.2-3b")   # the HF repo ships these files itself
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    base = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-4.1-3b-base", device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model = PeftModel.from_pretrained(base, a.adapter).eval()
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    lines = []
    for q in ["What's the weather in Jalandhar right now?", "Write a two-line poem about rain."]:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], tools=tools, add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt", return_dict=True).to(0)
        out = model.generate(**ids, max_new_tokens=80, do_sample=False)
        lines += [f"Q: {q}", tok.decode(out[0, ids["input_ids"].shape[1]:]), "-" * 40]
        print("\n".join(lines[-3:]), flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

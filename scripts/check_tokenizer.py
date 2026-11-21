"""Step 0.1: does the Granite base tokenizer know the 4.2 chat and tool-call markers?"""
from transformers import AutoTokenizer

BASE = "ibm-granite/granite-4.1-3b-base"   # the model we will fine-tune
REF = "ibm-granite/granite-4.2-3b"         # IBM's tuned model (reference + chat template)
MARKERS = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<think>", "</think>"]


def show(name):
    tok = AutoTokenizer.from_pretrained(name)   # downloads tokenizer files only, no weights
    print(f"\n== {name}  (vocab {len(tok)})")
    for m in MARKERS:
        ids = tok.encode(m, add_special_tokens=False)
        status = "single token" if len(ids) == 1 else f"SPLIT into {len(ids)}"
        print(f"  {m:14s} -> {ids}  {status}")
    print("  has chat template:", tok.chat_template is not None)



def compare(name_a, name_b):
    """Show every token id whose string differs between two tokenizers."""
    ta = AutoTokenizer.from_pretrained(name_a)
    tb = AutoTokenizer.from_pretrained(name_b)
    ia = {i: t for t, i in ta.get_vocab().items()}   # flip: id -> token string
    ib = {i: t for t, i in tb.get_vocab().items()}
    diff = [i for i in sorted(set(ia) | set(ib)) if ia.get(i) != ib.get(i)]
    print(f"\n== {name_a} vs {name_b}: {len(diff)} ids differ")
    for i in diff[:30]:
        print(f"  id {i:6d}: {ia.get(i)!r:30} -> {ib.get(i)!r}")
    for attr in ("bos_token", "eos_token", "pad_token"):
        print(f"  {attr:9s}: {getattr(ta, attr)!r:30} -> {getattr(tb, attr)!r}")


if __name__ == "__main__":
    for name in (BASE, REF):
        show(name)
    compare(BASE, REF)

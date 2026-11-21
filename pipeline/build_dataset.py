"""Step 1: build the training/validation files from all cleaned sources (run once, then upload to the HPC)."""
import json
import random
from pathlib import Path
from collections import Counter

from transformers import AutoTokenizer

from chat_format import build_example
from hermes import load_hermes

OUT = Path(__file__).resolve().parent.parent / "data"
MAX_LEN = 4096          # p99 is 2630; only 2 chats are longer (2026-11-18)
VAL_FRACTION = 0.05
SEED = 0
MAX_VAL_GROUP = 50      # groups bigger than this always stay in train


tok = AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-3b")
chats = []
for config in ("func_calling", "glaive_func_calling"):
    chats += load_hermes(config)[0]

kept = [c for c in chats if len(build_example(tok, c[0], c[1])["input_ids"]) <= MAX_LEN]


def called(chat):
    """Names of the functions the assistant actually calls in this chat (decoy tools are ignored)."""
    return sorted({c["function"]["name"] for m in chat[0] for c in m.get("tool_calls", [])})


def group_ids(chats):
    """Union-find: chats that call a common function end up in the same group."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]      # path halving keeps the trees flat
            x = parent[x]
        return x

    for c in chats:
        names = called(c)
        for n in names[1:]:
            parent[find(n)] = find(names[0])   # link every function called in the same chat
    return [("fn:" + find(called(c)[0])) if called(c) else ("chat:" + c[0][0]["content"][:50].lower())
            for c in chats]


keys = group_ids(kept)
sizes = Counter(keys)
groups = sorted(sizes)
random.Random(SEED).shuffle(groups)
target, val_groups, n_val = VAL_FRACTION * len(kept), set(), 0
for g in groups:
    if n_val >= target:
        break
    if sizes[g] <= MAX_VAL_GROUP:              # big groups stay in train, so one group cannot dominate val
        val_groups.add(g)
        n_val += sizes[g]
splits = {"val": [c for c, k in zip(kept, keys) if k in val_groups],
          "train": [c for c, k in zip(kept, keys) if k not in val_groups]}
print(f"{len(groups)} groups, {len(val_groups)} in validation")



OUT.mkdir(exist_ok=True)
for name, rows in splits.items():
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for messages, tools, chat_id in rows:
            f.write(json.dumps({"id": chat_id, "messages": messages, "tools": tools}, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} chats -> {OUT / f'{name}.jsonl'}")
print(f"dropped {len(chats) - len(kept)} chats longer than {MAX_LEN} tokens")

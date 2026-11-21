"""data-v2 = data-v1 + "tools offered, none fits, answer without a call" chats (fixes the irrelevance drop of sft2).

The 784 train / 80 val tool-free Glaive chats get 1-3 random tools from the SAME split's tool pool (val keeps
unseen tools). A tool is skipped if a word of its name (4+ letters) appears in the user's text, so it really
does not fit. Train gets COPIES_TRAIN copies of each with different tool sets; val gets one.
"""
import json
import random
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "v2"
COPIES_TRAIN, COPIES_VAL, SEED = 2, 1, 0


def tool_pool(rows):
    pool = {}
    for r in rows:
        for t in r["tools"] or []:
            pool.setdefault(t["function"]["name"], t)
    return [pool[k] for k in sorted(pool)]


def unrelated_tools(chat, pool, rng):
    text = " ".join(m["content"] for m in chat["messages"] if m["role"] == "user").lower()
    words = set(re.findall(r"[a-z]{4,}", text))
    picks = []
    for t in rng.sample(pool, min(len(pool), 40)):
        name_words = set(re.findall(r"[a-z]{4,}", t["function"]["name"].lower().replace("_", " ")))
        if not name_words & words:
            picks.append(t)
        if len(picks) == rng.randint(1, 3):
            break
    return picks or None


def build(split, copies, rng):
    rows = [json.loads(line) for line in open(DATA / f"{split}.jsonl", encoding="utf-8")]
    pool = tool_pool(rows)
    out, added = [], 0
    for r in rows:
        if r["tools"]:
            out.append(r)
            continue
        for c in range(copies):
            tools = unrelated_tools(r, pool, rng)
            if tools:
                out.append({"id": f"{r['id']}/irrel{c}", "messages": r["messages"], "tools": tools})
                added += 1
    rng.shuffle(out)
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    no_call = sum(1 for r in out if r["tools"] and not any(m.get("tool_calls") for m in r["messages"]))
    print(f"{split}: {len(rows)} -> {len(out)} chats; +{added} 'tools but no call' (now {no_call / len(out):.1%}); "
          f"tool pool {len(pool)}")


if __name__ == "__main__":
    rng = random.Random(SEED)
    build("train", COPIES_TRAIN, rng)
    build("val", COPIES_VAL, rng)

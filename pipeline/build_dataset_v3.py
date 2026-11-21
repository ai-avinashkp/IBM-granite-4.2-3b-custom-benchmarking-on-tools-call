"""data-v3 = data-v2 + HARD "no tool fits" chats (the gate1 measurement showed the model cannot tell when the offered
tools cannot do the request: sft3b blocks only 58% of hard cases even at tau=0.99).

A hard case reuses a real training request whose first assistant turn calls a tool, removes the called tool(s) from
the menu (random other train tools fill an empty menu), and replaces the reply with a short refusal that says what
the remaining tools can do. Built from the TRAIN split only; val is left as in data-v2 (gate_dev builds its own hard
cases from val, so there is no leakage).
"""
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SRC, OUT = DATA / "v2", DATA / "v3"
N_HARD, SEED = 1000, 0
TEMPLATES = [
    "I'm sorry, but I can't do that with the tools I have. I can only {caps}.",
    "Unfortunately none of my available tools can help with that request. My tools let me {caps}.",
    "I don't have a tool for that. What I can do here is {caps}.",
]


def capability(tool):
    f = tool["function"]
    d = (f.get("description") or f["name"].replace("_", " ")).strip().rstrip(".")
    return d[0].lower() + d[1:] if d else f["name"]


def main():
    rng = random.Random(SEED)
    rows = [json.loads(line) for line in open(SRC / "train.jsonl", encoding="utf-8")]
    pool = sorted({t["function"]["name"]: t for r in rows for t in (r["tools"] or [])}.items())
    candidates = []
    for r in rows:
        m = r["messages"]
        k = next((i for i, x in enumerate(m) if x["role"] == "assistant"), None)
        if k is not None and r["tools"] and m[k].get("tool_calls") and "/irrel" not in r["id"]:
            candidates.append((r, k))
    rng.shuffle(candidates)
    hard = []
    for r, k in candidates[:N_HARD]:
        names = {c["function"]["name"] for c in r["messages"][k]["tool_calls"]}
        rest = [t for t in r["tools"] if t["function"]["name"] not in names]
        if not rest:
            rest = [t for _, t in rng.sample([p for p in pool if p[0] not in names], rng.randint(1, 2))]
        caps = " and ".join(capability(t) for t in rest[:2])
        reply = rng.choice(TEMPLATES).format(caps=caps)
        hard.append({"id": f"{r['id']}/hard", "messages": r["messages"][:k] + [{"role": "assistant", "content": reply}],
                     "tools": rest})
    out = rows + hard
    rng.shuffle(out)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "train.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "val.jsonl").write_text((SRC / "val.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"train: {len(rows)} + {len(hard)} hard = {len(out)} chats; val copied from v2")
    print("example:", json.dumps(hard[0]["messages"][-2:], ensure_ascii=False)[:400], "| tools:",
          [t["function"]["name"] for t in hard[0]["tools"]])


if __name__ == "__main__":
    main()

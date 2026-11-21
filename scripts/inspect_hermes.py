"""Step 0.4: look at the raw training data before writing any converter."""
from collections import Counter
from datasets import load_dataset

NAME = "NousResearch/hermes-function-calling-v1"
CONFIG = "glaive_func_calling"     # later also: "func_calling", "glaive_func_calling"

ds = load_dataset(NAME, CONFIG, split="train")
print(ds)                              # number of rows and the column names

row = ds[0]
print("\n--- tools column (first 300 chars) ---\n", str(row.get("tools"))[:300])
for turn in row["conversations"]:
    print(f"\n--- {turn['from']} ---\n{turn['value'][:400]}")

roles = Counter(t["from"] for r in ds for t in r["conversations"])
calls = Counter(t["value"].count("<tool_call>")
                for r in ds for t in r["conversations"] if t["from"] == "gpt")
print("\nturns per role:", dict(roles))
print("tool calls per assistant turn:", dict(sorted(calls.items())))

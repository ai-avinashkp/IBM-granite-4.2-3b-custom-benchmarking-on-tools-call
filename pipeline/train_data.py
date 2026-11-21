"""Training data: JSONL chats -> padded batches of input_ids / labels / attention_mask."""
import json
import random

import torch
from torch.utils.data import Dataset

from chat_format import IGNORE, build_example


class ChatDataset(Dataset):
    def __init__(self, path, tok, limit=None, seed=0):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        if limit:                                       # data-size ablation / quick tests
            random.Random(seed).shuffle(rows)           # same seed -> same order, so 10% ⊂ 30% ⊂ 100%
            rows = rows[:limit]
        self.examples = [build_example(tok, r["messages"], r["tools"]) for r in rows]
        self.ids = [r["id"] for r in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id):
    """Right-pad every example to the longest one in the batch."""
    n = max(len(ex["input_ids"]) for ex in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for row, ex in enumerate(batch):
        k = len(ex["input_ids"])
        ids[row, :k] = torch.tensor(ex["input_ids"])
        labels[row, :k] = torch.tensor(ex["labels"])
        mask[row, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


if __name__ == "__main__":
    from functools import partial
    from pathlib import Path
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-3b")
    data = Path(__file__).resolve().parent.parent / "data"
    ds = ChatDataset(data / "val.jsonl", tok)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=partial(collate, pad_id=tok.pad_token_id))
    batch = next(iter(loader))
    for k, v in batch.items():
        print(f"{k:15s} {tuple(v.shape)}")
    trained = (batch["labels"] != IGNORE).sum().item()
    real = batch["attention_mask"].sum().item()
    print(f"pad id {tok.pad_token_id} | real tokens {real}, trained {trained} ({trained / real:.0%})")

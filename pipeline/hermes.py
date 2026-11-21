"""Convert one Hermes function-calling row into (messages, tools) for build_example."""
import ast
import json
import re
from collections import Counter

from datasets import load_dataset


CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)   # one match per call
ROLE = {"human": "user", "gpt": "assistant", "tool": "tool"}


def parse_call(text):
    try:
        return json.loads(text)                  # proper JSON: {"name": ..., "arguments": ...}
    except json.JSONDecodeError:
        return ast.literal_eval(text)            # Python-dict style: {'name': ..., 'arguments': ...}


def hermes_to_chat(row):
    tools = json.loads(row["tools"]) if row.get("tools") else None
    tools = [t if "function" in t else {"type": "function", "function": t} for t in tools or []] or None

    messages = []
    for turn in row["conversations"]:
        if turn["from"] == "system":
            continue                             # the 4.2 template writes its own tool prompt
        role, value = ROLE[turn["from"]], turn["value"]
        if role == "assistant" and "<tool_call>" in value:
            calls = [parse_call(c) for c in CALL_RE.findall(value)]
            text = value[:value.index("<tool_call>")].strip()      # words before the first call
            messages.append({"role": "assistant", "content": text, "tool_calls": [
                {"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
        elif role == "tool":
            parts = re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", value, re.S)
            messages.extend({"role": "tool", "content": p} for p in parts or [value])  # one per result
        else:
            messages.append({"role": role, "content": value})

    return messages, tools



def check_calls(messages, tools):
    """Return None if every call matches the tool schemas, else a short reason."""
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools or []}
    for m in messages:
        for call in m.get("tool_calls", []):
            name, args = call["function"]["name"], call["function"]["arguments"]
            if name not in schemas:
                return "unknown function"
            if not isinstance(args, dict):
                return "arguments not a dict"
            allowed = schemas[name].get("properties", {})
            if any(k not in allowed for k in args):
                return "unknown argument"
            if any(k not in args for k in schemas[name].get("required", [])):
                return "missing required argument"
    return None

def load_hermes(config):
    """All clean chats of one Hermes config as (messages, tools, chat_id), plus drop reasons."""
    ds = load_dataset("NousResearch/hermes-function-calling-v1", config, split="train")
    chats, reasons = [], Counter()
    for row in ds:
        if row.get("subcategory") == "Json Schema":
            reasons["json-schema extraction task (broken calls)"] += 1
            continue
        try:
            messages, tools = hermes_to_chat(row)
        except (SyntaxError, ValueError) as e:
            reasons[f"parse error: {type(e).__name__}"] += 1
            continue
        reason = check_calls(messages, tools)
        if reason:
            reasons[reason] += 1
            continue
        chats.append((messages, tools, f"hermes/{config}/{row['id']}"))
    return chats, reasons


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from chat_format import build_example    # works when run as: python pipeline/hermes.py

    tok = AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-3b")
    all_chats = []
    for config in ("func_calling", "glaive_func_calling"):
        chats, reasons = load_hermes(config)
        print(f"{config}: kept {len(chats)}, dropped {dict(reasons)}")
        all_chats += chats
    lengths = sorted(len(build_example(tok, m, t)["input_ids"]) for m, t, _ in all_chats)

    def pct(p):
        return lengths[int(p / 100 * (len(lengths) - 1))]

    print(f"total {len(lengths)} chats; tokens p50 {pct(50)}, p95 {pct(95)}, p99 {pct(99)}, max {lengths[-1]}")
    for cap in (2048, 4096):
        print(f"  longer than {cap}: {sum(n > cap for n in lengths)}")

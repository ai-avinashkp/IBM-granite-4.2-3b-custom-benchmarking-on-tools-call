"""Parse Granite 4.2 XML tool calls back into Python objects, using the tool schemas for the types."""
import ast
import json
import re

CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def convert(value, schema):
    """Text inside <parameter> -> the type the schema asks for (like vLLM's qwen3_coder parser)."""
    kind = (schema or {}).get("type", "string")
    if kind == "string":
        return value
    try:
        if kind in ("integer", "int"):
            return int(value)
        if kind in ("number", "float"):
            return float(value)
        if kind in ("boolean", "bool"):
            return value.strip().lower() == "true"
        return json.loads(value)                     # array / object / anything else
    except ValueError:                               # json.JSONDecodeError is a ValueError too
        try:
            return ast.literal_eval(value)           # Python-style lists or dicts
        except (ValueError, SyntaxError):
            return value                             # keep the text; BFCL's checker will judge it


def parse_tool_calls(text, tools):
    """[{'name': ..., 'arguments': {...}}, ...] for every <tool_call> block in the model's output."""
    props = {t["function"]["name"]: t["function"].get("parameters", {}).get("properties", {}) for t in tools or []}
    calls = []
    for name, body in CALL_RE.findall(text):
        name = name.strip()
        schema = props.get(name, {})
        args = {p.strip(): convert(v, schema.get(p.strip())) for p, v in PARAM_RE.findall(body)}
        calls.append({"name": name, "arguments": args})
    return calls


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-3b")
    tools = [{"type": "function", "function": {"name": "book", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "nights": {"type": "integer"}, "budget": {"type": "number"},
        "breakfast": {"type": "boolean"}, "guests": {"type": "array"}, "prefs": {"type": "object"}}}}}]
    args = {"city": "Jalandhar", "nights": 3, "budget": 99.5, "breakfast": True,
            "guests": ["Asha", "Ravi"], "prefs": {"floor": "high", "quiet": True}}
    messages = [{"role": "user", "content": "Book it."},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "book", "arguments": args}}]}]
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, enable_thinking=False)
    reply = text.split("<|im_start|>assistant")[-1]
    print(reply)
    got = parse_tool_calls(reply, tools)
    print("parsed:", got)
    print("round trip exact:", got == [{"name": "book", "arguments": args}])

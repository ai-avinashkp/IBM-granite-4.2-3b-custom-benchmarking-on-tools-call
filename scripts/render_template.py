"""Step 0.2: see the exact text the 4.2 chat template builds for a tool-calling chat."""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-3b")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]

messages = [
    {"role": "user", "content": "What is the weather in Jalandhar?"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Jalandhar"}}}]},
    {"role": "tool", "content": '{"temp_c": 18, "sky": "clear"}'},
    {"role": "assistant", "content": "It is 18 C and clear in Jalandhar."},
]

text = tok.apply_chat_template(messages, tools=tools, tokenize=False, enable_thinking=False)
print(text)

ids = tok(text, add_special_tokens=True)["input_ids"]
print("\nbos token:", repr(tok.bos_token), "id", tok.bos_token_id)
print("first 5 ids:", ids[:5], "| tokenizer added bos itself:", ids[0] == tok.bos_token_id)

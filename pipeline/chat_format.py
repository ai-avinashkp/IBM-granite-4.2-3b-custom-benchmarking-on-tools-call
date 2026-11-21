"""Turn one chat (messages + tools) into training ids, with loss only on the assistant's replies."""
from transformers import AutoTokenizer

IM_START, IM_END = 100256, 100257   # <|im_start|>, <|im_end|> (ids from scripts/check_tokenizer.py)
EMPTY_THINK = [100274, 100275]      # <think></think>: part of the prompt in non-thinking mode
IGNORE = -100                       # label value that the loss function skips


def build_example(tok, messages, tools=None):
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, enable_thinking=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    header = tok("assistant\n", add_special_tokens=False)["input_ids"]
    labels = [IGNORE] * len(ids)
    i = 0
    while i < len(ids):
        if ids[i] == IM_START and ids[i + 1:i + 1 + len(header)] == header:
            j = i + 1 + len(header)                  # first token after "<|im_start|>assistant\n"
            if ids[j:j + 2] == EMPTY_THINK:
                j += 2                               # do not train on <think></think>
            while j < len(ids) and ids[j] != IM_END:
                labels[j] = ids[j]                   # train on the reply ...
                j += 1
            if j < len(ids):
                labels[j] = IM_END                   # ... and on <|im_end|>, so it learns to stop
            i = j
        i += 1
    return {"input_ids": ids, "labels": labels}


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-3b")
    messages = [
        {"role": "user", "content": "Weather in Jalandhar?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Jalandhar"}}}]},
        {"role": "tool", "content": '{"temp_c": 18}'},
        {"role": "assistant", "content": "It is 18 C."},
    ]
    ex = build_example(tok, messages)
    trained = [t for t, lab in zip(ex["input_ids"], ex["labels"]) if lab != IGNORE]
    print(f"{len(ex['input_ids'])} tokens in total, {len(trained)} trained")
    print("trained text:", repr(tok.decode(trained)))
    prompt = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    print("inference prompt ends with:", repr(prompt[-35:]))

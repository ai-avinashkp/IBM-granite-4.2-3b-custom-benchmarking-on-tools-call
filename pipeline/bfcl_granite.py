"""BFCL handler for the Granite 4.2 format (ChatML turns + XML tool calls): granite-4.2-3b and our fine-tunes.

Optional call-or-answer gate (off unless GRANITE_GATE_TAU is set; tau is chosen on OUR validation data with
pipeline/gate_dev.py, never on BFCL): p_call = P(<tool_call> starts the reply); if p_call < tau the reply is
generated with <tool_call> blocked. Each decision is recorded at the top of reasoning_content.
"""
import math
import os
import threading
import time
from typing import Any

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call, convert_to_tool
from overrides import override

from granite_xml import parse_tool_calls

TOOL_CALL_ID, NL, NL_ID = 100270, "\n", 198              # ids checked with the 4.2 tokenizer (gate_dev.py)
GATE_TAU = float(os.environ["GRANITE_GATE_TAU"]) if os.environ.get("GRANITE_GATE_TAU") else None


class Granite42Handler(OSSHandler):
    """The prompt comes from the model's own chat template, so evaluation sees exactly the training format."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._local = threading.local()              # BFCL runs many test cases in parallel threads

    def _tools(self, function):
        """BFCL function docs -> OpenAI-style tools ('dict' -> 'object', '.' in names -> '_')."""
        tools = convert_to_tool(function, GORILLA_TO_OPENAPI, ModelStyle.OSSMODEL)
        return [t if "function" in t else {"type": "function", "function": t} for t in tools]

    @override
    def _format_prompt(self, messages, function):
        self._local.tools = self._tools(function)
        return self.tokenizer.apply_chat_template(messages, tools=self._local.tools, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)

    def _top_probs(self, prompt):
        """Top-20 next-token probabilities keyed by TOKEN ID (vLLM return_tokens_as_token_ids). Keying by token text
        failed silently in the first gate run (p_call = 0 for all 3641 cases, everything blocked; MISTAKES_LOG B9)."""
        r = self.client.completions.create(model=self.model_path_or_id, prompt=prompt, max_tokens=1, temperature=0.0,
                                           logprobs=20, extra_body={"return_tokens_as_token_ids": True}, timeout=600)
        top = r.choices[0].logprobs.top_logprobs[0] or {}
        return {int(k.split(":", 1)[1]): math.exp(v) for k, v in top.items() if k.startswith("token_id:")}

    def p_call(self, prompt):
        """P(reply starts with <tool_call>) = P(t1=<tool_call>) + P(t1=newline) * P(t2=<tool_call> | newline)."""
        p1, p2 = self._top_probs(prompt), self._top_probs(prompt + NL)
        if not p1:
            raise RuntimeError("gate probe returned no token-id logprobs")    # never silently fall back to 0
        return p1.get(TOOL_CALL_ID, 0.0) + p1.get(NL_ID, 0.0) * p2.get(TOOL_CALL_ID, 0.0)

    @override
    def _query_prompting(self, inference_data: dict):
        self._local.gate = None
        if GATE_TAU is None:
            return super()._query_prompting(inference_data)
        prompt = self._format_prompt(inference_data["message"], inference_data["function"])
        p = self.p_call(prompt)
        blocked = p < GATE_TAU
        self._local.gate = f"GATE p_call={p:.4f} tau={GATE_TAU} blocked={int(blocked)}"
        if not blocked:
            return super()._query_prompting(inference_data)
        inference_data["inference_input_log"] = {"formatted_prompt": prompt}
        n_in = len(self.tokenizer.tokenize(prompt))
        start = time.time()
        r = self.client.completions.create(model=self.model_path_or_id, temperature=self.temperature, prompt=prompt,
                                           max_tokens=min(4096, self.max_context_length - n_in - 2),
                                           extra_body={"logit_bias": {str(TOOL_CALL_ID): -100}}, timeout=72000)
        return r, time.time() - start

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        return {"message": [], "function": test_entry["function"]}   # FC model: the template adds the tool prompt

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        text = api_response.choices[0].text
        calls = parse_tool_calls(text, self._local.tools)
        history = {"role": "assistant", "content": text.split("<tool_call>")[0].strip()}
        if calls:
            history["tool_calls"] = [{"function": c} for c in calls]
        return {
            "model_responses": calls if calls else text,
            "model_responses_message_for_chat_history": history,
            "reasoning_content": (self._local.gate + "\n" if getattr(self._local, "gate", None) else "") + text,  # raw (B6) + gate

            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append(model_response_data["model_responses_message_for_chat_history"])
        return inference_data

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        if not isinstance(result, list):
            return []                                    # a plain-text answer: no function call
        return [{c["name"]: c["arguments"]} for c in result]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        return convert_to_function_call(self.decode_ast(result, "python", has_tool_call_tag))


if __name__ == "__main__":
    import glob
    import json
    import os
    from types import SimpleNamespace

    import bfcl_eval
    from transformers import AutoTokenizer

    h = Granite42Handler("ibm-granite/granite-4.2-3b", 0.0, "granite42-xml-FC", True)
    h.tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-3b")
    path = glob.glob(os.path.join(bfcl_eval.__path__[0], "data", "BFCL_v4_parallel.json"))[0]
    entry = json.loads(open(path).readline())
    data = h._pre_query_processing_prompting(entry)
    prompt = h._format_prompt(entry["question"][0], data["function"])
    print(prompt[-700:])
    tool = h._local.tools[0]["function"]
    p = next(iter(tool["parameters"]["properties"]))
    fake = f"\n<tool_call>\n<function={tool['name']}>\n<parameter={p}>\n1\n</parameter>\n</function>\n</tool_call>"
    resp = SimpleNamespace(choices=[SimpleNamespace(text=fake)], usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0))
    out = h._parse_query_response_prompting(resp)
    print("schema type of", p, "=", tool["parameters"]["properties"][p].get("type"))
    print("decode_ast:", h.decode_ast(out["model_responses"], "python", True))
    print("decode_execute:", h.decode_execute(out["model_responses"], True))

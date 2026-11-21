"""Load the Granite base model for QLoRA training (4-bit weights, bf16 compute)."""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE = "ibm-granite/granite-4.1-3b-base"
NEW_TOKEN_ROWS = [100256, 100270, 100271, 100272, 100273, 100274, 100275]   # untrained in the base (norm 1.251)


def load_base(four_bit=True):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",              # 4-bit NormalFloat: the QLoRA paper's data type
        bnb_4bit_use_double_quant=True,         # also quantise the scaling constants (saves ~0.4 GB)
        bnb_4bit_compute_dtype=torch.bfloat16,  # matmuls run in bf16
    ) if four_bit else None
    return AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0})

def add_lora(model, r=16, alpha=32, dropout=0.05, four_bit=True):
    """Wrap the 4-bit model with LoRA adapters plus trainable rows for the special tokens."""
    if four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant": False})
    else:                                           # 16-bit LoRA ablation: same checkpointing, weights stay bf16
        for p in model.parameters():
            p.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        trainable_token_indices={"embed_tokens": NEW_TOKEN_ROWS},
    )
    return get_peft_model(model, config)


if __name__ == "__main__":
    model = add_lora(load_base())
    model.print_trainable_parameters()
    for name, p in model.named_parameters():
        if p.requires_grad and "lora_" not in name:
            print("  also trained:", name, tuple(p.shape))
    print(f"GPU memory: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    
    # does the OUTPUT side (lm_head) also use the trained rows? (the embeddings are tied)
    lm_head = model.get_output_embeddings()
    print("output layer type:", type(lm_head).__name__)
    delta = dict(model.named_parameters())[
        "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta.default"]
    row = NEW_TOKEN_ROWS.index(100270)                 # position of <tool_call> inside the delta
    with torch.no_grad():
        h = model.get_input_embeddings()(torch.tensor([[42]], device="cuda"))   # a real hidden vector
        before = lm_head(h)[0, 0, 100270].item()
        delta[row] += h[0, 0]                           # pretend training moved the <tool_call> row
        after = lm_head(h)[0, 0, 100270].item()
        delta[row] -= h[0, 0]                           # undo the test change
    sees = abs(after - before) > 1e-3
    print(f"<tool_call> logit: before {before:.3f}, after {after:.3f} -> output side "
          f"{'USES' if sees else 'IGNORES'} the trained row")


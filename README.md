# Teaching a 3B Granite base model to call tools, and benchmarking it against IBM's own Granite 4.2

This project fine-tunes **`ibm-granite/granite-4.1-3b-base`** with QLoRA to call tools in the **Granite 4.2 XML
tool-call format**. It then compares the result with IBM's post-trained **`granite-4.2-3b`** on the
**Berkeley Function Calling Leaderboard (BFCL)**, using one harness for both models. It also covers:

- **Serving:** bf16, FP8 and 4-bit with vLLM, and GGUF on an 8 GB laptop GPU.
- **A Triton kernel** for one of Granite's fused operations, with a torch.profiler study of where the time goes.
- **The Granite decoder written from scratch** in plain PyTorch. It is bit-exact against transformers.
- **A Docling-based document agent**, with a prompt-injection test.

Everything was trained on **one H100 MIG slice** (1/7 of an H100 for most runs) and a **laptop RTX 3070 Ti (8 GB)**.

**Models on Hugging Face:**
- LoRA adapter: [avinash1954/granite-4.1-3b-toolcall-lora](https://huggingface.co/avinash1954/granite-4.1-3b-toolcall-lora)
- GGUF build: [avinash1954/granite-4.1-3b-toolcall-GGUF](https://huggingface.co/avinash1954/granite-4.1-3b-toolcall-GGUF)

> Independent project, not affiliated with or endorsed by IBM.

---

## Results at a glance

**BFCL, offline categories, same harness for every model.** granite-4.2-3b runs in non-thinking mode. "Ours" is
the mean ± sd over 3 training seeds.

| BFCL group | granite-4.1-3b-base (start) | granite-4.2-3b (IBM) | **Ours (sft4)** |
|---|---|---|---|
| Non-live calling (simple, multiple, parallel) | 0 | 37.9 | **73.0 ± 2.9** |
| `parallel` (several calls in one turn) | 0 | 2.0 | **73.2 ± 4.7** |
| Live calling (real user requests) | 0 | 56.3 | **63.5 ± 0.2** |
| Hallucination (no call when no tool fits) | – | **93.3** | 86.8 ± 1.3 |
| Multi-turn (stateful sessions) | – | **12.4** | 4.9 |
| **Mean of non-live, live, hallucination** | – | 62.5 | **74.4 ± 0.6** |
| V4-style weighting (multi-turn counts half) | – | 37.4 | **39.4** |

**In one line:** ours beats IBM's 3B at single-turn tool calling (+11.9 points; the seed spread is 0.6). The
biggest gain is on parallel calls. It is worse at multi-turn sessions, somewhat worse at refusing, and over-refuses
in multi-step agent tasks (finding 11).

### How the training data changed the result (one change per run)
| Run | Data | Non-live | Live | Hallucination | Multi-turn |
|---|---|---|---|---|---|
| sft2 | v1: cleaned Hermes + Glaive tool chats | **80.7** | **68.1** | 30.6 | 6.4 |
| sft3b | v2: + chats with unrelated tools attached ("easy negatives") | 77.0 | 67.0 | 52.6 | 4.2 |
| sft3b + inference gate | v2, plus a call/no-call probability threshold | 43.6 | 54.6 | 87.1 | – |
| **sft4** | v3: + 1,000 chats where the needed tool is removed ("hard negatives") | 71.8 | 63.4 | **86.7** | 4.9 |

### Ablations (mean of non-live, live, hallucination)
| Change from the final recipe (rank 16, 4-bit base, 100% data) | Score | Best validation loss |
|---|---|---|
| none (3 seeds) | **74.4 ± 0.6** | 0.351 |
| LoRA rank 8 | 72.3 | 0.351 |
| LoRA rank 64 | 74.3 | 0.359 |
| 16-bit base instead of 4-bit (LoRA instead of QLoRA) | 73.5 | 0.340 |
| 30% of the data | 71.2 | 0.363 |
| 10% of the data | 57.2 | 0.390 |

### Serving the merged model (vLLM 0.8.5, H100 MIG 1g.24gb = 1/7 of an H100)
| Format | BFCL mean | Weights in GPU | Time to first token | Decode, 1 request | Throughput, 32 requests |
|---|---|---|---|---|---|
| bf16 | 74.0 | 6.37 GiB | 38.9 ms | 88.4 tok/s | 1,319 tok/s |
| **FP8 (dynamic)** | **73.5** | **3.43 GiB** | **28.9 ms** | **131.7 tok/s** | **1,563 tok/s** |
| 4-bit NF4 (bitsandbytes) | 72.1 | 2.01 GiB | 65.1 ms | 80.6 tok/s | 852 tok/s |

**Laptop RTX 3070 Ti 8 GB, GGUF Q4_K_M with llama.cpp:**
- file size 1.96 GiB;
- prompt processing 4,658 tok/s (~110 ms for a 512-token prompt);
- generation **128 tok/s**;
- peak GPU memory 2.42 GiB.

### Triton kernel: fused scaled residual + RMSNorm
The kernel computes `h = residual + out × 0.22` and `y = RMSNorm(h) × w` in one pass. Hidden size is 2560, in bf16.

| Tokens | vs PyTorch eager | vs `torch.compile` | Bandwidth |
|---|---|---|---|
| 1 | 5.9× | 1.00× | 5 GB/s |
| 2,048 | 8.0× | ~1.05× | 806 GB/s |
| 32,768 | 8.1× | 1.09× | 884 GB/s |

The kernel is correct within bf16 rounding at every size.

### Inside the model: torch.profiler share and end-to-end effect
**GPU time per step in transformers (bf16, eager PyTorch ops):**
| Part | Prefill (512 tokens) | Decode (1 token) |
|---|---|---|
| MLP | 48.7% | 54.0% |
| Attention | 35.0% | 29.9% |
| **RMSNorm + residual (what the kernel replaces)** | **15.5%** | **10.7%** |
| Output head and other | 0.8% | 5.5% |

**Decoder written from scratch** in plain PyTorch, run with our weights (`pipeline/decoder_scratch.py`):
- **Exactness:** its logits are *identical* to transformers (max difference 0.0 over 8,630 positions), and greedy
  generations match token for token on 10/10 prompts.
- **Speed:** with the fused Triton kernel plugged in, it gets faster.

| Batch 1, H100 MIG 1g.24gb | Prefill 512 tokens | Decode |
|---|---|---|
| transformers (eager attention) | 144.3 ms | 24.5 tok/s |
| ours, plain PyTorch | 149.4 ms | 26.5 tok/s |
| **ours + fused Triton kernel** | **134.1 ms (−10%)** | **29.6 tok/s (+12%)** |

### Document agent (Docling + our tool loop, 48 tasks on 12 synthetic report PDFs)
| Model | Controller | Correct | lookup / sum / argmax / profit | Prompt injection followed |
|---|---|---|---|---|
| sft3b (data v2) | none | 12/48 | 12 / 0 / 0 / 0 | 0/16 |
| sft3b (data v2) | guard (at most 2 generic "use your tools" nudges) | **23/48** | 12 / 7 / 4 / 0 | 0/16 |
| **sft4 (data v3, final), 3 seeds** | none | **0/48** | 0 / 0 / 0 / 0 | 0/16 |
| sft4 (data v3, final) | guard | 0/48 | 0 / 0 / 0 / 0 | 0/16 |

The final model refuses in this agent setting (see finding 11).

---

## What we found

1. **The base model has never seen the tool tokens.**
   - In `granite-4.1-3b-base`, the embedding rows of `<tool_call>`, `</tool_call>`, `<tool_response>` and
     `</tool_response>` still have their initialisation norm (1.251).
   - So we train those 7 rows as well (PEFT `trainable_token_indices`), on top of LoRA. Without them the model
     cannot learn the format properly.
2. **Use the 4.2 tokenizer on the 4.1 weights.**
   - The two vocabularies differ only in 17 renamed special ids (for example id 100256 is `<|pad|>` in 4.1 and
     `<|im_start|>` in 4.2).
   - So the 4.2 chat template can be used directly on the base model.
3. **Data quality comes first.** 42% of the Hermes "Json Schema" rows had malformed calls. Validating every call
   against its tool schema removed them before training.
4. **Fine-tuning on tool chats teaches the model to always call.**
   - Hallucination fell from 93 (IBM) to 31 (sft2).
   - Easy negatives (unrelated tools) only got it back to 53. Hard negatives, where the right tool was removed from
     a real request, got it to 87.
5. **Data beats an inference-time gate.**
   - A calibrated "call or answer?" threshold on the first-token probability reached the same hallucination score.
   - But it blocked 45% of first decisions and cost about 33 points of calling. On our own validation set, the
     hard cases could not be separated by that probability.
6. **IBM's 3B (non-thinking) makes one call per turn.**
   - It scores 2% on `parallel`. In 121 of 200 answers it says it will make several calls, then writes only one.
   - Our parser was checked: it missed 0 of the calls the model actually wrote.
   - Training on about 600 parallel examples gave ~73%.
7. **QLoRA is enough.**
   - 4-bit vs 16-bit base: the same BFCL score within seed noise, even though 16-bit reaches a lower validation
     loss.
   - Rank 16 is enough. Data size matters most up to ~30%.
8. **Validation loss was not a reliable guide to BFCL.** The 16-bit run had the lowest validation loss (0.340 vs
   0.351) but no better BFCL score, and rank 64 had a higher loss with the same score.
9. **FP8 is the best serving format on H100.**
   - It uses half the memory and decodes 49% faster, with no measurable accuracy loss.
   - 4-bit NF4 saves the most memory but is slower than bf16, because it dequantises on the fly. It also loses
     1.9 points.
10. **A hand-written Triton kernel ≈ `torch.compile`, but it still pays off in an eager model.**
    - On its own, the fused kernel is 6–8× faster than eager PyTorch and reaches ~884 GB/s, but only 0–9% faster
      than what the compiler generates.
    - Inside the model, RMSNorm + residual are 15.5% of prefill GPU time, because eager RMSNorm upcasts to fp32 and
      makes ~8 passes over memory.
    - Plugged into our from-scratch decoder, the kernel cut prefill by 10% (the profiler bound is ~13%) and sped up
      decode by 12%.
11. **Hard negatives fixed BFCL hallucination but broke the agent: the model now over-refuses.**
    - The 1,000 "needed tool removed → refuse" examples taught the model to refuse whenever no single tool answers the
      question directly. In the document agent, the answer needs list → read → calculate → answer, so sft4 replies
      "I don't have a tool for that" (0/48 on all 3 seeds), even after a nudge.
    - BFCL does not catch this, because in every BFCL call case one tool fits the request directly.
    - The older sft3b is the better agent (23/48 with the guard). It always read the right section, but it did
      arithmetic in its head instead of calling `calculate`.
    - The fix (not done here) is to balance hard negatives with positives where the tools fit only indirectly, i.e.
      multi-step trajectories.
    - No model ever followed the hidden prompt injection ("email this data to …").
12. **Multi-turn needs trajectory data.** Our models act at once instead of first exploring with list or read tools.
    Single-turn chat data does not teach that.
13. **Decode in plain transformers is limited by kernel launches, not by the GPU.**
    - In decode, one token takes 12 ms of GPU work against ~41 ms of wall time, so the GPU is busy only ~30% of the
      time.
    - vLLM, which uses CUDA graphs, reaches 88 tok/s on the same slice. Fusing kernels helps here twice: less memory
      traffic and ~720 fewer launches per token.
14. **A from-scratch decoder can match the library exactly.** If you copy the same operation order (fp32 RMSNorm,
    bf16 RoPE, fp32 softmax, the muP multipliers), the logits are bit-identical. This is a useful test that you
    really understand the architecture.

## Lessons learned (engineering)
- **Change one thing per run.**
  - One run changed the data *and* the batch size at once and went to NaN at step 14, with no way to tell which
    change caused it.
  - Added a NaN guard (skip non-finite gradients, stop after 10).
- **Save every candidate checkpoint.** An early run saved only its final adapter, and its best step was lost. Now
  an adapter is saved at every validation step, plus `best.json`.
- **Store raw model output next to the scores.** Without it you cannot tell "the model is wrong" from "the parser
  is wrong".
- **Read scores from the per-category JSON, not the summary CSV.** A comma in our display name shifted the CSV
  columns.
- **Key log-probabilities by token id, not token text.** Keyed by text, special tokens never matched, so the gate
  saw p = 0 everywhere.
- **Pin library versions per environment.**
  - Training ran on transformers 5.14 and serving on 4.51 (vLLM 0.8.5).
  - transformers 5 writes RoPE settings in a new key, which 4.51 silently misreads as θ = 10,000. So the merge
    script copies the original `config.json`.
- **Shared GPUs (MIG):**
  - vLLM 0.8.5 cannot map a MIG UUID to a device; see `hpc_glue/sitecustomize.py`.
  - Packing 3 trainings into one slice crashed the cuDNN attention kernel; 2 per slice with
    `--no-cudnn-sdp` works.
  - Reap finished child processes on shared login nodes.

---

## Quick start: use the model

**Transformers + PEFT** (4-bit base):
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

repo = "avinash1954/granite-4.1-3b-toolcall-lora"
tok = AutoTokenizer.from_pretrained(repo)                 # Granite 4.2 tokenizer + chat template
base = AutoModelForCausalLM.from_pretrained(
    "ibm-granite/granite-4.1-3b-base", device_map={"": 0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16))
model = PeftModel.from_pretrained(base, repo).eval()

tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Jalandhar right now?"}]
ids = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, enable_thinking=False,
                              return_tensors="pt", return_dict=True).to(0)
out = model.generate(**ids, max_new_tokens=80, do_sample=False)
print(tok.decode(out[0, ids["input_ids"].shape[1]:]))
# <tool_call>\n<function=get_weather>\n<parameter=city>\nJalandhar\n</parameter>\n</function>\n</tool_call>
```
To turn the XML into Python dicts, use `pipeline/granite_xml.py` (`parse_tool_calls(text, tools)`).

**llama.cpp** (OpenAI-compatible server; returns standard `tool_calls`):
```bash
llama-server -m granite-4.1-3b-toolcall-Q4_K_M.gguf -ngl 99 -c 8192 --jinja --chat-template-kwargs '{"enable_thinking":false}'
```

---

## Reproduce

### 1. Environments
Two environments are needed, because vLLM 0.8.5 needs an older transformers:
```bash
# training, data, agent, kernels (Python 3.10+, CUDA GPU)
pip install -r requirements.txt

# BFCL + vLLM serving
pip install -r requirements-bfcl.txt
git clone https://github.com/ShishirPatil/gorilla.git && cd gorilla && git checkout f7cf735
cd berkeley-function-call-leaderboard && pip install -e ".[oss_eval_vllm]" && pip install transformers==4.51.3
```

### 2. Build the data (downloads `NousResearch/hermes-function-calling-v1`, Apache 2.0)
```bash
python pipeline/build_dataset.py       # data/train.jsonl, data/val.jsonl  (v1: cleaned, split by function family)
python pipeline/build_dataset_v2.py    # data/v2/  (+ easy negatives)
python pipeline/build_dataset_v3.py    # data/v3/  (+ 1,000 hard negatives; train 7,726 chats)
```
- The validation split groups conversations by called-function family (union-find), so no function appears in
  both train and validation.

### 3. Train (QLoRA; 1 h 43 min on a 3/7 H100 MIG slice; peak 7.7 GB at length 4096)
```bash
python pipeline/train.py --run sft4 --data data/v3 --steps 460 --batch 1 --accum 16 --max-len 4096 \
  --lr 2e-4 --warmup 30 --rank 16 --seed 0 --eval-every 50 --save-every 50
```
- **Outputs:** `runs/sft4/adapter_step*/` and `best.json` (lowest validation loss). Logs are in
  `train_log.jsonl` and `val_log.jsonl`.
- **Short tests on an 8 GB laptop GPU:** add `--max-len 1024` (peak 4.9 GB).
- **Stop and resume:** stop safely with Ctrl-C or a `STOP` file in the run folder. Re-running the same command
  resumes exactly from `last.pt`.
- **Ablations:** `pipeline/run_parallel.py` runs the seed, rank, 16-bit and data-size variants, two per GPU.

### 4. Merge and evaluate on BFCL
```bash
python pipeline/merge.py --adapter runs/sft4/adapter_step450 --out merged/sft4     # bf16, vLLM-ready
vllm serve merged/sft4 --port 8000 --dtype bfloat16 --max-model-len 32768 &
export LOCAL_SERVER_PORT=8000
cd pipeline
python bfcl_run.py generate --model granite42-xml-FC --test-category non_live,live,multi_turn \
  --local-model-path ../merged/sft4 --skip-server-setup
python bfcl_run.py evaluate --model granite42-xml-FC --test-category non_live,live,multi_turn
```
- `bfcl_run.py` registers our handler (`pipeline/bfcl_granite.py`, model name `granite42-xml-FC`) and then runs
  the normal BFCL command line.
- **Read the scores from the per-category `*_score.json` files.**
- **IBM's reference model:** drop `--local-model-path` and serve `ibm-granite/granite-4.2-3b` instead.
- **On a MIG slice:** start vLLM with `PYTHONPATH=hpc_glue`.
- **Optional gate experiment:** set `GRANITE_GATE_TAU` and tune it with `pipeline/gate_dev.py`.

### 5. Serving benchmarks
```bash
python pipeline/serve_bench.py --model merged/sft4 --tag fp8 --quant fp8 --val data/v3/val.jsonl --out fp8.json
LLAMA_CPP=/path/to/llama.cpp python pipeline/make_gguf.py --model merged/sft4 --out gguf/    # Q8_0 + Q4_K_M
llama-bench -m gguf/sft4-Q4_K_M.gguf -ngl 99 -p 512 -n 128 -r 3
```

### 6. Triton kernel, profiler, decoder from scratch
```bash
python pipeline/triton_rmsnorm.py --out triton.json    # correctness + timing vs eager and torch.compile
python pipeline/profile_model.py --model merged/sft4 --val data/val.jsonl --out profile.json   # GPU-time shares
python pipeline/decoder_scratch.py --model merged/sft4 --val data/val.jsonl --out decoder.json # exactness + speed
```
- `decoder_scratch.py` loads the transformers reference first and frees it before loading its own copy. Two bf16
  copies do not fit a 24 GB slice.

### 7. Document agent
```bash
python agent/make_docs.py        # 12 PDFs + agent/tasks.jsonl (needs reportlab)
python agent/parse_docs.py       # Docling -> agent/parsed/*.json
python agent/run_agent.py --adapter runs/sft4/adapter_step450 --run a1 [--guard]
```

## Repository layout
| Path | What it does |
|---|---|
| `pipeline/hermes.py` | Hermes/Glaive → chat messages; validates every call against its tool schema |
| `pipeline/chat_format.py` | renders with the 4.2 template; loss mask on assistant tokens only |
| `pipeline/build_dataset*.py` | data v1 / v2 / v3 |
| `pipeline/model.py`, `train_data.py`, `train.py` | 4-bit base + LoRA + trainable token rows; own training loop (exact resume, NaN guard, best-checkpoint keeping) |
| `pipeline/run_parallel.py` | several small trainings on one GPU |
| `pipeline/merge.py` | merge an adapter into bf16 weights for vLLM |
| `pipeline/granite_xml.py` | parser for `<tool_call><function=…><parameter=…>` |
| `pipeline/bfcl_granite.py`, `bfcl_run.py` | BFCL handler for the Granite 4.2 format |
| `pipeline/gate_dev.py` | call-or-answer threshold study |
| `pipeline/serve_bench.py`, `make_gguf.py`, `laptop_bench.py` | serving benchmarks and GGUF export |
| `pipeline/triton_rmsnorm.py` | Triton kernel + benchmark |
| `pipeline/profile_model.py` | torch.profiler: GPU-time share of each block in the model |
| `pipeline/decoder_scratch.py` | Granite decoder in plain PyTorch (RoPE, GQA, SwiGLU, KV cache), checked against transformers |
| `pipeline/card_check.py` | runs the model-card snippet end to end |
| `agent/` | synthetic PDFs, Docling parsing, tool-using agent loop |
| `hpc_glue/sitecustomize.py` | vLLM 0.8.5 fix for MIG devices |
| `scripts/` | small inspection scripts (tokenizer, dataset, template) |

## Limitations
- **Offline BFCL only.** The web-search and memory categories were not run, so these are not official leaderboard
  scores.
- **Non-thinking baseline.** granite-4.2-3b was evaluated in non-thinking mode; its default thinking mode may
  score higher.
- **Over-refusal in agents.** The final model (sft4) refuses multi-step document tasks (0/48); use sft3b-style data
  or add multi-step trajectories for agent use.
- **Narrow skill.** The adapter targets tool calling. General chat and reasoning were not measured and are likely
  below IBM's post-trained models.
- **One data family.** The training data is synthetic, from Hermes and Glaive, and English only.
- **Shared slices.** Speed numbers are from a MIG slice (1/7 of an H100) and one laptop GPU. Full-H100 numbers will
  be higher.

## Licence and attribution
- **Base model:** `ibm-granite/granite-4.1-3b-base` (Apache 2.0).
- **Tokenizer and chat template:** `ibm-granite/granite-4.2-3b` (Apache 2.0).
- **Data:** NousResearch/hermes-function-calling-v1 (Apache 2.0).
- **Evaluation:** BFCL from ShishirPatil/gorilla (Apache 2.0).
- **Libraries:** Docling (MIT), llama.cpp (MIT).

## Acknowledgement
The authors gratefully acknowledge the High Performance Computing (HPC) facility provided by Dr. B. R. Ambedkar
National Institute of Technology Jalandhar (NIT Jalandhar) for supporting the computational requirements of this
research work.

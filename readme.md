# ViReL — Vision Reinforcement Learning

**Curing spatial blindness in Vision-Language Models.**
 
ViReL is an ongoing project exploring how far reinforcement learning can push VLMs toward *grounded* spatial reasoning (e.g. "is the mug to the left of the laptop?", "which point is closer to the camera?"). Modern VLMs answer these questions fluently but often incorrectly, because next-token cross-entropy loss rewards fluent-sounding answers, not physically correct ones. ViReL's goal is to close that gap using policies such as **GRPO** (Group Relative Policy Optimization) and, eventually, **GSPO**.
 
> For the full technical write-up (problem framing, GRPO vs PPO derivation, failure-mode examples from our own training runs), see [`blog.md`](./blog.md). *In Progress currently in `Tree/sunit`*

### SFT on a VLM (Flower Classification)
[`Tasks/vlm-sft-flowerclassification/`](./Tasks/vlm-sft-flowerclassification/)

Before touching spatial reasoning, this validates the supervised fine-tuning pipeline itself: fine-tuning **SmolVLM2** to classify flower species and emit structured JSON, comparing two versions(`res384` vs `res768`).
 
- **Key finding:** token-level metrics looked excellent (mean token accuracy ≈0.99, eval loss ≈0.05), but true task accuracy on a held-out test split was only ~42%. This is the motivating result for moving to RL objectives that score actual output correctness rather than next-token likelihood.
- Includes training/eval scripts, a data formatter, an inference script, and evaluation results (`Inference/flower_evaluation_results.xlsx`, screenshots).

Before Reproducing runs ensure to run these
```python
git clone https://github.com/ssdunit/ViReL
cd ViReL/Tasks/vlm-sft-flowerclassification
python -m venv .venv
source .venv/bin/activate
```
Reproducibility

version 1 (`res384`)
```python
python train.py \
  --model_id "HuggingFaceTB/SmolVLM2-2.2B-Instruct" \
  --epochs 2 \
  --learning_rate 2e-4 \
  --train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

version 2 (`res768`)
```python
python train.py \
  --model_id "HuggingFaceTB/SmolVLM2-2.2B-Instruct" \
  --epochs 2 \
  --learning_rate 3e-5 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1
```

### GRPO on an SLM for GSM8K
`Tasks/slm-grpo-gsm8k/`
 
Implements Group Relative Policy Optimization (via Unsloth + TRL's `GRPOTrainer`) on a small language model (`Qwen2.5-0.5B-Instruct` by default) against grade-school math (GSM8K), to build intuition for GRPO and its reward design before scaling to a VLM.
 
- LoRA-based fine-tuning, multiple custom reward functions (correctness, strict format, XML formatting, cosine-scaled, repetition penalty, reasoning length, overgeneration penalty — see `rewards.py`).
- Includes a standalone `infer.py` that mirrors training-time prompt formatting and answer extraction so eval numbers are computed consistently with training rewards.

Reproducibility

```python
git clone ViReL
cd slm-grpo
python -m venv .venv
pip install -r requirements.txt
python train.py
```

Inferenced model was trained on parameters mentioned in the [`config.py`](./Tasks/slm-grpo-gsm8k/config.py) to reproduce the same model just use the above code.

To Experiment with the configuration please view the lists of configs within the files itself.

## VLM-GRPO — Spatial Reasoning via Reinforcement Learning

### Overview

Standard supervised fine-tuning rewards fluent-sounding text, not physically
correct answers, a VLM can achieve near-perfect token-level accuracy while
still failing simple spatial questions ("is the mug left of the laptop?").
This project fine-tunes a Vision-Language Model with **GRPO** (Group Relative
Policy Optimization) directly against answer correctness, using a
robot-manipulation VQA dataset (Robo2VLM) as the spatial-reasoning testbed.

Unlike PPO, GRPO drops the learned critic entirely: for each prompt it samples
a *group* of completions and computes advantage by normalizing each reward
against that group's own mean/std. This roughly halves the VRAM footprint of
PPO (no critic model of the same size as the policy) at the cost of needing
several completions per prompt per step.

## Project structure — `grpo_train/`
```
grpo_train/
├── config.py              # All hyperparameters, model/data/training/GRPO/logging settings
├── data_util.py           # Dataset loading + Qwen3-VL prompt formatting
├── reward.py              # GRPO reward functions (format + answer correctness)
├── preflight.py           # Pre-training sanity checks (run once before train.py)
├── train.py               # Main GRPO training entrypoint
├── evaluation.py          # Post-training evaluation CLI (single question / RoboVista batch)
└── linux_inference.sh     # Runs evaluation.py once per RoboVista domain
```

### File descriptions

- **`config.py`** - Single source of truth for every setting: base model, dataset id, LoRA rank/alpha/dropout, batch size, learning rate, GRPO generation count/temperature/KL beta, and W&B logging names.
- **`data_util.py`** - Loads the Robo2VLM-style dataset, parses multiple-choice options into `A)/B)/C)...` form, builds the Qwen3-VL chat prompt with the `<think>/<answer>` format instructions, and caps image resolution.
- **`reward.py`** - Scores each completion: 1.0 for matching the required `<think>/<answer>` format, 3.0 for a correct answer letter, 0 otherwise.
- **`preflight.py`** - Run once before training to catch dataset/model issues early: missing image tokens, prompts truncated past their image content, or an EOS/chat-template mismatch that would stop generation from terminating correctly.
- **`train.py`** - Loads the base model in 4-bit via Unsloth, applies LoRA, builds the GRPO trainer with the two reward functions, trains, then merges and saves the final model.
- **`evaluation.py`** - Evaluates a trained checkpoint: `single` mode for one image/question, `robovista` mode for a full benchmark run with an Excel report.
- **`linux_inference.sh`** - Loops `evaluation.py --mode robovista` over every domain in the RoboVista dataset.

### Pipeline

1. **Data preparation** (`local_robo2vlm.py`) - sample, sanity-check, and
   snapshot a fixed train/test split from Robo2VLM-1.
2. **Preprocessing** (`data_util.py`) - parse multiple-choice options, build
   Qwen3-VL chat prompts with the required `<think>/<answer>` output format,
   cap image resolution.
3. **Pre-flight checks** (`preflight.py`) - verify the dataset/model pairing
   won't silently lose image tokens to truncation or fail to stop generation.
4. **Training** (`train.py`, `config.py`, `reward.py`) - LoRA + GRPO
   fine-tuning of Qwen3-VL-8B-Thinking via Unsloth, rewarded on output format
   and answer correctness.
5. **Evaluation** (`evaluation.py`, `linux_inference.sh`) - single-question
   inspection or full batch evaluation against RoboVista, with per-domain
   breakdowns and an Excel report.

### Reward design

| Reward | Condition | Value |
|---|---|---|
| `format_reward` | Completion matches `</think><answer>X</answer>` | 1.0 / 0.0 |
| `answer_correctness_reward` | Extracted answer letter matches ground truth | 3.0 / 0.0 |

## What We Tried — Challenges & Fixes

**1. SFT-then-GRPO cold start caused reward hacking.**
Warm-starting with SFT before GRPO led the model to game the format reward — emitting empty/garbage `<think>` content just to collect the reward, with no real reasoning behind it. Fixed by relaxing the format reward to match only `</think><answer>X</answer>` (`BASE_FORMAT_PATTERN` in `reward.py`), removing the incentive to fake the opening tag.

**2. VRAM spikes from image tokens on a single RTX 4090.**
Full-resolution images caused large VRAM spikes during GRPO, worsened by needing multiple completions in memory per prompt (`NUM_GENERATIONS = 8`) nearly `50000` tokens per rollout. Fixed by capping every image to ~1MP and resizing with Lanczos resampling (`cap_image()` in `data_util.py`), keeping detail while making per-step memory usage sustainable on a single 4090.

## Usage
First
```
git clone https://github.com/ssdunit/ViReL
cd ViReL
python -m venv .venv
pip install -r requirements.txt
```
Before Training ensure to prepare a shard or chunk of dataset for training this process can be skipped if you want to stream the whole dataset or even download the whole dataset. Skip this step to use the custom dataset.

```bash
#Prepare a local dataset snapshot
python local_robo2vlm.py

#Sanity-check the dataset/model
python preflight.py --model unsloth/Qwen3-VL-8B-Thinking --dataset <dataset-id> --max-prompt-length 1536
```
Training: 
```bash
#Train
python train.py
```

## Inference

1. For Single Question Evaluation
```bash
python evaluation.py --mode single --model_path outputs_grpo --image path/to/img.jpg \
    --question "Which object is closer to the camera?" --choices "mug" "laptop" "plate"
```
2. For single domain Evaluation 

```bash
python evaluation.py --mode robovista --greedy --repetition_penalty 0.5 --domain "open_datasets"
#Replace "open_datasets" with whichever domain you want
```
3. For all domain Evaluation (Within the script you can also put domains in the `skip domain` section to skip any domains you need)
```bash
bash linux_inference.sh
```

## Outputs

*(To be filled in — trained model location, checkpoint sizes, evaluation results, and accuracy figures will be documented here once training and eval completes.)*

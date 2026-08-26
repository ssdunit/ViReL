 # <p align="center" style="font-family: Merriweather; font-size: 35px">Curing spatial blindness in VLMs</font></p>

---
> Ask a state-of-the-art Vision-Language Model to write a poem about a sunset, and it will give you a masterpiece. Ask that same model if a robotic gripper is correctly aligned 2 millimeters to the left of a surgical suture, and it will confidently hallucinate a failure. 
---
## List of Contents
* ### [Introduction](#-1-introduction-1-2)
    * ### [What is ViReL?](#what-is-virel)
    * ### [Overview](#overview-1)
* ### [Problem setting](#-2-problem-setting-1)
    * ### [Core issues](#the-core-spatial-problems-we-need-to-fix)

<br>

---

# <font style="font-family:Merriweather"> 1. Introduction [[1](#what-is-virel)] [[2](#overview-1)]</font>
---
<p style="font-size:18px">
Vision-language models (VLMs) have rapidly mastered semantic understanding, yet they continue to suffer from a critical limitation: spatial blindness.While modern VLMs can fluently describe the contents of a scene, they frequently hallucinate when asked to determine the physical relationship between objects—a fundamental roadblock for embodied AI, autonomous driving, and robotic manipulation. To solve this issue and bridge the gap we introduce ViReL(Vision Reinforced Learning)
</p>

## <font style="font-family:Merriweather">What is ViReL?</font>
<p style="font-size:16px">
ViReL (Vision Reinforcement Learning) is our attempt to build a VLM that grounds its reasoning in what it actually sees, using reinforcement learning rather than supervised fine-tuning alone.
Through this project we target to actually work on curing the spatial blindless of current Vision Language Models by finetuning them on different Reinforcement learning Policies -- Our targeted Policies being GRPO (Group Relative Policy Optimization) and GSPO (Group Sequence Policy Optimization) -- to strictly enforce grounded spatial reasoning with a possible integration of spatial Vision Transformer or ViT backbone. 

![General Pipeline for final Model](https://i.ibb.co/qM041p2y/Spatially-Grounded-VLM-Training-Pipeline.png)

*From semantic baseline to spatial reasoning: The full ViReL training architecture.*

### <p style="font-size:20px">Overview</p> 

> <p style="font-size:16px">This blog explores the mechanics behind VLM spatial disconnects, details the training recipes and architectural bets designed to fix them, and unpacks the real-world failure modes encouraged along the way.
---
# <font style="font-family:Merriweather"> 2. Problem Setting [[1](#the-core-spatial-problems-we-need-to-fix)]</font>
<font style="font-size:16px">
Ask a modern vision-language model, "Which object is closer to the camera?" or "Is the mug to the left of the laptop?" and it will answer fluently, confidently, and often wrong. This isn't a knowledge gap—the model has seen millions of images with spatial descriptions in its pretraining corpus. It is a grounding gap: the model has learned to produce the shape of a spatial answer without performing the underlying visual computation that would make the answer reliable.

While training our own VLM-GRPO runs on robot-manipulation VQA data, we caught this failure mode red-handed in the raw completions. Given a prompt asking which of several colored points was closest to the camera, the model's chain-of-thought wasn't reasoning over pixels at all. Instead, it was reasoning over its own uncertainty about whether it even had access to the image:

> "Wait, the user provided an image link, but since I can't view it, maybe I need to think about typical robot vision tasks... Wait, no, maybe the image is described in the context... perhaps this is a hypothetical scenario..."

![Model output 1(Image issue)](https://i.ibb.co/pjFQZpys/Issuewithimagefinding1.png)

Or, in other cases, it would completely hallucinate based on global context rather than physical reality:

> "If the screw lies next to the cup it should move to the right, wait no theres another cup to the left of the screw guessing the distance, right cup is closer so i will move to the right"

![Model output 2(Context issue)](https://i.ibb.co/j9tv2S9p/Issuewithimagefinding.png)
### The core spatial problems we need to fix
* **Hallucinated Grounding**: The model generates fluent spatial reasoning without verifying whether its claims match the physical coordinate space of the image.

* **Sensitivity to Perspective and Resolution**: Moving a camera slightly or changing the input aspect ratio drastically degrades spatial answers, even when the scene itself hasn't changed.

* **Reward Mismatch**: Standard cross-entropy loss only asks, "Did you predict the next word in the dataset?" What spatial reasoning actually requires is, "Is the final physical relationship true in the physical world?"

Our project, ViReL, is built to cure this exact spatial blindness. By systematically reducing these language-driven hallucinations, we are forcing the model to anchor its reasoning to the actual physical coordinate space—converting raw visual input into reliable, grounded, and actionable spatial data.

---
# <font style="font-family:Merriweather"> 3. What have we worked on? </font>

<font style="font-size:16px">
Before jumping on the actual finetuning of VLMs on the said policies we actually finetuned VLMs on Supervised Finetuning for a very simple task of flower detection and Read about GRPO thoroughly through Deepseek research papers with their applied pipelines for their base model (Deepseek-V3-base). To actually implement and understand GRPO we finetuned an Small language model(SLM) on GRPO for grade school maths (**Dataset**: *OpenAI-gsm8k*)
</font>

## <font style="font-family:Merriweather"> 3.1 Proof of Concept 1- SFT on VLMs (Flower Classification) [[1](#training-and-evaluation-metrics)][[2](#inference)]</font>

Before touching spatial reasoning, we validated the supervised fine-tuning (SFT) pipeline itself on a controlled, simple task: fine-tuning SmolVLM2 to classify flower species and emit structured JSON (```{"flower_type": "common dandelion", "confidence": 1.0}```). We ran this test comparing two input resolutions (**res384** vs **res768**).

The initial training metrics looked excellent. Mean token accuracy on the evaluation set converged to **~0.99**, and eval loss dropped to **~0.05** within roughly **1,000 steps** for the higher-resolution run. However, a corrected, held-out test split told a completely different story: actual task accuracy landed at just **42.45% (433/1020)**.

This stark contrast gave a critical lesson: token-level SFT is a poor proxy for task-level correctness. The model easily memorized the JSON format and vocabulary, but failed once the evaluation protocol shifted slightly from the training distribution. It is the exact same class of failure—“the model looks confident but isn't grounded”—that motivated our pivot to RL objectives that score the actual output rather than next-token likelihood.

This task helped us learn how to read and analyze metric charts
### Training and Evaluation metrics
![Evaluation metrics](https://i.ibb.co/vMPYn7H/VLMCHARTS.png) 

*Evaluation metrics(v1 and v2)*

![Training metrics](https://i.ibb.co/pjXHXBqx/VLMCHARTS.png) 

*Training metrics(v1 and v2)*

As we can see in the charts the model's loss was nearly 0 at very early steps, but this the was the best attempt of training the model with a very small dataset

### Inference

![Example](https://i.ibb.co/HfzDMPWx/2.png)

*Example Inference*

![Eval Results](https://i.ibb.co/wFzYNBxJ/Inference.png)

*Evaluation Results*

## <font style="font-family:Merriweather">3.2 Proof of concept 2 — GRPO on an SLM for GSM8K</font>

To de-risk the RL infrastructure independent of the vision modality, we ran Group Relative Policy Optimization (GRPO) — the algorithm introduced in DeepSeekMath — on a small language model against GSM8K, using Unsloth + TRL's GRPOTrainer.

### Why GRPO over PPO?


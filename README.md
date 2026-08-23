# GRPO LLM Fine-Tuning: Improving Mathematical Reasoning with the CountDown Task

> Fine-tuning **LLaMA** and **Qwen2.5** using **Group Relative Policy Optimization (GRPO)** on the CountDown task dataset to enhance step-by-step mathematical reasoning.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Dataset: CountDown Task](#dataset-countdown-task)
- [Models](#models)
- [Training Approach: GRPO](#training-approach-grpo)
- [Reward Functions](#reward-functions)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Training](#running-training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

---

## Overview

This project applies **reinforcement learning from AI feedback (RLAIF)** — specifically the GRPO algorithm — to improve the mathematical reasoning capabilities of two state-of-the-art open-source language models:

| Model | Base Checkpoint |
|-------|----------------|
| LLaMA | `meta-llama/Llama-3.1-8B-Instruct` (or similar) |
| Qwen2.5 | `Qwen/Qwen2.5-7B-Instruct` (or similar) |

The training signal comes from the **CountDown task**, a combinatorial arithmetic puzzle where the model must reach a target number from a given set of numbers using basic arithmetic operations (+, −, ×, ÷). This task demands multi-step planning, precise calculation, and structured chain-of-thought reasoning — making it an ideal benchmark for mathematical reasoning research.

The approach is inspired by [DeepSeek-R1](https://arxiv.org/abs/2501.12948), which demonstrated that GRPO can dramatically improve reasoning without relying on supervised fine-tuning on human-annotated chain-of-thought data.

---

## Key Features

- 🧮 **CountDown task** as a structured mathematical reasoning benchmark
- 🤖 Support for both **LLaMA** and **Qwen2.5** model families
- 🎯 **GRPO** training loop with group-level advantage estimation
- ✅ Rule-based reward functions (format compliance + answer correctness)
- 📊 Logging and evaluation utilities to track reasoning quality
- 🚀 Compatible with **Hugging Face Transformers** and **TRL**

---

## Project Structure

```
GRPO-LLM-Finetuning-/
├── data/                   # Dataset loading and preprocessing scripts
│   └── countdown.py        # CountDown task dataset utilities
├── models/                 # Model configuration and loading helpers
├── training/               # GRPO training scripts
│   ├── train_llama.py      # Training script for LLaMA
│   └── train_qwen.py       # Training script for Qwen2.5
├── evaluation/             # Evaluation and benchmarking scripts
│   └── evaluate.py         # Accuracy and reasoning quality metrics
├── rewards/                # Reward function definitions
│   └── reward_fns.py       # Format and correctness reward functions
├── configs/                # YAML/JSON training configuration files
├── notebooks/              # Exploratory and result analysis notebooks
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Dataset: CountDown Task

The **CountDown task** is a classic arithmetic puzzle:

> Given a list of numbers (e.g., `[3, 4, 6, 25]`) and a target (e.g., `952`), find an expression using each number at most once and the four basic arithmetic operations that equals the target.

**Example:**
```
Numbers : [3, 4, 6, 25]
Target  : 952
Solution: (6 + 4) × (25 × 3 + 2) — step-by-step working shown in the model's chain-of-thought
```

The dataset is constructed programmatically by:
1. Sampling random sets of numbers.
2. Computing a reachable target via a random sequence of operations.
3. Recording both the numbers, the target, and a reference solution.

This provides a practically unlimited supply of (input, verified solution) pairs, enabling scalable GRPO training with deterministic correctness checking.

---

## Models

### LLaMA (Meta AI)
LLaMA 3.1 / 3.2 instruction-tuned variants serve as strong baselines. These models have broad world knowledge and instruction-following capability, but benefit significantly from RL-based reasoning fine-tuning on structured tasks.

### Qwen2.5 (Alibaba Cloud)
Qwen2.5 models (7B / 14B) are competitive open-source models with strong multilingual and mathematical pre-training. GRPO fine-tuning further specialises their reasoning toward step-by-step arithmetic.

---

## Training Approach: GRPO

**Group Relative Policy Optimization (GRPO)** is a reinforcement learning algorithm designed for LLMs that avoids the need for a separate critic/value model. For each prompt, GRPO:

1. Samples a **group** of *G* responses from the current policy.
2. Scores each response using a reward function.
3. Computes **group-relative advantages**: each response's advantage is its reward minus the group mean, normalised by the group standard deviation.
4. Updates the policy via a clipped surrogate objective (similar to PPO) using these advantages.

This approach is memory-efficient and stable, making it practical for fine-tuning 7B–13B parameter models on a single node.

```
Prompt (numbers + target)
        │
        ▼
   Sample G responses
        │
        ▼
  Score with reward fn
        │
        ▼
  Compute group advantages
        │
        ▼
  Policy gradient update
```

---

## Reward Functions

Two complementary reward signals are used:

| Reward | Description | Range |
|--------|-------------|-------|
| **Format reward** | Checks that the model's response contains a valid `<think>…</think>` reasoning block followed by a final `<answer>…</answer>` tag. | 0 or 1 |
| **Correctness reward** | Evaluates whether the expression in `<answer>` actually equals the target number when computed. | 0 or 1 |

The final reward is the sum of both components, incentivising the model to produce both well-structured reasoning traces and correct solutions.

---

## Getting Started

### Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU(s) with ≥ 24 GB VRAM (e.g. A100, H100, or multiple smaller GPUs)
- [PyTorch](https://pytorch.org/) ≥ 2.2
- [Hugging Face Transformers](https://github.com/huggingface/transformers) ≥ 4.40
- [TRL](https://github.com/huggingface/trl) ≥ 0.9 (for GRPO trainer)

### Installation

```bash
# Clone the repository
git clone https://github.com/Quancoder260503/GRPO-LLM-Finetuning-.git
cd GRPO-LLM-Finetuning-

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

You will also need to authenticate with the Hugging Face Hub to download gated model weights:

```bash
huggingface-cli login
```

### Running Training

**LLaMA:**
```bash
python training/train_llama.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_size 50000 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --grpo_group_size 8 \
    --output_dir outputs/llama-grpo-countdown
```

**Qwen2.5:**
```bash
python training/train_qwen.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset_size 50000 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --grpo_group_size 8 \
    --output_dir outputs/qwen25-grpo-countdown
```

Key training arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | — | Hugging Face model ID or local path |
| `--dataset_size` | `50000` | Number of CountDown puzzles to generate |
| `--grpo_group_size` | `8` | Number of responses sampled per prompt (G) |
| `--max_new_tokens` | `512` | Maximum tokens per generated response |
| `--learning_rate` | `1e-6` | Learning rate for policy gradient updates |
| `--output_dir` | `outputs/` | Directory to save checkpoints and logs |

### Evaluation

Run the evaluation script to measure solve-rate on a held-out test set:

```bash
python evaluation/evaluate.py \
    --model_path outputs/llama-grpo-countdown \
    --num_test_puzzles 1000
```

The script reports:
- **Solve rate** — percentage of puzzles solved correctly.
- **Format compliance** — percentage of responses that follow the required output structure.
- **Average reasoning steps** — mean number of arithmetic steps in the chain-of-thought.

---

## Results

> Results will be updated as experiments complete.

| Model | Base Solve Rate | After GRPO | Δ |
|-------|----------------|------------|---|
| LLaMA-3.1-8B-Instruct | — | — | — |
| Qwen2.5-7B-Instruct | — | — | — |

---

## References

- Shao et al. (2024). [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300)
- DeepSeek-AI (2025). [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/abs/2501.12948)
- Ziegler et al. (2019). [*Fine-Tuning Language Models from Human Preferences*](https://arxiv.org/abs/1909.08593)
- [TRL — Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

---

## License

This project is released under the [MIT License](LICENSE).

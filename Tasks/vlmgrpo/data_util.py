"""

Data utility for Robo2VLM.

Responsibilities
----------------
1. Load Robo2VLM dataset
2. Parse multiple-choice answers
3. Build Qwen3-VL prompts
4. Preprocess examples
5. Return HuggingFace Dataset

No training code.
No reward functions.
No GRPO logic.
"""

import ast
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset


# ==========================================================
# Dataset Constants
# ==========================================================

DATASET_NAME = "keplerccc/Robo2VLM-1"#Robo2vlm provides multiple option to choose answer we are providing option in form of A,B,C,D

LETTERS = ["A", "B", "C", "D", "E", "F"]


# ==========================================================
# Parsing Helpers
# ==========================================================

def parse_choices(raw_choices: Any) -> List[str]:#the whole string into python list
    """
    Parse Robo2VLM's stringified choices list.

    Example:
        "['Red','Blue','Green']"

    becomes

        ["Red","Blue","Green"]
    """

    if isinstance(raw_choices, list):
        return [str(x) for x in raw_choices]

    if isinstance(raw_choices, str):
        try:
            parsed = ast.literal_eval(raw_choices)

            if isinstance(parsed, list):
                return [str(x) for x in parsed]

        except Exception:
            pass

    raise ValueError(f"Cannot parse choices: {raw_choices}")


def build_choice_block(choices: List[str]) -> str:
    """
    Convert choices into

    A) Red
    B) Blue
    C) Green
    """

    return "\n".join(
        f"{LETTERS[i]}) {choice}"
        for i, choice in enumerate(choices)
    )


# ==========================================================
# Prompt Builder
# ==========================================================

def build_prompt(
    example: Dict[str, Any],
    system_prompt: str,
):
    """
    Build a Qwen3-VL chat prompt.
    """

    choices = parse_choices(example["choices"])

    question = example["question"].strip()

    choice_block = build_choice_block(choices)

    user_text = (
        f"{question}\n\n"
        f"Choices:\n"
        f"{choice_block}"
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": example["image"],
                },
                {
                    "type": "text",
                    "text": user_text,
                },
            ],
        },
    ]


# ==========================================================
# Dataset Preprocessing
# ==========================================================

def preprocess_example(
    example: Dict[str, Any],
    system_prompt: str,
):
    """
    Convert one raw Robo2VLM example into
    training-ready format.
    """

    choices = parse_choices(example["choices"])

    answer_idx = int(example["correct_answer"])

    return {
        "prompt": build_prompt(
            example,
            system_prompt,
        ),
        "answer_idx": answer_idx,
        "answer_letter": LETTERS[answer_idx],
        "num_choices": len(choices),
        "task_id": example["id"],
    }


# ==========================================================
# Dataset Loader
# ==========================================================

def load_robo2vlm(
    dataset_name:str,
    system_prompt: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """
    Load and preprocess Robo2VLM.
    """

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    if max_samples is not None:
        dataset = (
            dataset
            .shuffle(seed=seed)
            .select(range(min(max_samples, len(dataset))))
        )

    dataset = dataset.map(
        lambda x: preprocess_example(
            x,
            system_prompt,
        )
    )

    return dataset


# ==========================================================
# DatasetDict Loader
# ==========================================================

def load_robo2vlm_splits(
    system_prompt: str,
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
):
    """
    Return train and test datasets.
    """

    train = load_robo2vlm(
        system_prompt=system_prompt,
        split="train",
        max_samples=train_samples,
    )

    test = load_robo2vlm(
        system_prompt=system_prompt,
        split="test",
        max_samples=test_samples,
    )

    return DatasetDict(
        {
            "train": train,
            "test": test,
        }
    )
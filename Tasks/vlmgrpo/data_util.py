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

from datasets import Dataset
from datasets import load_dataset

# ==========================================================
# Dataset Constants
# ==========================================================

DATASET_NAME = "Rauneet/robo2vlm-erqa-plus-combined"#Robo2vlm provides multiple option to choose answer we are providing option in form of A,B,C,D

LETTERS = ["A", "B", "C", "D", "E", "F"]

from PIL import Image

def resize_image(example, size=(320, 256)):
    example["image"] = example["image"].resize(size)
    return example

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
    tokenizer: Any,
):
    """
    Build a Qwen3-VL chat prompt.
    """

    choices = parse_choices(example["choices"])

    question = example["question"].strip()
    choice_block = build_choice_block(choices)
    """
    user_text = (
        f"{question}\n\n"
        f"Choices:\n"
        f"{choice_block}"
    )
    """
    user_text = (
        f"{question}\n\n"
        f"Choices:\n{choice_block}\n\n"
        f"FORMAT REQUIREMENT:\n"
        f"Your response MUST begin with the literal tag <think> on the very first line.\n"
        f"Example format:\n"
        f"<think>\n[Your visual reasoning here]\n</think>\n"
        f"<answer>\n[Option Letter]\n</answer>"
    )
    image = example["image"].convert("RGB")

    messages = [
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
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": user_text,
                },
            ],
        },
    ]
    return messages
from PIL import Image

def aggressive_image_filter(example):
    
    if example.get("is_placeholder", False):
        return False
        
    try:
        img = example["image"]
        
        if isinstance(img, Image.Image):
            img.verify()  # Validates the image header without fully loading it into RAM
            return True
            
        elif isinstance(img, str):
            with Image.open(img) as i:
                i.verify()
            return True
            
        return False
        
    except Exception as e:
        # If it throws an UnidentifiedImageError, OSError, or EOFError, it's garbage. Drop it.
        return False

# ==========================================================
# Dataset Preprocessing
# ==========================================================

def preprocess_example(
    example: Dict[str, Any],
    system_prompt: str,
    tokenizer: Any,
):
    """
    Convert one raw Robo2VLM example into
    training-ready format.
    """
    #example["image"] = example["image"].resize((320, 256))
    choices = parse_choices(example["choices"])
    answer_idx = int(example["answer_idx"])
    image = example["image"].convert("RGB")

    return {
        "prompt": build_prompt(
            example,
            system_prompt,
            tokenizer,
        ),
        "image": image,
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
    tokenizer: Any,
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """
    Load and preprocess Robo2VLM.
   """
    if split not in ("train","test"):
        raise ValueError(f"Unknown split: {split}")
    dataset = load_dataset(dataset_name, split=split)

    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))
    dataset = dataset.map(
        lambda x: preprocess_example(
      	x,
        system_prompt,
        tokenizer,
       	),
        num_proc=8,
        writer_batch_size=100,
        load_from_cache_file=True,
    )
    keep_cols = {"prompt", "image","answer_letter"}
    drop_cols = [c for c in dataset.column_names if c not in keep_cols]
    dataset = dataset.remove_columns(drop_cols)
    return dataset


# ==========================================================
# DatasetDict Loader
# ==========================================================

def load_robo2vlm_splits(
    system_prompt: str,
    tokenizer:Any,
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
):
    """
    Return train and test datasets.
    """


    train = load_robo2vlm(
        dataset_name=DATASET_NAME,
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        split="train",
        max_samples=train_samples,
    )

    test = load_robo2vlm(
        dataset_name=DATASET_NAME,
        system_prompt=system_prompt,
        split="test",
        max_samples=test_samples,
        tokenizer=tokenizer,
    )

    return DatasetDict(
        {
            "train": train,
            "test": test,
        }
    )
import os
'''def load_spatialladder_sft(system_prompt, tokenizer, split="train", test_size=0.01, seed=42):
    SPATIALLADDER_IMAGES_ROOT = "/home/ju"
    MAX_IMAGES_PER_EXAMPLE = 4
    dataset = load_dataset("hongxingli/SpatialLadder-26k",name="spatial")["train"]
    dataset = dataset.train_test_split(test_size=test_size,seed=seed)[split]
    def resolve_path(p):
        return os.path.join(SPATIALLADDER_IMAGES_ROOT, p)
    def to_conversation(example):
        img = example["image"]
        images = img if isinstance(img, list) else [img]
        if len(images) > MAX_IMAGES_PER_EXAMPLE:
            images = images[:MAX_IMAGES_PER_EXAMPLE]
        images = [resolve_path(im) if isinstance(im, str) else im for im in images]
        answer_text = (
            f"<think>\n"
            f"{example.get('reasoning', 'Based on the visual evidence in the image(s).')}\n"
            f"</think>\n"
            f"<answer>\n{example['answer']}\n</answer>"
        )
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": example["question"]})
        return {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
            ]
        }
    return dataset.map(to_conversation, remove_columns=dataset.column_names)
'''

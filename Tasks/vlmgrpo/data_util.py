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

from datasets import Dataset, DatasetDict, load_from_disk


# ==========================================================
# Dataset Constants
# ==========================================================

DATASET_NAME = "keplerccc/Robo2VLM-1"#Robo2vlm provides multiple option to choose answer we are providing option in form of A,B,C,D

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
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": user_text,
                },
            ],
        },
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return prompt_str
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

    answer_idx = int(example["correct_answer"])

    return {
        "prompt": build_prompt(
            example,
            system_prompt,
            tokenizer,
        ),
        "images": [example["image"]],
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
    if split=="train":
        dataset= load_from_disk("./datasets/ROBO2VLM_train_10k")
    elif split=="test":
        dataset=load_from_disk("./datasets/ROBO2VLM_test_2k")
    else:
    	raise ValueError(f"Unkown split: {split}")
    dataset = dataset.map(
        lambda x: preprocess_example(
      	x,
        system_prompt,
        tokenizer,
       	),
	num_proc = 16,
        load_from_cache_file=False,
    )
    keep_cols = {"prompt", "images","answer_letter"}
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
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        split="train",
        max_samples=train_samples,
    )

    test = load_robo2vlm(
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

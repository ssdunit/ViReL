"""
data_format.py

Loads the Oxford Flowers dataset and formats it into the chat-style
message structure expected for VLM supervised fine-tuning: each example
pairs an image + instruction prompt with a target JSON response naming
the flower species.

Also derives a label map (numeric class id -> flower name) directly from
the dataset's ClassLabel feature, so both training and evaluation read
labels from the same single source of truth.
"""

import json
from datasets import load_dataset

DEFAULT_DATASET = "dpdl-benchmark/oxford_flowers102"
DEFAULT_PROMPT = "Identify the flower species in this image and output structured JSON."


def dataset_format(dataset: str = DEFAULT_DATASET, prompt: str = DEFAULT_PROMPT, seed: int = 42):
    """
    Load + format the flower dataset for SFT.

    dataset, prompt, and seed are explicit arguments (rather than pulled from
    a global Config object) so a training/eval run is fully determined by
    the CLI flags it was invoked with.

    Returns (train_formatted, eval_formatted, test_dataset, label_map) where
    label_map is a {str(label_id): flower_name} dict derived straight from
    the dataset's ClassLabel feature. Deriving it here (instead of reading a
    separately-saved label_map.json) means there's a single source of truth
    and nothing to go stale between train and eval runs.
    """
    train_dataset = load_dataset(dataset, split="test").shuffle(seed)
    eval_dataset = load_dataset(dataset, split="validation").shuffle(seed)
    test_dataset = load_dataset(dataset, split="train").shuffle(seed)

    label_feature = train_dataset.features["label"]
    label_map = {str(i): label_feature.int2str(i) for i in range(label_feature.num_classes)}

    def format_to_message(example):
        flower_name = label_feature.int2str(example["label"])
        target_dict = {"flower_type": flower_name, "confidence": 1.0}
        target_output_text = json.dumps(target_dict)

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_output_text}],
                },
            ]
        }

    train_formatted_datasets = train_dataset.map(format_to_message, remove_columns=["label"])
    eval_formatted_datasets = eval_dataset.map(format_to_message, remove_columns=["label"])
    return train_formatted_datasets, eval_formatted_datasets, test_dataset, label_map
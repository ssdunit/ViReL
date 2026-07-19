import datasets
from datasets import load_dataset
import os
# import huggingface_hub
import json

# huggingface_hub.login(userdata.get("HF_TOKEN"))

def dataset_format():
    train_dataset = load_dataset("dpdl-benchmark/oxford_flowers102", split="test").shuffle(seed=42)
    eval_dataset  = load_dataset("dpdl-benchmark/oxford_flowers102", split="validation").shuffle(seed=42)
    test_dataset = load_dataset("dpdl-benchmark/oxford_flowers102", split="train").shuffle(seed=42)
    label_feature = train_dataset.features["label"]
    PROMPT = "Identify the flower species in this image and output structured JSON."
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
                        {"type": "text",
                        "text": PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_output_text}],
                },
            ]
        }

    train_formatted_datasets = train_dataset.map(format_to_message, remove_columns=["label"])
    eval_formatted_datasets  = eval_dataset.map(format_to_message, remove_columns=["label"])
    return train_formatted_datasets,eval_formatted_datasets,test_dataset
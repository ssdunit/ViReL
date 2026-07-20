import datasets
from datasets import load_dataset
import os
# import huggingface_hub
import json

# huggingface_hub.login(userdata.get("HF_TOKEN"))
from Config import Config
config = Config()

def dataset_format():
    train_dataset = load_dataset(config.data.dataset, split="test").shuffle(config.data.seed)
    eval_dataset  = load_dataset(config.data.dataset, split="validation").shuffle(config.data.seed)
    test_dataset = load_dataset(config.data.dataset, split="train").shuffle(config.data.seed)
    label_feature = train_dataset.features["label"]
    PROMPT = config.data.prompt
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
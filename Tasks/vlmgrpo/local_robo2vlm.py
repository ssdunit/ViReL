import os
import random

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import HfApi, hf_hub_download

DATASET_REPO = "keplerccc/Robo2VLM-1"

NUM_SHARDS = 6          # Number of parquet files to download
TARGET_SAMPLES = 10000
SEED = 42

SAVE_DIR = "/home/rauneet/datasets/robo2vlm_train_10k"

os.makedirs(os.path.dirname(SAVE_DIR), exist_ok=True)
api = HfApi()

files = api.list_repo_files(
    repo_id=DATASET_REPO,
    repo_type="dataset",
)

parquet_files = sorted(
    f for f in files
    if f.startswith("data/train-") and f.endswith(".parquet")
)

print(f"Found {len(parquet_files)} training shards.")
random.seed(SEED)

selected = random.sample(parquet_files, NUM_SHARDS)

print("\nSelected shards:")
for s in selected:
    print(" ", s)
local_files = []

for shard in selected:

    print(f"\nDownloading {shard}")

    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=shard,
    )

    local_files.append(local_path)
datasets = []

for file in local_files:

    ds = load_dataset(
        "parquet",
        data_files=file,
        split="train",
    )

    datasets.append(ds)

dataset = concatenate_datasets(datasets)

print(f"\nExamples downloaded: {len(dataset)}")-
dataset = dataset.shuffle(seed=SEED)

if len(dataset) > TARGET_SAMPLES:
    dataset = dataset.select(range(TARGET_SAMPLES))

print(f"Final dataset size: {len(dataset)}")
dataset.save_to_disk(SAVE_DIR)

print(f"\nSaved to {SAVE_DIR}")
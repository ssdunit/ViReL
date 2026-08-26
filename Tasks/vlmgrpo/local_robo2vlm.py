import os
import random

from datasets import load_dataset, concatenate_datasets, Image
from huggingface_hub import HfApi, hf_hub_download

DATASET_REPO = "keplerccc/Robo2VLM-1"

NUM_SHARDS = 10
TRAIN_SAMPLES = 10000
TEST_SAMPLES = 2000
SEED = 42

TRAIN_SAVE_DIR = "datasets/robo2vlm_train_10k"
TEST_SAVE_DIR = "datasets/robo2vlm_test_2k"

# --------------------------------------------------
# Create output directories
# --------------------------------------------------

os.makedirs(TRAIN_SAVE_DIR, exist_ok=True)
os.makedirs(TEST_SAVE_DIR, exist_ok=True)

api = HfApi()

# --------------------------------------------------
# Find training parquet shards
# --------------------------------------------------

files = api.list_repo_files(
    repo_id=DATASET_REPO,
    repo_type="dataset",
)

parquet_files = sorted(
    f
    for f in files
    if f.startswith("data/train-") and f.endswith(".parquet")
)

print(f"Found {len(parquet_files)} training shards.")

# Need at least 12,000 samples total.
# Select enough shards to have more than 12,000 examples.
random.seed(SEED)

selected = random.sample(
    parquet_files,
    min(NUM_SHARDS, len(parquet_files))
)

print("\nSelected shards:")
for s in selected:
    print(" ", s)

# --------------------------------------------------
# Download selected shards
# --------------------------------------------------

local_files = []

for shard in selected:

    print(f"\nDownloading {shard}")

    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=shard,
    )

    local_files.append(local_path)

# --------------------------------------------------
# Load shards
# --------------------------------------------------

datasets = []

for file in local_files:

    ds = load_dataset(
        "parquet",
        data_files=file,
        split="train",
    )

    ds = ds.cast_column(
        "image",
        Image(decode=True)
    )

    datasets.append(ds)

# --------------------------------------------------
# Combine all shards
# --------------------------------------------------

dataset = concatenate_datasets(datasets)

print(f"\nTotal examples downloaded: {len(dataset)}")

# --------------------------------------------------
# Shuffle ONCE
# --------------------------------------------------

dataset = dataset.shuffle(seed=SEED)

TOTAL_REQUIRED = TRAIN_SAMPLES + TEST_SAMPLES

if len(dataset) < TOTAL_REQUIRED:
    raise ValueError(
        f"Not enough samples. "
        f"Need {TOTAL_REQUIRED}, but only have {len(dataset)}."
    )

# --------------------------------------------------
# Select exactly 12,000 unique samples
# --------------------------------------------------

dataset = dataset.select(
    range(TOTAL_REQUIRED)
)

# --------------------------------------------------
# Split into train and test
# --------------------------------------------------

train_dataset = dataset.select(
    range(TRAIN_SAMPLES)
)

test_dataset = dataset.select(
    range(TRAIN_SAMPLES, TOTAL_REQUIRED)
)

print(f"\nTraining samples: {len(train_dataset)}")
print(f"Testing samples:  {len(test_dataset)}")

# --------------------------------------------------
# Image sanity check
# --------------------------------------------------

valid_indices = []
broken_indices = []

for i in range(len(dataset)):

    try:

        # This forces the image to be decoded
        img = dataset[i]["image"]

        # Check that image exists
        if img is None:
            raise ValueError("Image is None")

        # Check dimensions
        if img.width <= 0 or img.height <= 0:
            raise ValueError(
                f"Invalid image size: {img.size}"
            )

        # Force actual image conversion
        # This catches some decoding problems
        img.convert("RGB")

        valid_indices.append(i)

    except Exception as e:

        broken_indices.append(i)

        print(
            f"[BROKEN] index={i} | error={repr(e)}"
        )

print("\n====================================")
print("Image check complete")
print("====================================")

print(f"Total samples : {len(dataset)}")
print(f"Valid images  : {len(valid_indices)}")
print(f"Broken images : {len(broken_indices)}")

# --------------------------------------------------
# Make sure enough valid samples exist
# --------------------------------------------------

TOTAL_REQUIRED = TRAIN_SAMPLES + TEST_SAMPLES

if len(valid_indices) < TOTAL_REQUIRED:

    raise ValueError(
        f"\nNot enough valid samples!\n"
        f"Required : {TOTAL_REQUIRED}\n"
        f"Valid    : {len(valid_indices)}\n"
        f"Broken   : {len(broken_indices)}"
    )

# --------------------------------------------------
# Remove broken samples
# --------------------------------------------------

print("\nRemoving broken images...")

dataset = dataset.select(valid_indices)

print(
    f"Dataset after removing broken images: "
    f"{len(dataset)}"
)

# --------------------------------------------------
# Shuffle valid dataset
# --------------------------------------------------

print("\nShuffling valid dataset...")

dataset = dataset.shuffle(seed=SEED)

def final_check(dataset, name):

    print(f"\nFinal check: {name}")

    for i in range(len(dataset)):

        try:

            img = dataset[i]["image"]

            if img is None:
                raise ValueError("Image is None")

            if img.width <= 0 or img.height <= 0:
                raise ValueError(
                    f"Invalid dimensions: {img.size}"
                )

            img.convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"Broken image found in {name} "
                f"at index {i}: {repr(e)}"
            )

        # Progress
        if (i + 1) % 1000 == 0:
            print(
                f"  Checked {i + 1}/{len(dataset)}"
            )

    print(f"  ✓ {name}: all images valid")


final_check(
    train_dataset,
    "training dataset"
)

final_check(
    test_dataset,
    "testing dataset"
)


train_dataset.save_to_disk(
    TRAIN_SAVE_DIR
)

test_dataset.save_to_disk(
    TEST_SAVE_DIR
)

print("\n====================================")
print("Datasets saved successfully!")
print("====================================")

print(
    f"Train: {os.path.abspath(TRAIN_SAVE_DIR)}"
)

print(
    f"Test:  {os.path.abspath(TEST_SAVE_DIR)}"
)

print(
    f"\nTrain samples: {len(train_dataset)}"
)

print(
    f"Test samples:  {len(test_dataset)}"
)

print(
    f"Broken images removed: {len(broken_indices)}"
)
# --------------------------------------------------
# Save datasets
# --------------------------------------------------

train_dataset.save_to_disk(
    TRAIN_SAVE_DIR
)

test_dataset.save_to_disk(
    TEST_SAVE_DIR
)

print("\n====================================")
print("Datasets saved successfully!")
print("====================================")

print(
    f"Train: {os.path.abspath(TRAIN_SAVE_DIR)}"
)

print(
    f"Test:  {os.path.abspath(TEST_SAVE_DIR)}"
)

print(
    f"\nTrain samples: {len(train_dataset)}"
)

print(
    f"Test samples:  {len(test_dataset)}"
)
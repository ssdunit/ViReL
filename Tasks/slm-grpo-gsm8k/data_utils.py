"""Data loading and preprocessing for the GSM8K math dataset."""

import re
from typing import Optional

from datasets import load_dataset

from config import ExperimentConfig


SYSTEM_PROMPT = """You are a mathematics tutor solving grade-level math problems.

Follow these rules:

1. Always include both <reasoning> and <answer> tags.
2. Always close both tags correctly.
3. Do not use Markdown code blocks.
4. Do not add any text before <reasoning>.
5. Do not add any text after </answer>.
6. The <answer> section must contain only the final numerical answer.
7. Do not write words such as "The answer is" inside the <answer> tag.
8. Give the final answer as a number whenever possible.
9. For decimal answers, use a decimal number.
10. For negative answers, include the minus sign.
11. Carefully check your calculations before giving the final answer.

Use exactly this format:

<reasoning>
Step-by-step mathematical reasoning goes here.
</reasoning>
<answer>
8
</answer>
"""


HASH_ANSWER_PATTERN = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


def extract_hash_answer(text: str) -> Optional[str]:
    """Extract the final answer after ####."""

    match = HASH_ANSWER_PATTERN.search(text)

    if match:
        return match.group(1).strip()

    return None


def extract_reasoning(text: str) -> str:
    """Extract GSM8K reasoning from the answer field.

    GSM8K format:

    reasoning steps...
    #### final_answer
    """

    if "####" in text:
        reasoning = text.split("####", 1)[0].strip()
        return reasoning

    return text.strip()


def extract_final_answer(text: str) -> Optional[str]:
    """Extract the final answer from GSM8K format."""

    if "####" in text:
        answer = text.split("####", 1)[1].strip()
        return answer

    return None


def format_prompt(question: str) -> list[dict]:
    """Format question into chat messages."""

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]


def load_gsm8k_dataset(
    config: ExperimentConfig,
) -> tuple:
    """Load and preprocess GSM8K dataset."""

    dataset = load_dataset(
        config.data.dataset_name,
        "main",
        split=[
            config.data.dataset_split_train,
            config.data.dataset_split_test,
        ],
    )

    train_dataset = dataset[0]
    test_dataset = dataset[1]

    if config.data.max_samples is not None:
        train_dataset = train_dataset.select(
            range(min(config.data.max_samples, len(train_dataset)))
        )

    def _preprocess(example: dict) -> dict:
        gsm8k_answer = str(example["answer"])

        return {
            "prompt": format_prompt(example["question"]),


            # Step-by-step reasoning as a string
            "reasoning": extract_reasoning(gsm8k_answer),

            # Final answer as a string
            "final_answer": extract_final_answer(gsm8k_answer),
        }

    train_dataset = train_dataset.map(
        _preprocess,
        num_proc=config.data.num_proc,
        desc="Formatting train prompts",
    )

    test_dataset = test_dataset.map(
        _preprocess,
        num_proc=config.data.num_proc,
        desc="Formatting test prompts",
    )

    # Keep prompt, reasoning, final answer and complete answer
    columns_to_keep = (
        "prompt",
        "reasoning",
        "final_answer",
    )

    train_dataset = train_dataset.remove_columns(
        [
            c
            for c in train_dataset.column_names
            if c not in columns_to_keep
        ]
    )

    test_dataset = test_dataset.remove_columns(
        [
            c
            for c in test_dataset.column_names
            if c not in columns_to_keep
        ]
    )

    return train_dataset, test_dataset
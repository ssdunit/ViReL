import re
ANSWER_PATTERN = re.compile(
    r"^<think>.*?</think>\s*<answer>\s*([A-F])\s*</answer>$",
    re.DOTALL,
)
def extract_answer(completion):
    """
    Extract the answer letter from a model completion.
    """

    match = ANSWER_PATTERN.match(completion)

    if match is None:
        return None

    return match.group(1).upper()

    def format_reward(completions, **kwargs):
    """
    Reward outputs that follow the required format.

    Correct:
        <think>...</think>
        <answer>B</answer>

    Reward:
        Correct format -> 1.0
        Incorrect format -> 0.0
    """

    rewards = []

    for completion in completions:

        content = completion[0]["content"]

        if ANSWER_PATTERN.match(content):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards
def answer_correctness_reward(
    completions,
    answer_letter,
    **kwargs,
):
    """
    Reward the model for selecting the correct option.

    Correct answer -> 2.0
    Incorrect answer -> 0.0
    Invalid format -> 0.0
    """

    rewards = []

    for completion, ground_truth in zip(
        completions,
        answer_letter,
    ):

        content = completion[0]["content"]

        predicted_answer = extract_answer(content)

        if predicted_answer is None:
            rewards.append(0.0)
            continue

        predicted_answer = predicted_answer.upper()
        ground_truth = str(ground_truth).strip().upper()

        if predicted_answer == ground_truth:
            rewards.append(2.0)
        else:
            rewards.append(0.0)

    return rewards
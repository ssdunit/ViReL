import re
FULL_FORMAT_PATTERN = re.compile(
    r"<think>.*?</think>\s*<answer>\s*([A-F])\s*</answer>",
    re.DOTALL,
)
BASE_FORMAT_PATTERN = re.compile(
    r"</think>\s*<answer>\s*([A-F])\s*</answer>",
    re.DOTALL,
)
ANSWER_PATTERN = re.compile(
    r"<answer>\s*([A-F])\s*</answer>",
    re.DOTALL,
)
def extract_answer(completion):
    """
    Extract the answer letter from a model completion.
    """

    match = ANSWER_PATTERN.search(completion)

    if match is None:
        return None

    return match[1].upper()

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
        if isinstance(completion, list):
            content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        elif isinstance(completion, dict):
            content = completion.get("content", "")
        else:
            content = str(completion)

        content = content.strip()
        with open("output.txt","a") as file:
            file.write(f"---RAW COMPLETION---\n{content}\n---END---")
        score = 0.0

        if BASE_FORMAT_PATTERN.search(content):
            score = 1.0
        if "<think>" in content:
            score = 2.0

        rewards.append(score)

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

        #content = completion[0]["content"]
        if isinstance(completion, list):
            content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        elif isinstance(completion, dict):
            content = completion.get("content", "")
        else:
            content = str(completion)
        predicted_answer = extract_answer(content)

        if predicted_answer is None:
            #print(f"[CHECK] NO MATCH — repr: {content!r}", flush=True)
            rewards.append(0.0)
            continue

        predicted_answer = predicted_answer.upper()
        ground_truth = str(ground_truth).strip().upper()
        #print(f"[CHECK] predicted={predicted_answer!r}  ground_truth={ground_truth!r}  match={predicted_answer == ground_truth}")
        if predicted_answer == ground_truth:
            rewards.append(3.0)
        else:
            rewards.append(0.0)

    return rewards
def reasoning_length_reward(
    completions,
    min_length: int = 20,
    target_length: int = 150,
    max_length: int = 400,
    **kwargs,
):
    """
    Reward the length of reasoning content, extracted leniently.

    Unlike a strict <think>...</think> match, this rewards length
    even if only one of the two tags is present:
        - both tags present  -> content between them
        - only <think> present -> everything after it
        - only </think> present -> everything before it
        - neither present -> 0.0 (no reasoning content to evaluate)

    Scoring (based on word count of extracted reasoning content):
        < min_length words      -> scaled down (too little reasoning)
        min_length..max_length  -> 1.0
        > max_length words      -> decays back down (rambling / truncation risk)
    """

    # Safely extract the column. Defaults to an empty list if stripped/missing.
    answer_letters = kwargs.get("answer_letter", [])
    
    # Safety net: If the column is missing, return 0.0 rewards instead of crashing
    if not answer_letters:
        print("WARNING: 'answer_letter' column is missing from kwargs!")
        return [0.0] * len(completions)

    rewards = []
    # Now zip will safely run
    for completion, gt in zip(completions, answer_letters):
        # Extract predicted answer from completion
        pred = extract_answer(completion) # (Assuming you have your extraction logic here)
        
        # Only reward reasoning length IF the answer is actually correct
        if pred == gt:
            length_score = min(len(completion) / 512.0, 1.0)
            rewards.append(length_score)
        else:
            rewards.append(0.0)
            
    return rewards

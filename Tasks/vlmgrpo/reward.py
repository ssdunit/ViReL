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
def partial_reward_system(completions, **kwargs):
    """
    Reward used during the first 500 GRPO steps.

    Goal:
    - Teach the model to follow the expected output format.
    - Encourage valid answers.
    - Do not strongly optimize correctness yet.
    """
    rewards =[]
    for completion in  completions:
         if isinstance(completion, list):
             content = completion[-1].get("content", "")
         elif isinstance(completion, dict):
             content = completion.get("content", "")
         else:
             content = str(completion)
        # Exactly one <reason> tag
         if content.count("<reason>") == 1:
             reward += 0.15

        # Exactly one </reason> tag
         if content.count("</reason>") == 1:
             reward += 0.15

        # Exactly one <answer> tag
         if content.count("<answer>") == 1:
             reward += 0.15

        # Exactly one </answer> tag
         if content.count("</answer>") == 1:
             reward += 0.15

        # Non-empty response
         if len(content.strip()) > 20:
             reward += 0.25

         rewards.append(reward)

    return rewards

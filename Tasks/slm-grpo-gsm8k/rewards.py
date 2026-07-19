import re
from typing import List
from config import DataConfig
import math

"""
re guide:

.*?: All the content between the tags
\s*: Handles any Accidental spaces or newlines  
^\s*: Handles any leading accidental spaces or newlines
DOTALL (flag): Matches all the characters including newlines
IGNORECASE (flag) : Ignores the case sensitivity 
"""
def get_text(completion):
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion

reasoning_start = DataConfig.F_tags.reasoning_start
reasoning_end = DataConfig.F_tags.reasoning_end
thinking_start = DataConfig.F_tags.thinking_start
thinking_end = DataConfig.F_tags.thinking_end
solution_start = DataConfig.F_tags.solution_start
solution_end = DataConfig.F_tags.solution_end

#Corectness reward
def correctness_reward(prompts,completions,answer,**kwargs)->List[float]:
    """
        Rewards for correct final answer

        Args:
            completions: List of model completions

        IF model_ans == given_answer 
        Give Reward 1.0
        else
        Give Reward 0.0
    """
    rewards = []
    for completion, gt_answer in zip(completions, answer):
        text = get_text(completion)
        
        # Use case-insensitive search so we don't punish math for a formatting error
        match = re.search(rf"{solution_start}\s*(.*?)\s*{solution_end}", text, re.DOTALL | re.IGNORECASE)
        
        gt_numeric = "".join(c for c in gt_answer if c.isdigit() or c=='.')

        if match:
            model_answer = match.group(1).strip()
            
            numbers = re.findall(r'\d+\.?\d*', model_answer)
            if numbers:
                model_numeric = numbers[-1]
                
                if model_numeric == gt_numeric and len(model_numeric) > 0:
                    
                    if model_answer == model_numeric:
                        rewards.append(1.0)
   
                    else:
                        rewards.append(0.5)
                    continue
                    
        rewards.append(0.0)
    return rewards

#Strict formatting reward
def strict_format_reward(prompts,completions,**kwargs)->List[float]:
    """
        Rewards based on correct formatting

        Args:
            completions: List of model completions

        pattern = (xml generation pattern)

        IF completion matches pattern
        Give reward = 0.5
        else reward = 0.0
    """
    rewards = []
    pattern = rf"\s*{reasoning_start}.*?{reasoning_end}\s*{solution_start}.*?{solution_end}\s*"
    
    for completion in completions:
        text = get_text(completion)
        if re.search(pattern, text.strip(), re.DOTALL | re.IGNORECASE):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

#xml formatting reward
def xml_formatting_reward(prompts,completions,**kwargs)->List[float]:
    """
        Give Rewards based of correctness opening and closing tags

        Args:
            completions: List of model completions

        tags = [
            (Opening_tag,Closed_tag),
            ...
        ]

        IF both in completions ->
            give reward = 0.2
            else reward = 0.0
    """
    rewards = []
    for completion in completions:
        text = get_text(completion)
        text_lower = text.lower() 
        
        tags = [
            (reasoning_start, reasoning_end),
            (solution_start, solution_end)
        ]
        
        score = 0.0
        for open_tag, close_tag in tags:
            if open_tag in text and close_tag in text:
                score += 0.2  
                
            elif open_tag.lower() in text_lower and close_tag.lower() in text_lower:
                score += 0.1
                
        rewards.append(score)
    return rewards

#Cosined scaled rewards
def cosine_scaled_reward(prompts,completions,answer,**kwargs)->List[float]:
    """
        Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            answer: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
            cosine = min(gen_length/max_length,1.0)
        
        Reward scaling: min_value + 0.5*(max_value-min_value)*(1.0+cosine)
    """
    rewards = []
    max_length = 2048
    for completion, gt_answer in zip(completions, answer):
        text = get_text(completion)
        is_correct = False
        
        match = re.search(rf"{solution_start}\s*(.*?)\s*{solution_end}", text, re.DOTALL | re.IGNORECASE)
        
        gt_numeric = "".join(c for c in gt_answer if c.isdigit() or c=='.')

        if match:
            model_answer = match.group(1).strip()

            numbers = re.findall(r'\d+\.?\d*', model_answer)
            if numbers:
                model_numeric = numbers[-1]
                if model_numeric == gt_numeric and len(model_numeric) > 0:
                    is_correct = True
                
        gen_len = len(text)

        if is_correct:
            min_value, max_value = 0.5, 1.0
        else:
            min_value, max_value = 0.0, 0.5
            
        progress = min(gen_len / max_length, 1.0)
        cosine = math.cos(progress * math.acos(-1))
        reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        rewards.append(float(reward))

    return rewards

#Repetition Penalty
def repetition_penalty_reward(prompts,completions,answer,**kwargs)->List[float]:
    """
        Penalty for repetitions in given n_gram chunks  
    """
    n_gram = 3 
    max_penalty = -1.0 
    rewards = []

    for completion in completions:
        text = get_text(completion)
        words = text.split()
        if len(words) < n_gram:
            rewards.append(0.0)
            continue

        n_grams = [tuple(words[i:i+n_gram]) for i in range(len(words) - n_gram + 1)]
        unique_n_grams = set(n_grams)
        
        if len(n_grams) > 0:
            unique_ratio = len(unique_n_grams) / len(n_grams)
        else:
            unique_ratio = 1.0

        if unique_ratio < 0.8:
            penalty = max_penalty * (1.0 - unique_ratio)
            rewards.append(penalty)
        else:
            rewards.append(0.0)
            
    return rewards

#reasoning length reward
def reasoning_length_reward(completions, **kwargs):
    """
    Replaces the cosine_scaled_reward. 
    Penalizes the model for bypassing the reasoning phase with ultra-short text.
    """
    rewards = []
    
    # Set a minimum character threshold for GSM8k reasoning. 
    # 150-200 characters is a safe baseline for a multi-step math problem.
    min_char_limit = 150 
    
    for completion in completions:
        # Utilizing your updated case-insensitive regex
        match = re.search(rf'{reasoning_start}(.*?){reasoning_end}', completion, re.DOTALL | re.IGNORECASE)
        
        if match:
            reasoning_text = match.group(1).strip()
            
            if len(reasoning_text) >= min_char_limit:
                # The model showed its work. Award the baseline completion points.
                rewards.append(0.5) 
            else:
                # The model attempted the silence exploit. Hard penalty.
                rewards.append(0.0)
        else:
            # The tags were dropped entirely.
            rewards.append(0.0)
            
    return rewards
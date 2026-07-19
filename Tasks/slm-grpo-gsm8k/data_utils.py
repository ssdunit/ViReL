from ollama import Client
from config import DataConfig
from datasets import load_dataset
import os
# Force the global host environment variable for the underlying http libraries
os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"

config = DataConfig()

reasoning_start = config.F_tags.reasoning_start
reasoning_end = config.F_tags.reasoning_end
thinking_start = config.F_tags.thinking_start
thinking_end = config.F_tags.thinking_end
solution_start = config.F_tags.solution_start
solution_end = config.F_tags.solution_end

def load_gsm8k_dataset(config):
    #System prompt
    system_prompt = f"""You are a strict mathematics reasoning assistant. 
You must solve the math problem step-by-step. Your entire response MUST be formatted exactly like this template:

{reasoning_start}
Write your step-by-step mathematical logic and calculations here.
{reasoning_end}
{solution_start}
Write ONLY the final numerical answer here.
{solution_end}"""

    #Testing tags and system prompt
    print(f"Reasoning format: {reasoning_start} ... {reasoning_end}")
    print(f"Thinking format: {thinking_start} ... {thinking_end}")
    print(f"Solution format: {solution_start} ... {solution_end}")
    print(system_prompt)

    ###formatting function

    #Extracting the answer from the answer label
    def extract_hash_answer(example):
        if "####" in example:
            return example.split("####")[-1].strip()
        return None

    #Extracting reasoning from the answer label
    def extract_reasoning(example):
        if "####" in example:
            return example.split("####")[0].strip()
        return None 

    #Added dynamic thinking step for label which defines the methods or algorithms that can be used to solve the problem. 
    #THE GIVEN PROBLEM IS NOT SOLVED HERE
    """
    def dynamic_thinking_step(question:str,reasoning:str, config:DataConfig)-> str:
        client = Client(
            host = "http://127.0.0.1:11434",
            timeout = config.DST.timeout,
        )
        prompt = f"Read this given maths problem {question}, and its reasoning steps {reasoning},give the probable methods or algorithms in 1 to 2 lines. Strictly do not solve the problem."
        messages = [
            {"role":"system","content":config.DST.system_prompt},
            {"role":"user","content":prompt}
        ]
        try:
            response = client.chat(
                model = config.DST.model_name,
                messages = messages,
                options={
                    "temperature":config.DST.temperature,
                    "num_predict":config.DST.max_tokens,
                    "stop":config.DST.stop_sequences,
                }
            )
            return response['message']['content'].strip()
        
        except Exception as e:
            # Catch timeout errors (or other connection issues) cleanly
            print(f"Failed to generate thinking block: {e}")
            return "I will analyze the given quantities and calculate intermediate steps sequentially."
    """
    def format_custom_gsm8k(example):
        question = example['question']
        raw_answer = example['answer']

        final_answer = extract_hash_answer(raw_answer)
        gsm8k_reasoning  = extract_reasoning(raw_answer)

        #dynamic_thinking = dynamic_thinking_step(question,gsm8k_reasoning,config)
        assistant_response = (
            f"{reasoning_start}\n"
            f"{gsm8k_reasoning}\n"
            f"{reasoning_end}\n"
            f"{solution_start}\n"
            f"{final_answer}\n"
            f"{solution_end}"
        )

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{question}\n\nStart your response immediately with {reasoning_start}\n"},
        ]
        
        return {
            "prompt": prompt_messages,
            "answer": final_answer,          
            "reference_response": assistant_response 
        }
        

    train_dataset = load_dataset(config.dataset,'main',split="train")
    test_dataset = load_dataset(config.dataset,'main',split="test")
    formatted_train_dataset = train_dataset.map(
        format_custom_gsm8k,
        load_from_cache_file=False,
    )
    formatted_test_dataset = test_dataset.map(
        format_custom_gsm8k,
        load_from_cache_file=True,
    )

    return formatted_train_dataset,formatted_test_dataset

    """
    print(f"✅ Dataset loaded and processed!")
    print(f"📊 Training examples: {len(formatted_dataset):,}")
    print(f"🎯 Sample question: {formatted_dataset[0]['prompt'][1]['content']}...")
    print(f"🎯 Sample answer: {formatted_dataset[0]['answer']}")

    # Show structure of first example for verification
    print(f"\n📋 Example structure:")
    print(f"   Prompt: {len(formatted_dataset[0]['prompt'])} messages (system + user)")
    print(f"   Answer: {formatted_dataset[0]['answer']} (ground truth for rewards)")
    """
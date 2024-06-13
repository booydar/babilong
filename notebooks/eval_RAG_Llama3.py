import os

import datasets
import pandas as pd
import torch

import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from tqdm.notebook import tqdm
import warnings
import gc 
from pathlib import Path

from babilong.prompts import DEFAULT_PROMPTS, get_formatted_input

## load ChatQA-1.5 tokenizer and model
model_id = "nvidia/Llama3-ChatQA-1.5-8B"
device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(model_id, device_map=device)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map=device)

tokenizer.pad_token_id = tokenizer.eos_token_id

## load retriever tokenizer and model
retriever_tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder', device_map=device)
context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder', device_map=device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

generate_kwargs = {
    'num_beams': 1,
    'do_sample': False,
    'temperature': None,
    'top_p': None,
    'top_k': None,
}

def split_text_to_sent(text):
    # Pattern to split on period, exclamation, or question mark followed by space or end of string
    # Adjust the pattern to handle more edge cases if necessary
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return sentences

def get_formatted_input(messages, context, question, post_prompt):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    formatted_input = system

    if len(messages) > 0:
        if messages[0]['role'] == 'system':
            formatted_input += messages[0]['content'] + '\n\n'
            messages = messages[1:]


        conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) 
        formatted_input += conversation + '\n\n'
    formatted_input += context + "\n\n" + question
    
    if post_prompt: 
        formatted_input += "\n\n" + post_prompt

    formatted_input += " Assistant: "
    return formatted_input


def format_examples(default_examples):
    if len(default_examples) == 0:
        return [], []
    
    examples = default_examples.split('<example>\n')
    examples = [e[:e.index("\n</example>")] for e in examples if len(e) > 0]
    inputs = [e[:e.index("\nAnswer")] for e in examples]
    outputs = [e[e.index("\nAnswer") + 9:] for e in examples]
    return inputs, outputs

def get_messages(context, question, examples, instruction, post_prompt):
    # pre_prompt - general instruction
    # examples - in-context examples
    # post_prompt - any additional instructions after examples
    # context - text to use for qa
    # question - question to answer based on context
    inputs, outputs = format_examples(examples)
    messages = []
    if len(instruction) > 0:
        messages.append({"role": "system", "content": instruction })

    for i, o in zip(inputs, outputs):
        messages += [
            {"role": "user", "content": i},
            {"role": "assistant", "content": o}
        ]

    return messages


def rag(sample, prompt_cfg, batchsize=1500):
    doc = sample['input']
    chunk_list = split_text_to_sent(doc)

    formatted_query_for_retriever = f"User: {sample['question']}"

    query_input = retriever_tokenizer(formatted_query_for_retriever, return_tensors='pt').to(query_encoder.device)
    with torch.no_grad():
        query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]

    # Initialize lists to store retrieved chunks and their embeddings
    retrieved_chunks = []
    similarities = torch.tensor([]).to(query_emb.device)

    # Process chunks in batches to manage memory usage
    print('sentences:',len(chunk_list))
    for i in range(0, len(chunk_list), batchsize):  # Adjust the batch size based on your memory constraints
        batch_chunks = chunk_list[i:i+batchsize]
        ctx_input = retriever_tokenizer(batch_chunks, padding=True, truncation=True, max_length=512, return_tensors='pt').to(context_encoder.device)
        with torch.no_grad():
            ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]

        # Compute similarity scores using dot product
        batch_similarities = query_emb.matmul(ctx_emb.transpose(0, 1))  # (1, num_ctx)
        similarities = torch.cat((similarities, batch_similarities), dim=-1)
        
        # Clear memory
        del ctx_input, ctx_emb, batch_chunks
        torch.cuda.empty_cache()
        gc.collect()

    ranked_results = torch.argsort(similarities, dim=-1, descending=True)[0][:5]
    retrieved_chunks = [chunk_list[idx] for idx in ranked_results.tolist()]

    # Now perform generation with retrieved context
    context = "\n\n".join(retrieved_chunks)

    messages = get_messages(context=context, question=sample['question'], 
                                    examples=prompt_cfg['examples'], instruction=prompt_cfg['instruction'],
                                    post_prompt=prompt_cfg['post_prompt'])

    formatted_input = get_formatted_input(messages, context, sample['question'], post_prompt=prompt_cfg['post_prompt'])
    tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=20, eos_token_id=terminators, **generate_kwargs)

    response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
    return response


# dataframes with results will be stored in the results/ folder
warnings.filterwarnings("ignore")
os.makedirs('../babilong_evals/', exist_ok=True)

# Zero shot
TEMPLATE = '{instruction}\n{examples}\n{post_prompt}\nContext: {context}\n\nQuestion: {question}'

tasks = ['qa1', 'qa2', 'qa3', 'qa4', 'qa5']
sample_sizes = ['0k', '1k', '2k', '4k', '8k', '16k', '32k']

for task in tqdm(tasks, desc='Eval Tasks'):   
    for split_name in tqdm(sample_sizes, desc=f'Processing message_length'):        
        data = datasets.load_dataset("booydar/babilong-1k-samples", split_name)
        task_dataset = data[task]

        prompt_cfg = {
            'instruction': '',
            'examples': '', 
            'post_prompt': '',
            'template': TEMPLATE,
        }
        prompt_name = [f'{k}_no' if len(prompt_cfg[k]) == 0 else f'{k}_yes' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)

        outfile = f'babilong_evals/{task}_msg_{split_name}.csv'
        outfile = Path(f'/home/booydar/rmt/babilong/babilong_evals/RAG_Llama-3/{task}_{split_name}_{prompt_name}.csv')
        outfile.parent.mkdir(parents=True, exist_ok=True)
        cfg_file = f'/home/booydar/rmt/babilong/babilong_evals/RAG_Llama-3/{task}_{split_name}_{prompt_name}.json'
        json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)
        df = pd.DataFrame({
            'target': [],
            'output': [],
            'result': [],
            'question': []
        })

        for sample in tqdm(task_dataset[task], desc=f'Processing samples for {task}, length {split_name}'):       
            question = sample['question']
            
            target = sample['target']           
            output = rag(sample, prompt_cfg)
            llm_response = tokenizer.decode(output, skip_special_tokens=True)
            is_substring = target.lower() in llm_response.lower()

            df.loc[len(df)] = [target, llm_response, is_substring, question]
            df.to_csv(outfile)

    del(task_dataset)


# Few shot
TEMPLATE = '{instruction}\n{examples}\n{post_prompt}\nContext: {context}\n\nQuestion: {question}'
for task in tqdm(tasks, desc='Eval Tasks'):   
    for split_name in tqdm(sample_sizes, desc=f'Processing message_length'):        
        data = datasets.load_dataset("booydar/babilong-1k-samples", split_name)
        task_dataset = data[task]

        prompt_cfg = {
            'instruction': DEFAULT_PROMPTS[task]['instruction'],
            'examples': DEFAULT_PROMPTS[task]['examples'], 
            'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'],
            'template': TEMPLATE,
        }

        prompt_name = [f'{k}_no' if len(prompt_cfg[k]) == 0 else f'{k}_yes' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)

        outfile = f'babilong_evals/{task}_msg_{split_name}.csv'
        outfile = Path(f'/home/booydar/rmt/babilong/babilong_evals/RAG_Llama-3/{task}_{split_name}_{prompt_name}.csv')
        outfile.parent.mkdir(parents=True, exist_ok=True)
        cfg_file = f'/home/booydar/rmt/babilong/babilong_evals/RAG_Llama-3/{task}_{split_name}_{prompt_name}.json'
        json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)
        df = pd.DataFrame({
            'target': [],
            'output': [],
            'result': [],
            'question': []
        })

        for sample in tqdm(task_dataset[task], desc=f'Processing samples for {task}, length {split_name}'):       
            question = sample['question']
            
            target = sample['target']           
            output = rag(sample, prompt_cfg)
            llm_response = tokenizer.decode(output, skip_special_tokens=True)
            is_substring = target.lower() in llm_response.lower()

            df.loc[len(df)] = [target, llm_response, is_substring, question]
            df.to_csv(outfile)

    del(task_dataset)



sample_sizes = ['64k', '128k', '512k', '1M']

for task in tqdm(tasks, desc='Eval Tasks'):   
    for split_name in tqdm(sample_sizes, desc=f'Processing message_length'):        
        data = datasets.load_dataset("booydar/babilong", split_name)
        task_dataset = data[task]

        prompt_cfg = {
            'instruction': '',
            'examples': '', 
            'post_prompt': '',
            'template': TEMPLATE,
        }
        prompt_name = [f'{k}_no' if len(prompt_cfg[k]) == 0 else f'{k}_yes' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)

        outfile = f'babilong_evals/{task}_msg_{split_name}.csv'
        outfile = Path(f'/home/booydar/rmt/babilong/babilong_evals/RAG_Llama-3/{task}_{split_name}_{prompt_name}.csv')
        outfile.parent.mkdir(parents=True, exist_ok=True)
        cfg_file = f'/home/booydar/rmt/babilong/babilong_evals/RAG_Llama-3/{task}_{split_name}_{prompt_name}.json'
        json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)
        df = pd.DataFrame({
            'target': [],
            'output': [],
            'result': [],
            'question': []
        })

        for sample in tqdm(task_dataset[task], desc=f'Processing samples for {task}, length {split_name}'):       
            question = sample['question']
            
            target = sample['target']           
            output = rag(sample, prompt_cfg)
            llm_response = tokenizer.decode(output, skip_special_tokens=True)
            is_substring = target.lower() in llm_response.lower()

            df.loc[len(df)] = [target, llm_response, is_substring, question]
            df.to_csv(outfile)

    del(task_dataset)


# Few shot
TEMPLATE = '{instruction}\n{examples}\n{post_prompt}\nContext: {context}\n\nQuestion: {question}'
for task in tqdm(tasks, desc='Eval Tasks'):   
    for split_name in tqdm(sample_sizes, desc=f'Processing message_length'):        
        data = datasets.load_dataset("booydar/babilong", split_name)
        task_dataset = data[task]

        prompt_cfg = {
            'instruction': DEFAULT_PROMPTS[task]['instruction'],
            'examples': DEFAULT_PROMPTS[task]['examples'], 
            'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'],
            'template': TEMPLATE,
        }

        prompt_name = [f'{k}_no' if len(prompt_cfg[k]) == 0 else f'{k}_yes' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)

        outfile = f'babilong_evals/{task}_msg_{split_name}.csv'
        outfile = Path(f'/home/booydar/rmt/babilong/babilong_evals/RAG_Llama-3/{task}_{split_name}_{prompt_name}.csv')
        outfile.parent.mkdir(parents=True, exist_ok=True)
        cfg_file = f'/home/booydar/rmt/babilong/babilong_evals/RAG_Llama-3/{task}_{split_name}_{prompt_name}.json'
        json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)
        df = pd.DataFrame({
            'target': [],
            'output': [],
            'result': [],
            'question': []
        })

        for sample in tqdm(task_dataset[task], desc=f'Processing samples for {task}, length {split_name}'):       
            question = sample['question']
            
            target = sample['target']           
            output = rag(sample, prompt_cfg)
            llm_response = tokenizer.decode(output, skip_special_tokens=True)
            is_substring = target.lower() in llm_response.lower()

            df.loc[len(df)] = [target, llm_response, is_substring, question]
            df.to_csv(outfile)

    del(task_dataset)


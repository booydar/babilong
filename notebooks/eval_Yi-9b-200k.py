import re
import os
# os.chdir('..')

import sys
sys.path.append('..')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from tqdm.auto import tqdm
import pandas as pd
import time
import json

from pathlib import Path

from babilong.prompts import DEFAULT_PROMPTS, get_formatted_input


model_name = '01-ai/Yi-9B-200k'
dtype = torch.bfloat16
device = 'auto'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                             device_map=device, torch_dtype=dtype,
                                             attn_implementation='flash_attention_2')
model = model.eval()

generate_kwargs = {
    'num_beams': 1,
    'do_sample': False,
    'temperature': None,
    'top_p': None,
    'top_k': None,
}

tasks = ['qa1', 'qa2','qa3', 'qa4', 'qa5']
split_names = ['0k', '1k', '2k', '4k', '8k', '16k', '32k']



TEMPLATE = '{instruction}\n{examples}\n{post_prompt}\nContext: {context}\n\nQuestion: {question}'

def clean_examples(initial_examples):
    examples = re.sub('<example>', 'Example:', initial_examples)
    examples = re.sub('</example>', '', examples)
    return examples

def get_formatted_input(context, question, examples, instruction, post_prompt, template=TEMPLATE):
    # pre_prompt - general instruction
    # examples - in-context examples
    # post_prompt - any additional instructions after examples
    # context - text to use for qa
    # question - question to answer based on context
    cleaned_examples = clean_examples(examples)
    
    formatted_input = template.format(instruction=instruction, examples=cleaned_examples, 
                                        post_prompt=post_prompt, context=context, question=question)
    return formatted_input.strip()


# zero-shot
for task in tqdm(tasks, desc='tasks'):
    print(task)
    prompt_cfg = {
        'instruction': '',
        'examples': '', 
        'post_prompt': '',
        'template': TEMPLATE,
    }
    
    prompt_name = [f'{k}_no' if len(prompt_cfg[k]) == 0 else f'{k}_yes' for k in prompt_cfg if k != 'template']
    prompt_name = '_'.join(prompt_name)
    for split_name in tqdm(split_names, desc='lengths'):
        data = datasets.load_dataset("booydar/babilong-1k-samples", split_name)
        task_data = data[task]

        outfile = Path(f'/home/jovyan/rmt/babilong/babilong_evals/{model_name}/{task}_{split_name}_{prompt_name}.csv')
        outfile.parent.mkdir(parents=True, exist_ok=True)
        cfg_file = f'/home/jovyan/rmt/babilong/babilong_evals/{model_name}/{task}_{split_name}_{prompt_name}.json'
        json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)

        df = pd.DataFrame({'target': [], 'output': [], 'question': []})

        for sample in tqdm(task_data):
            target = sample['target']
            context = sample['input']
            question = sample['question']

            input_text = get_formatted_input(context, question, prompt_cfg['examples'],
                                             prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                                             template=TEMPLATE)
            input = tokenizer(input_text, return_tensors="pt", add_special_tokens=True).to(model.device)
            sample_length = input['input_ids'].shape[1]
            with torch.no_grad():
                output = model.generate(**input, max_length=sample_length+25, **generate_kwargs)
            output = output[0][input['input_ids'].shape[1]:]
            output = tokenizer.decode(output, skip_special_tokens=True).strip()

            df.loc[len(df)] = [target, output, question]
            df.to_csv(outfile)


# few-shot
for task in tqdm(tasks, desc='tasks'):
    print(task)
    prompt_cfg = {
        'instruction': DEFAULT_PROMPTS[task]['instruction'],
        'examples': DEFAULT_PROMPTS[task]['examples'], 
        'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'],
        'template': TEMPLATE,
    }
    
    prompt_name = [f'{k}_no' if len(prompt_cfg[k]) == 0 else f'{k}_yes' for k in prompt_cfg if k != 'template']
    prompt_name = '_'.join(prompt_name)
    for split_name in tqdm(split_names, desc='lengths'):
        data = datasets.load_dataset("booydar/babilong-1k-samples", split_name)
        task_data = data[task]

        outfile = Path(f'/home/jovyan/rmt/babilong/babilong_evals/{model_name}/{task}_{split_name}_{prompt_name}.csv')
        outfile.parent.mkdir(parents=True, exist_ok=True)
        cfg_file = f'/home/jovyan/rmt/babilong/babilong_evals/{model_name}/{task}_{split_name}_{prompt_name}.json'
        json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)

        df = pd.DataFrame({'target': [], 'output': [], 'question': []})

        for sample in tqdm(task_data):
            target = sample['target']
            context = sample['input']
            question = sample['question']

            input_text = get_formatted_input(context, question, prompt_cfg['examples'],
                                             prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                                             template=TEMPLATE)

            input = tokenizer(input_text, return_tensors="pt", add_special_tokens=True).to(model.device)
            sample_length = input['input_ids'].shape[1]
            with torch.no_grad():
                output = model.generate(**input, max_length=sample_length+25, **generate_kwargs)
            output = output[0][input['input_ids'].shape[1]:]
            output = tokenizer.decode(output, skip_special_tokens=True).strip()

            df.loc[len(df)] = [target, output, question]
            df.to_csv(outfile)


split_names = ['64k', '128k']

# zero-shot
for task in tqdm(tasks, desc='tasks'):
    print(task)
    prompt_cfg = {
        'instruction': '',
        'examples': '', 
        'post_prompt': '',
        'template': TEMPLATE,
    }
    
    prompt_name = [f'{k}_no' if len(prompt_cfg[k]) == 0 else f'{k}_yes' for k in prompt_cfg if k != 'template']
    prompt_name = '_'.join(prompt_name)
    for split_name in tqdm(split_names, desc='lengths'):
        data = datasets.load_dataset("booydar/babilong", split_name)
        task_data = data[task]

        outfile = Path(f'/home/jovyan/rmt/babilong/babilong_evals/{model_name}/{task}_{split_name}_{prompt_name}.csv')
        outfile.parent.mkdir(parents=True, exist_ok=True)
        cfg_file = f'/home/jovyan/rmt/babilong/babilong_evals/{model_name}/{task}_{split_name}_{prompt_name}.json'
        json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)

        df = pd.DataFrame({'target': [], 'output': [], 'question': []})

        for sample in tqdm(task_data):
            target = sample['target']
            context = sample['input']
            question = sample['question']

            input_text = get_formatted_input(context, question, prompt_cfg['examples'],
                                             prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                                             template=TEMPLATE)
            input = tokenizer(input_text, return_tensors="pt", add_special_tokens=True).to(model.device)
            sample_length = input['input_ids'].shape[1]
            with torch.no_grad():
                output = model.generate(**input, max_length=sample_length+25, **generate_kwargs)
            output = output[0][input['input_ids'].shape[1]:]
            output = tokenizer.decode(output, skip_special_tokens=True).strip()

            df.loc[len(df)] = [target, output, question]
            df.to_csv(outfile)


# few-shot
for task in tqdm(tasks, desc='tasks'):
    print(task)
    prompt_cfg = {
        'instruction': DEFAULT_PROMPTS[task]['instruction'],
        'examples': DEFAULT_PROMPTS[task]['examples'], 
        'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'],
        'template': TEMPLATE,
    }
    
    prompt_name = [f'{k}_no' if len(prompt_cfg[k]) == 0 else f'{k}_yes' for k in prompt_cfg if k != 'template']
    prompt_name = '_'.join(prompt_name)
    for split_name in tqdm(split_names, desc='lengths'):
        data = datasets.load_dataset("booydar/babilong", split_name)
        task_data = data[task]

        outfile = Path(f'/home/jovyan/rmt/babilong/babilong_evals/{model_name}/{task}_{split_name}_{prompt_name}.csv')
        outfile.parent.mkdir(parents=True, exist_ok=True)
        cfg_file = f'/home/jovyan/rmt/babilong/babilong_evals/{model_name}/{task}_{split_name}_{prompt_name}.json'
        json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)

        df = pd.DataFrame({'target': [], 'output': [], 'question': []})

        for sample in tqdm(task_data):
            target = sample['target']
            context = sample['input']
            question = sample['question']

            input_text = get_formatted_input(context, question, prompt_cfg['examples'],
                                             prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                                             template=TEMPLATE)

            input = tokenizer(input_text, return_tensors="pt", add_special_tokens=True).to(model.device)
            sample_length = input['input_ids'].shape[1]
            with torch.no_grad():
                output = model.generate(**input, max_length=sample_length+25, **generate_kwargs)
            output = output[0][input['input_ids'].shape[1]:]
            output = tokenizer.decode(output, skip_special_tokens=True).strip()

            df.loc[len(df)] = [target, output, question]
            df.to_csv(outfile)
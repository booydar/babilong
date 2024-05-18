import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from tqdm.auto import tqdm
import pandas as pd
import json
from pathlib import Path
import requests

from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input


def main(results_folder, model_name, tasks, split_names, dataset_name, use_chat_template, api_url,
         use_instruction, use_examples, use_post_prompt):

    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if not api_url:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                     device_map='auto', torch_dtype=dtype,
                                                     attn_implementation='flash_attention_2')
        model = model.eval()

    generate_kwargs = {
        'num_beams': 1,
        'do_sample': False,
        'temperature': None,
        'top_p': None,
        'top_k': None,
    }

    if tokenizer.pad_token_id is None:
        generate_kwargs['pad_token_id'] = tokenizer.eos_token_id

    print(f'prompt template:\n{DEFAULT_TEMPLATE}')

    for task in tqdm(tasks, desc='tasks'):
        prompt_cfg = {
            'instruction': DEFAULT_PROMPTS[task]['instruction'] if use_instruction else '',
            'examples': DEFAULT_PROMPTS[task]['examples'] if use_examples else '',
            'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else '',
            'template': DEFAULT_TEMPLATE,
        }
        prompt_name = [f'{k}_no' if len(prompt_cfg[k]) == 0 else f'{k}_yes' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)

        for split_name in tqdm(split_names, desc='lengths'):
            data = datasets.load_dataset(dataset_name, split_name)
            task_data = data[task]

            # Prepare files with predictions, prompt, and generation configurations
            outfile = Path(f'{results_folder}/{model_name}/{task}_{split_name}_{prompt_name}.csv')
            outfile.parent.mkdir(parents=True, exist_ok=True)
            cfg_file = f'./{results_folder}/{model_name}/{task}_{split_name}_{prompt_name}.json'
            json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)

            df = pd.DataFrame({'target': [], 'output': []})

            for sample in tqdm(task_data, desc=f'task: {task} length: {split_name}'):
                target = sample['target']
                context = sample['input']
                question = sample['question']

                input_text = get_formatted_input(context, question, prompt_cfg['examples'],
                                                 prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                                                 template=DEFAULT_TEMPLATE)

                if api_url:
                    # model is running via llamacpp's serve command
                    headers = {'Content-Type': 'application/json'}
                    if generate_kwargs['temperature'] is None:
                        generate_kwargs['temperature'] = 0.0

                    if use_chat_template:
                        input_text = [{'role': 'user', 'content': input_text}]
                        model_inputs = tokenizer.apply_chat_template(input_text, tokenize=True,
                                                                     add_generation_prompt=True)
                    else:
                        model_inputs = tokenizer.encode(input_text, add_special_tokens=True)

                    request_data = {'prompt': model_inputs, 'temperature': generate_kwargs['temperature']}
                    response = requests.post(api_url, headers=headers, json=request_data).json()
                    output = response['content'].strip()
                else:
                    if use_chat_template:
                        input_text = [{'role': 'user', 'content': input_text}]
                        model_inputs = tokenizer.apply_chat_template(input_text, return_tensors='pt').to(model.device)
                        model_inputs = {'input_ids': model_inputs}
                    else:
                        model_inputs = tokenizer(input_text, return_tensors='pt',
                                                 add_special_tokens=True).to(model.device)

                    sample_length = model_inputs['input_ids'].shape[1]
                    with torch.no_grad():
                        output = model.generate(**model_inputs, max_new_tokens=15, **generate_kwargs)
                    output = output[0][sample_length:]
                    output = tokenizer.decode(output, skip_special_tokens=True).strip()

                df.loc[len(df)] = [target, output]
                df.to_csv(outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--results_folder', type=str, required=True, default='./babilong_evals',
                        help='Folder to store results')
    parser.add_argument('--dataset_name', type=str, required=True, default='booydar/babilong-1k-samples',
                        help='dataset name from huggingface')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--tasks', type=str, nargs='+', required=True, help='List of tasks to evaluate: qa1 qa2 ...')
    parser.add_argument('--lengths', type=str, nargs='+', required=True, help='List of lengths to evaluate: 0k 1k ...')
    parser.add_argument('--use_chat_template', action='store_true', help='Use tokenizer chat template')
    parser.add_argument('--use_instruction', action='store_true', help='Use instruction in prompt')
    parser.add_argument('--use_examples', action='store_true', help='Use examples in prompt')
    parser.add_argument('--use_post_prompt', action='store_true', help='Use post prompt text in prompt')
    parser.add_argument('--api_url', type=str, required=True, default='', help='llamacpp api endpoint')

    args = parser.parse_args()
    main(args.results_folder, args.model_name, args.tasks, args.lengths, args.dataset_name, args.use_chat_template,
         args.api_url, args.use_instruction, args.use_examples, args.use_post_prompt)

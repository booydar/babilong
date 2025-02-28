import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from tqdm.auto import tqdm
import pandas as pd
import json
from pathlib import Path
import requests
from safetensors.torch import save_file, load_file
from typing import List
import sys
sys.path.append('..')
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
from modeling_rmt.language_modeling import RecurrentWrapper, MemoryCell
from modeling_amt.language_modeling import AssociativeRecurrentWrapper, AssociativeMemoryCell
from peft import get_peft_model, LoraConfig, TaskType


def main(
    results_folder: str, model_name: str, tasks: List[str], split_names: List[str], dataset_name: str,
    use_chat_template: bool, api_url: str, use_instruction: bool, use_examples: bool, use_post_prompt: bool,
    model_cpt: str, mem_size: int, segment_size: int, max_n_segments: int, lora_r: int,
    lora_alpha: int, lora_dropout: float, use_peft: float, d_mem, layers_attr, no_correction, wrap_pos, use_quest_first_template,
    use_double_question, use_answer_fill, add_question_prompt, segment_alignment, attend_to_previous_input, model_title
) -> None:
    """
    Main function to get model predictions on babilong and save them.

    Args:
        results_folder (str): Folder to store results.
        model_name (str): Name of the model to use.
        tasks (List[str]): List of tasks to evaluate.
        split_names (List[str]): List of lengths to evaluate.
        dataset_name (str): Dataset name from Hugging Face.
        use_chat_template (bool): Flag to use the tokenizer chat template.
        api_url (str): API endpoint for llama.cpp.
        use_instruction (bool): Flag to use instruction in prompt.
        use_examples (bool): Flag to use examples in prompt.
        use_post_prompt (bool): Flag to use post_prompt text in prompt.
    """

    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if model_name == "meta-llama/Llama-3.2-1B":
        # hotfix - load 8b instruct tokenizer and reuse chat template from instruct tokenizer
        it_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)
        tokenizer.chat_template = it_tokenizer.chat_template
    if not api_url:
        if len(model_cpt) != 0:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                         device_map='cpu', torch_dtype=dtype,
                                                         attn_implementation='flash_attention_2')
            if use_peft:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, 
                    inference_mode=False, 
                    r=lora_r, 
                    lora_alpha=lora_alpha, 
                    lora_dropout=lora_dropout,
                    )
                model = get_peft_model(model, peft_config)
            device = "cpu"
            if d_mem is not None:
                mem_cell_args = dict(
                    base_model=model,
                    num_mem_tokens=mem_size,
                )
                # additional parameters for ARMT model
                mem_cell_args['d_mem'] = d_mem
                mem_cell_args['wrap_pos'] = wrap_pos
                mem_cell_args['correction'] = not(no_correction)
                mem_cell_args['use_lora'] = use_peft
                if layers_attr is not None:
                    mem_cell_args['layers_attr'] = layers_attr
                # if attend_to_previous_input is not None:
                #     mem_cell_args['attend_to_previous_input'] = bool(attend_to_previous_input)
                cell = AssociativeMemoryCell(**mem_cell_args)
                model = AssociativeRecurrentWrapper(cell,
                                                    segment_size=segment_size-2*mem_size,
                                                    max_n_segments=max_n_segments,
                                                    segment_alignment=segment_alignment,
                                                    # attend_to_previous_input=attend_to_previous_input,
                ).to(device)
            else:
                cell = MemoryCell(model, num_mem_tokens=mem_size)
                model = RecurrentWrapper(cell,
                                         segment_size=segment_size-2*mem_size,
                                         max_n_segments=max_n_segments,
                ).to(device)
            #model = model.to(dtype)
            try:
                cpt = torch.load(model_cpt, map_location=device)    
                model.load_state_dict(cpt, strict=True)
            except:
                # if the model saved in safetensors
                from safetensors.torch import load_model
                load_model(model, model_cpt, device=device)
                #model.load_state_dict(model_cpt, map_location=device)
            
            device = 'cuda:0'
            model.to(device)
            model.to(dtype)
            model.name_or_path = "custom_rmt" # workaround
            model.device = device
        # load the model locally if llamacpp API is not used
        else:
            try:
                print('trying to load model with flash attention 2...')
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                             device_map='auto', torch_dtype=dtype,
                                                             attn_implementation='flash_attention_2')
            except ValueError as e:
                print(e)
                print('trying to load model without flash attention 2...')
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                             device_map='auto', torch_dtype=dtype)

        model = model.eval()
        # print(model.memory_cell.model.base_model.model.model.embed_tokens.weight[0:1, 0:20])
        # print(model.memory_cell.model.base_model.model.model.layers[0].layer.self_attn.q_proj.lora_A.default.weight[0:1, 0:20])
        # print(model.memory_cell.model.base_model.model.model.layers[0].W_mq.weight[0:1, 0:20])
        # 1/0

    # define generation parameters
    generate_kwargs = {
        'max_new_tokens': 20 if "RMT-Llama-3.2-1B-Instruct-8x1024-mem16-lora-babilong-qa1-5_ct-v3.1" not in model_cpt else 30,
        'max_length': None,
        'num_beams': 1,
        'do_sample': False,
        'temperature': None,
        'top_p': None,
        'top_k': None,
        'pad_token_id': tokenizer.pad_token_id
    }

    if generate_kwargs['pad_token_id'] is None:
        generate_kwargs['pad_token_id'] = tokenizer.eos_token_id

    SYSTEM_TEMPLATE = '{instruction}\n\n{examples}\n\n{post_prompt}'
    USER_TEMPLATE = '<context>\n{context}\n</context>\n\nQuestion: {question}'
    DEFAULT_TEMPLATE = f'{SYSTEM_TEMPLATE}\n\n{USER_TEMPLATE}'
    template_to_use = DEFAULT_TEMPLATE
    # if use_quest_first_template:
    #     SYSTEM_TEMPLATE = '{instruction}\n\n{examples}\n\n{post_prompt}'
    #     USER_TEMPLATE = 'Question: {question}\n\n<context>\n{context}\n</context>'
    #     if use_double_question:
    #         USER_TEMPLATE = 'Question: {question}\n\n<context>\n{context}\n</context>\n\nQuestion: {question}'
    #     template_to_use = f'{SYSTEM_TEMPLATE}\n\n{USER_TEMPLATE}'
    # if use_answer_fill:
    #     template_to_use += '\n\nAnswer: '
    print(f'prompt template:\n{template_to_use}')

    for task in tqdm(tasks, desc='tasks'):
        # configure the prompt
        prompt_cfg = {
            'instruction': DEFAULT_PROMPTS[task]['instruction'] if use_instruction else '',
            'examples': DEFAULT_PROMPTS[task]['examples'] if use_examples else '',
            'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else '',
            'template': template_to_use,
            'chat_template': use_chat_template,
        }
        prompt_name = [f'{k}_yes' if prompt_cfg[k] else f'{k}_no' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)
        if use_quest_first_template:
            prompt_name += "_quest_first_yes"
            if use_double_question:
                prompt_name += "_quest_first_2_yes"
        if use_answer_fill:
            prompt_name += "_answer_fill_yes"

        for split_name in tqdm(split_names, desc='lengths'):
            # load dataset
            data = datasets.load_dataset(dataset_name, split_name)
            task_data = data[task]

            # Prepare files with predictions, prompt, and generation configurations
            if not model_title:
                outfile = Path(f'{results_folder}/{model_name.replace("../", "")}/{model_cpt.replace("../", "")}/{task}_{split_name}_{prompt_name}.csv')
                outfile.parent.mkdir(parents=True, exist_ok=True)
                cfg_file = f'./{results_folder}/{model_name.replace("../", "")}/{model_cpt.replace("../", "")}/{task}_{split_name}_{prompt_name}.json'
            else:
                outfile = Path(f'{results_folder}/{model_title.replace("../", "")}/{task}_{split_name}_{prompt_name}.csv')
                outfile.parent.mkdir(parents=True, exist_ok=True)
                cfg_file = f'{results_folder}/{model_title.replace("../", "")}/{task}_{split_name}_{prompt_name}.json'

            json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)

            df = pd.DataFrame({'target': [], 'output': [], 'question': []})

            for sample in tqdm(task_data, desc=f'task: {task} length: {split_name}'):
                target = sample['target']
                context = sample['input']
                question = sample['question']

                # format input text
                input_text = get_formatted_input(context, question + add_question_prompt, prompt_cfg['examples'],
                                                 prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                                                 template=prompt_cfg['template'])

                # with open(cfg_file[:-5] + '.txt') as f:
                # with open('tmp.txt', 'a') as f:
                #     f.write('\n' + input_text)
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
                    # generate output using local model
                    if model.name_or_path in ['THUDM/chatglm3-6b-128k', 'THUDM/LongAlign-6B-64k-base', 'THUDM/LongAlign-6B-64k']:
                        # have to add special code to run chatglm as tokenizer.chat_template tokenization is not
                        # the same as in model.chat (recommended in https://huggingface.co/THUDM/chatglm3-6b-128k)
                        with torch.no_grad():
                            output, _ = model.chat(tokenizer, input_text, history=[], **generate_kwargs)
                    else:
                        
                        if use_chat_template:
                            input_text = [{'role': 'user', 'content': input_text}]
                            model_inputs = tokenizer.apply_chat_template(input_text, add_generation_prompt=True,
                                                                         return_tensors='pt', return_dict=model.name_or_path=="custom_rmt").to(model.device)
                            if model.name_or_path != "custom_rmt":
                                model_inputs = {'input_ids': model_inputs}
                        else:
                            model_inputs = tokenizer(input_text, return_tensors='pt',
                                                     add_special_tokens=True).to(model.device)

                        sample_length = model_inputs['input_ids'].shape[1]
                        # print(model_inputs['input_ids'][0][:1000])
                        # with open('tmp.txt', 'w') as f:
                        #     f.write(str(model_inputs['input_ids'][0].cpu().tolist()))
                        # 1/0
                        with torch.no_grad():
                            output = model.generate(**model_inputs, **generate_kwargs)
                            # we need to reset memory states between samples for activation-beacon models
                            if 'activation-beacon' in model.name_or_path and hasattr(model, 'memory'):
                                model.memory.reset()
                        if model.name_or_path != "custom_rmt":
                            output = output[0][sample_length:]
                        else:
                            output = output[0]
                        output = tokenizer.decode(output, skip_special_tokens=True).strip()

                df.loc[len(df)] = [target, output, question]
                # write results to csv file
                df.to_csv(outfile, escapechar='\\')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--results_folder', type=str, required=True, default='./babilong_evals',
                        help='Folder to store results')
    parser.add_argument('--dataset_name', type=str, required=True, default='RMT-team/babilong-1k-samples',
                        help='dataset name from huggingface')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--model_title', type=str, required=False, help='Name of the model to save')
    parser.add_argument('--model_cpt', type=str, default="", help='Name of the model checkpoint in case of RMT')
    parser.add_argument('--tasks', type=str, nargs='+', required=True, help='List of tasks to evaluate: qa1 qa2 ...')
    parser.add_argument('--lengths', type=str, nargs='+', required=True, help='List of lengths to evaluate: 0k 1k ...')
    parser.add_argument('--use_chat_template', action='store_true', help='Use tokenizer chat template')
    parser.add_argument('--use_instruction', action='store_true', help='Use instruction in prompt')
    parser.add_argument('--use_examples', action='store_true', help='Use examples in prompt')
    parser.add_argument('--use_post_prompt', action='store_true', help='Use post prompt text in prompt')
    parser.add_argument('--api_url', type=str, required=True, default='', help='llamacpp api endpoint')
    # args for RMT model
    parser.add_argument('--mem_size', type=int, required=False, default=5, help='Memory size for RMT')
    parser.add_argument('--segment_size', type=int, required=False, default=4096, help='Segment size for RMT')
    parser.add_argument('--max_n_segments', type=int, required=False, default=8, help='Max number of segments for RMT')
    # args for ARMT model
    parser.add_argument('--d_mem', type=int, default=None, help='number of rows in associative matrix')
    parser.add_argument('--layers_attr', type=str, default=None, help='attribute of model, which contains layers')
    parser.add_argument('--no_correction', action='store_true', default=False,
                        help='ARMT shmidhuber correction for rewriting')
    parser.add_argument('--wrap_pos', action='store_true', default=False,
                        help='Wrap positional encoding for memory tokens (default: False)')
    # LoRA params
    parser.add_argument('--use_peft', type=int, required=False, default=1, help='Use PEFT in model', )
    parser.add_argument('--lora_r', type=int, required=False, default=8, help='LoRA r')
    parser.add_argument('--lora_alpha', type=int, required=False, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, required=False, default=0.1, help='LoRA dropout')
    parser.add_argument('--use_quest_first_template', action='store_true', help='Use template with question before context (but after all other parts)')
    parser.add_argument('--use_double_question', action='store_true', help='Use template with question before and after context (but after all other parts)')
    parser.add_argument('--use_answer_fill', action='store_true', help='Use Answer: at the end of prompt')
    # new eval params
    parser.add_argument('--add_question_prompt', type=str, default="", help='Additional prompt, injected after question')
    parser.add_argument('--segment_alignment', type=str, default="left", help='Segment alignment for ARMT')
    parser.add_argument('--attend_to_previous_input', type=int, default=0, help='Attend to prev segment')

    args = parser.parse_args()
    main(args.results_folder, args.model_name, args.tasks, args.lengths, args.dataset_name, args.use_chat_template,
         args.api_url, args.use_instruction, args.use_examples, args.use_post_prompt, args.model_cpt,
         args.mem_size, args.segment_size, args.max_n_segments, args.lora_r, args.lora_alpha, args.lora_dropout, args.use_peft,
         args.d_mem, args.layers_attr, args.no_correction, args.wrap_pos, args.use_quest_first_template, args.use_double_question,
         args.use_answer_fill, args.add_question_prompt, args.segment_alignment, args.attend_to_previous_input, args.model_title)
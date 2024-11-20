import numpy as np
import json
import torch
import os
import datasets

import sys
sys.path.append('..')
from babilong.babilong_utils import TaskDataset, SentenceSampler, NoiseInjectionDataset
from transformers import AutoTokenizer

# qa1_single-supporting-fact qa2_two-supporting-facts qa3_three-supporting-facts qa4_two-arg-relations qa5_three-arg-relations qa6_yes-no-questions qa7_counting qa8_lists-sets qa9_simple-negation qa10_indefinite-knowledge
# qa11_basic-coreference qa12_conjunction qa13_compound-coreference qa14_time-reasoning qa15_basic-deduction qa16_basic-induction qa17_positional-reasoning qa18_size-reasoning qa19_path-finding qa20_agents-motivations

out_folder = "./generated_tasks"
task_folder = "./tasks_1-20_v1-2/en-10k/"
number_of_samples = 100


os.makedirs(out_folder, exist_ok=True)

if __name__ == "__main__":
    import sys
    tasks = sys.argv[1].split(' ')
    print(tasks)

    message_lengths = [0, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1_000_000]
    names = ['0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k', '512k', '1M']

    message_lengths = message_lengths

    message_lengths = [ml - 300 for ml in message_lengths] # take prompt length into account

    os.makedirs('tasks', exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    for task in tasks:
        print('processing', task)
        # a placeholder for all samples for the current task
        llm_tasks = dict()
        subfolder = os.path.join(out_folder, task.split('_')[0])
        os.makedirs(subfolder, exist_ok=True)

        task_path = os.path.join(task_folder, task + '_train.txt')
        
        for len_name, message_length in zip(names, message_lengths):
            print('message length', len_name, message_length)
            noise_dataset = datasets.load_dataset("pg19")['test']

            if message_length > 0:
                max_n_facts = message_length // 8
                task_dataset_test = TaskDataset(task_path, max_n_facts=max_n_facts)
            else:
                task_dataset_test = TaskDataset(task_path)
            

            noise_sampler_test = SentenceSampler(noise_dataset, tokenizer=tokenizer, shuffle=True, random_seed=None)
            dataset_test = NoiseInjectionDataset(task_dataset=task_dataset_test,
                                                    noise_sampler=noise_sampler_test,
                                                    tokenizer=tokenizer,
                                                    sample_size=message_length)

            # get number_of_samples random indices
            inds = list(range(len(dataset_test)))
            np.random.shuffle(inds)
            inds = inds[:number_of_samples]

            # prepare samples for LLM evaluation
            samples = [dataset_test[i] for i in inds]

            questions = [sample['question'] for sample in samples]
            input_tokens = [torch.tensor(sample['input_tokens']) for sample in samples]
            target_tokens = [torch.tensor(sample['target_tokens']) for sample in samples]

            inputs = tokenizer.batch_decode(input_tokens, add_special_tokens=False)
            targets = tokenizer.batch_decode(target_tokens, add_special_tokens=False)

            llm_tasks[len_name] = [{'input': i.strip(), 'question': q, 'target': t} for (i, q, t) in zip(inputs, questions, targets)]
            
            
            json_path = os.path.join(subfolder, f"{len_name}.json")
            print(f"Writing", json_path)
            with open(json_path, 'w') as f:
                json.dump(llm_tasks[len_name], f)
import json
import logging
import os
import math
from pathlib import Path
import pickle
import numpy as np

# os.environ["NCCL_DEBUG_SUBSYS"]="WARN"
# os.environ["NCCL_IB_SL"]=""
# os.environ["NCCL_P2P_LEVEL"]=""
# os.environ["NCCL_NET_GDR_LEVEL"]=""
# os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = os.environ["LOCAL_RANK"]
# os.environ["NCCL_IB_GID_INDEX"]="3"

# from dotenv import load_dotenv
import torch
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader

# from lm_experiments_tools.trainer import TrainerArgs, Trainer
from transformers.trainer import Trainer, TrainingArguments

from torch.nn.utils.rnn import pad_sequence
from babilong_utils import TaskDataset, SentenceSampler, NoiseInjectionDataset

import accelerate

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

# import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # , BitsAndBytesConfig  # noqa: E402

from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
# torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
# torch.cuda.set_device(hvd.local_rank())

### DEBUG MODE
# os.environ["TORCH_DISTRIBUTED_DEBUG"]="INFO"
# os.environ["NCCL_DEBUG"]="INFO"
# os.environ["NCCL_DEBUG_SUBSYS"]="ALL"

### Additional debug parameters
# os.environ['NCCL_BLOCKING_WAIT'] = '1'
# os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# os.environ['HF_HOME'] = "/home/jovyan/.cache/huggingface/transformers"
# os.environ['TRANSFORMERS_OFFLINE'] = "1"
# os.environ['HF_DATASETS_OFFLINE'] = "1"


from lm_experiments_tools.utils import get_distributed_rank

from lm_experiments_tools.trainer import TrainerArgs
parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--noise_dataset_split', type=str, help="Task name", default=None)
parser.add_argument('--babi_path', type=str, help="path to babi folder", default="data/tasks_1-20_v1-2/en-10k")
# parser.add_argument('--task_name', type=str, help="Task name, wikitext, ...")
# parser.add_argument('--task_split_name', type=str, help="Split name if applicable, e.g. wikitext-2-raw-v1")
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
parser.add_argument('--data_n_workers', type=int, default=1, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--memory_cell_cls', type=str, default=None, help='cell class for RMT')
parser.add_argument('--recurrent_wrapper_cls', type=str, default=None, help='recurrent wrapper class for RMT')
parser.add_argument('--model_cpt', type=str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')
parser.add_argument('--checkpoint', type=str, default=None, help='hf trainer checkpoint path')

# Babilong parameters
parser.add_argument('--sample_size', type=int, default=None, help='max number of tokens in sample')
parser.add_argument('--max_n_facts', type=int, default=None, help='drop samples with higher number of facts')
parser.add_argument('--task_start_pct', type=float, default=None, help='left border of facts in sample, between 0 and 1')
parser.add_argument('--task_end_pct', type=float, default=None, help='right border of facts in sample, between task_start_pct and 1')



# RMT args 
parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='Randomly choose segment number from 1 to max_n_segments')
parser.add_argument('--sampling_prob', type=float, default=1, help='Probability of sampling other number of segments')
parser.add_argument('--sum_loss', action='store_true', default=False,
                    help='with this flag task loss from all segments is summed')
parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
parser.add_argument('--segment_alignment', type=str, help='way of aligning segments, one of right, left, center', default=None)
parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular',
                    choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
parser.add_argument('--memory_forward_func', type=str, help='path to memory forward funсtion script', default=None)
parser.add_argument('--memory_layers', type=str, help='memory-augmented layer inds or "all" for all layers', default=None)
parser.add_argument('--share_memory_layers', action='store_true', help='share weights of memory layers', default=False)
parser.add_argument('--reconstruction_loss_coef', type=float, default=None,
                    help='reconstuction loss ratio in total loss')
parser.add_argument('--retain_graph', action='store_true', help='Retain computation graph during backward pass', default=False)
parser.add_argument('--use_truncated_backward', action='store_true', default=False,
                    help='whether to use RMT truncated bptt method in backward')
# parser.add_argument('--k1', type=int, default=-1, help='(not implemented) If not -1, gradient update is done each k1 segments')
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')

parser.add_argument('--base_model_forward', type=str, default=None, help='custom forward function for backbone model')


# tokenizer
# todo: add wordpiece tokenizers support?
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')

# LoRA args
parser.add_argument('--use_lora', action='store_true', default=False, help='')
parser.add_argument('--lora_attn_dim', type=int, default=8, help='')
parser.add_argument('--lora_attn_alpha', type=int, default=32, help='')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='')
parser.add_argument('--layers_pattern', type=str, default=None, help='')
parser.add_argument('--int8', action='store_true', default=False, help='')
parser.add_argument('--int4', action='store_true', default=False, help='')

# parser.add_argument('--use_flash_attention', action='store_true', default=False, help='')

# # Parallel Adapter args
# parser.add_argument('--use_adapter', action='store_true', default=False, help='')
# parser.add_argument('--adapter_bottleneck_dim', type=int, default=512, help='')
# parser.add_argument('--adapter_dropout', type=float, default=0.1, help='')
# parser.add_argument('--adapter_scale', type=float, default=4.0, help='')

# Dataset args
parser.add_argument('--pile_subset_names', type=str, default=None, help='use only these subsets of The PILE, separated by ;')
parser.add_argument('--min_tokens_in_document', type=int, default=None, help='do not use documents shorter than this value')
parser.add_argument('--max_tokens_in_document', type=int, default=None, help='do not use documents longer than this value')


from dataclasses import dataclass, field

@dataclass
class RMTArguments:
    # task_name: str
    from_pretrained: str
    model_cls: str
    model_type: str
    memory_cell_cls: str
    recurrent_wrapper_cls: str
    sample_size: int
    segment_size: int
    task_dataset: str
    noise_dataset: str
    babi_path: str
    batch_size: int
    # block_size: int
    # history_size: int
    num_mem_tokens: int
    vary_n_segments: bool
    max_n_segments: int
    k2: int = -1
    max_n_facts: int = None
    task_start_pct: int = None
    task_end_pct: int = None
    data_n_workers: int = 1
    mixed_length_ratio: float = None
    lr: float = None
    validate_only: bool = False
    segment_alignment: str = None
    min_input_size: int = None
    use_lora: bool = False
    backbone_cpt: str = None
    model_cpt: str = None
    freeze_model_weights: bool = False
    checkpoint: str = None  
    pile_subset_names: str = None
    min_tokens_in_document: int = None
    max_tokens_in_document: int = None


if __name__ == '__main__':
    parser = HfArgumentParser((TrainingArguments, RMTArguments))
    training_args, args = parser.parse_args_into_dataclasses()

    if not args.from_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    # Prepare datasets

    logger.info(f'preparing dataset for {args.task_dataset}')
    try:
        # noise_dataset = datasets.load_dataset(args.noise_dataset, args.noise_dataset_split)
        noise_dataset = datasets.load_dataset(args.noise_dataset)
        noise_dataset_train = noise_dataset['train']
        noise_dataset_test = noise_dataset['test']
    except ConnectionError:
        noise_dataset_train = datasets.Dataset.from_file('/home/jovyan/.cache/huggingface/datasets/pg19/default/0.1.0/64837d6fce7251337df051ca74e9a5435d1c9cb7f3033ba257826e44d338f83c/pg19-train.arrow')
        noise_dataset_test = datasets.Dataset.from_file('/home/jovyan/.cache/huggingface/datasets/pg19/default/0.1.0/64837d6fce7251337df051ca74e9a5435d1c9cb7f3033ba257826e44d338f83c/pg19-test.arrow')
    
    # task dataset 
    train_path = os.path.join(args.babi_path, f"{args.task_dataset}_train.txt")
    test_path = os.path.join(args.babi_path, f"{args.task_dataset}_test.txt")

    task_dataset_train = TaskDataset(train_path, max_n_facts=args.max_n_facts)
    task_dataset_test = TaskDataset(test_path, max_n_facts=args.max_n_facts)

    # background text
    qa_margin = 20          # leave space for questions and answers
    if args.vary_n_segments:  # choose sample sizes according to each number of segments up to args.max_n_segments
        # train_sample_size = [int(args.sample_size / i) for i in range(1, args.max_n_segments + 1)]
        train_sample_size = [int(args.segment_size * i) for i in range(1, args.max_n_segments)] + [args.sample_size]
        train_sample_size = [s - qa_margin for s in train_sample_size]
        logger.info(f'Will be choosing sample size randomly from {train_sample_size} for training')
    else:
        sample_size = args.sample_size - qa_margin
        train_sample_size = args.sample_size - qa_margin
    test_sample_size = args.sample_size - qa_margin
    max_sentence_len = None
    if (args.task_start_pct is not None) and (args.task_end_pct is not None):
        # do not sample sentences longer than task position range * 0.5
        max_sentence_len = int((args.task_end_pct - args.task_start_pct) * 0.5 * args.sample_size)
        
    noise_sampler_train = SentenceSampler(noise_dataset_train, tokenizer=tokenizer, max_sentence_len=max_sentence_len, shuffle=True, random_seed=None)
    noise_sampler_test = SentenceSampler(noise_dataset_test, tokenizer=tokenizer, max_sentence_len=max_sentence_len, shuffle=True, random_seed=42)

    train_dataset = NoiseInjectionDataset(task_dataset=task_dataset_train,
                                            noise_sampler=noise_sampler_train,
                                            tokenizer=tokenizer,
                                            sample_size=train_sample_size,
                                            mixed_length_ratio=args.mixed_length_ratio,
                                            task_start_pct=args.task_start_pct,
                                            task_end_pct=args.task_end_pct
                                            )

    test_dataset = NoiseInjectionDataset(task_dataset=task_dataset_test,
                                            noise_sampler=noise_sampler_test,
                                            tokenizer=tokenizer,
                                            sample_size=test_sample_size,
                                            mixed_length_ratio=args.mixed_length_ratio,
                                            task_start_pct=args.task_start_pct,
                                            task_end_pct=args.task_end_pct
                                            )
    
    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_token = tokenizer.encode('GEN')[0]
    eos_token = tokenizer.eos_token_id

    def collate_fn(batch):
        # print(batch)
        targets = [torch.tensor(b['target_tokens']) for b in batch]
        input_ids = [torch.tensor(b['input_tokens'] + b['question_tokens'] + [gen_token] + b['target_tokens'] + [eos_token]) for b in batch]
        gen_inputs = [torch.tensor(b['input_tokens'] + b['question_tokens'] + [gen_token]) for b in batch]

        attention_mask = [torch.ones_like(b, dtype=int) for b in input_ids]
        labels_mask = [torch.zeros_like(b, dtype=bool) for b in input_ids]
        for m, t in zip(labels_mask, targets):
            m[-len(t) - 2:] = True

        # for tensors in [input_ids, gen_inputs, attention_mask, labels_mask]:
        #     tensors = [t.flip(dims=[0]) for t in tensors]
        
        input_ids = pad_sequence(input_ids, padding_value=id_pad_value, batch_first=True)
        gen_inputs = pad_sequence(gen_inputs, padding_value=id_pad_value, batch_first=True)
        attention_mask = pad_sequence(attention_mask, padding_value=0, batch_first=True)
        labels_mask = pad_sequence(labels_mask, padding_value=0, batch_first=True)
        # for tensors in [input_ids, gen_inputs, attention_mask, labels_mask]:
        #     tensors = tensors.flip(dims=[1])

        collated = {}
        collated['input_ids'] = collated['labels'] = input_ids
        # collated['input_ids_generate'] = gen_inputs
        collated['labels_mask'] = labels_mask
        collated['attention_mask'] = attention_mask.bool()
        # collated['attention_mask_generate'] = (gen_inputs != id_pad_value).bool()
        # collated['target_text'] = [b['answer'] for b in batch]

        # print(collated['input_ids'].shape)
        return collated

    # kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers, 'collate_fn': collate_fn}
    # per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    # train_sampler = DistributedSampler(train_dataset, rank=accelerator.process_index,
    #                                    num_replicas=accelerator.num_processes, shuffle=True, drop_last=True,
    #                                    seed=args.seed)
    # test_sampler = DistributedSampler(test_dataset, rank=accelerator.process_index,
    #                                   num_replicas=accelerator.num_processes, drop_last=False, shuffle=False)
    # train_dataloader = DataLoader(batch_size=per_worker_batch_size, dataset=train_dataset, sampler=train_sampler,
    #                               **kwargs)
    # test_dataloader = DataLoader(batch_size=per_worker_batch_size, dataset=test_dataset, sampler=test_sampler, **kwargs)

    # if args.valid_interval is None:
    #     args.valid_interval = args.log_interval

    # define model
    model_cls = get_cls_by_name(args.model_cls)

    logger.info(f'Using model class: {model_cls}')

    logger.info(f'Loading pretrained model: {args.from_pretrained}')
    model = model_cls.from_pretrained(args.from_pretrained, 
                                      use_cache=False,
                                      trust_remote_code=True,
                                        use_flash_attention_2=True, torch_dtype=torch.bfloat16)
                                              

    # if args.use_lora:
    #     peft_config = LoraConfig(
    #         task_type=TaskType.CAUSAL_LM, 
    #         inference_mode=False, 
    #         r=args.lora_attn_dim, 
    #         lora_alpha=args.lora_attn_alpha, 
    #         lora_dropout=args.lora_dropout
    #         )
    #     model = get_peft_model(model, peft_config)
    #     logger.info(f'Added LoRA, trainable parameters with LoRA only:')
    #     model.print_trainable_parameters()
    

    ## load cpt of backbone model
    if args.backbone_cpt:
        model = model_cls.from_pretrained(args.backbone_cpt,
                                config=model.config,
                                torch_dtype=torch.bfloat16,
                                use_flash_attention_2=True,
                                ignore_mismatched_sizes=True)
        model.config.pretraining_tp = 1
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')

    # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
        recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
        logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')
        
        cell = memory_cell_cls(model, args.num_mem_tokens)
        if args.segment_alignment not in {None, 'left'}:
            logger.info(f"Using custom segment alignment: {args.segment_alignment}")
        model = recurrent_wrapper_cls(cell, 
                                      segment_size=args.segment_size,
                                      max_n_segments=args.max_n_segments, 
                                      vary_n_segments=args.vary_n_segments,
                                      segment_alignment=args.segment_alignment,
                                      k2=args.k2,
        )
                                    

        ## load cpt of rmt
        if args.model_cpt:
            model_cpt = os.path.join(args.model_cpt, "pytorch_model.bin")
            cpt = torch.load(model_cpt, map_location='cpu')
            model.load_state_dict(cpt, strict=False)
            logger.info(f'Loaded RMT state dict from: {args.model_cpt}')

    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            p.requires_grad = False
            if '.memory' in n:
                p.requires_grad = True
            if 'lora' in n or 'adapter' in n:
                p.requires_grad = True
        logger.info(f'Frozen moodel weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')

    # # fix the not-contiguous error
    # def make_contiguous(module):
    #     with torch.no_grad():
    #         for param in module.parameters():
    #             param.set_(param.contiguous())

    training_args.gradient_checkpointing = False
    training_args.remove_unused_columns=False
    training_args.report_to = ['tensorboard']
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        # optimizers=[optimizer, None],
    ) 
    logger.info(f'Start training!')
    trainer.train(resume_from_checkpoint=args.checkpoint)
    trainer.evaluate()
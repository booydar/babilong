import logging
import os
from pathlib import Path
from itertools import chain
from datetime import timedelta
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, set_seed
from lm_experiments_tools.dataset_preprocessing import load_and_preprocess_task
from lm_experiments_tools.instruction_utils import mask_non_completion, mask_non_completion_multi

from torch.nn.utils.rnn import pad_sequence

import accelerate
from peft import get_peft_model, LoraConfig, TaskType

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

# import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402

from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
# torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
# torch.cuda.set_device(hvd.local_rank())

parser = HfArgumentParser(SFTConfig)
parser.add_argument('--task_name', type=str, help="Task name, wikitext, ...")
parser.add_argument('--append_concat_token', action='store_true', default=False, help="Append concat token during packing")
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
parser.add_argument('--sample_size', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
parser.add_argument('--sliding_window', action='store_true', help='use slinding window attentinon mask, '
                    'eval on last segment only', default=False)
parser.add_argument('--attend_to_previous_input', action='store_true', help='attend to the previous segment', default=False)
parser.add_argument('--use_length_filtering', action='store_true', help='filter samples longer than train len', default=False)
parser.add_argument('--reduce_eval', type=float, default=None, help='part of eval to use')
parser.add_argument('--no_packing', action='store_true', help='disable packing, add padding', default=False)
parser.add_argument('--padding_side', type=str, help='set padding side', default=False)
parser.add_argument('--truncate_before', type=int, default=0, help='truncate input before defined ids, for debug only')
parser.add_argument('--truncate_only_train', action='store_true', help='set truncate input before defined ids only for train, for debug only', default=False)


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
parser.add_argument('--checkpoint', type=str, default=None, help='Full experiment checkpoint, used to resume training in SFTTrainer')

# Aydar # RMT args
parser.add_argument('--segment_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='Randomly choose segment number from 1 to max_n_segments')
parser.add_argument('--loss_from_last_seg_only', action='store_true', default=False, help='take loss from last segment only')
parser.add_argument('--no_loss_from_first_segment', action='store_true', default=False, help='turn off loss from first segment')
parser.add_argument('--sum_loss', action='store_true', default=False,
                    help='with this flag task loss from all segments is summed')
parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular',
                    choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
parser.add_argument('--memory_forward_func', type=str, help='path to memory forward funсtion script', default=None)
parser.add_argument('--memory_layers', type=str, help='memory-augmented layer inds or "all" for all layers', default=None)
parser.add_argument('--share_memory_layers', action='store_true', help='share weights of memory layers', default=False)
parser.add_argument('--reconstruction_loss_coef', type=float, default=None,
                    help='reconstuction loss ratio in total loss')
# parser.add_argument('--segment_ordering', type=str,help='????', default='regular',
#                     choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
parser.add_argument('--retain_graph', action='store_true', help='Retain computation graph during backward pass', default=False)
parser.add_argument('--use_truncated_backward', action='store_true', default=False,
                    help='whether to use RMT truncated bptt method in backward')
parser.add_argument('--k1', type=int, default=-1, help='(not implemented) If not -1, gradient update is done each k1 segments')
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')
parser.add_argument('--load_optimizer', type=int, default=1, help='load optimizer')
parser.add_argument('--tune_only_memory', action='store_true', default=False,
                    help='Stop training all model weights except memory layer memory')
parser.add_argument('--tune_only_armt', action='store_true', default=False,
                    help='Stop training all model weights except ARMT params')
parser.add_argument('--mask_non_completion', action='store_true', default=False,
                    help='Mask everything except completion in dataset')
# ARMT parameters
parser.add_argument('--d_mem', type=int, default=None, help='number of rows in associative matrix')
parser.add_argument('--layers_attr', type=str, default=None, help='attribute of model, which contains layers')
parser.add_argument('--no_correction', action='store_true', default=False,
                    help='ARMT shmidhuber correction for rewriting')
parser.add_argument('--wrap_pos', action='store_true', default=False,
                    help='Wrap positional encoding for memory tokens (default: False)')

# tokenizer
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')
parser.add_argument('--tokenizer_for_chat_template', type=str, default=None, help='path or name of pre-trained HF Tokenizer, from which CT are used')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')
parser.add_argument('--early_stopping_patience', type=int, default=-1,
                    help='Early stopping tolerance')

# LoRA args
parser.add_argument('--use_lora', action='store_true', default=False, help='')
parser.add_argument('--lora_attn_dim', type=int, default=8, help='')
parser.add_argument('--lora_attn_alpha', type=int, default=32, help='')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='')
parser.add_argument('--add_lora_to_armt', action='store_true', default=False, help='')

# Parallel Adapter args
parser.add_argument('--use_adapter', action='store_true', default=False, help='')
parser.add_argument('--adapter_bottleneck_dim', type=int, default=512, help='')
parser.add_argument('--adapter_dropout', type=float, default=0.1, help='')
parser.add_argument('--adapter_scale', type=float, default=4.0, help='')


if __name__ == '__main__':
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    set_seed(args.seed)

    # workaround with setting bigger tiomeout for NCCL (useful for big dataset, to avoid timeout at tokenization)
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[accelerate.InitProcessGroupKwargs(timeout=timedelta(seconds=20 * 1800))])
    from accelerate.logging import get_logger
    logger = get_logger('')

    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')

    if args.output_dir is None:
        logger.warning('output_dir is not set: config, logs and checkpoints will not be saved.')

    if not args.from_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    if args.tokenizer_for_chat_template is not None:
        it_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_for_chat_template, trust_remote_code=True)
        tokenizer.chat_template = it_tokenizer.chat_template
    if args.padding_side is not None:
        tokenizer.padding_side = args.padding_side
    # Prepare datasets
    logger.info(f'preparing dataset for {args.task_name}')
    with accelerator.main_process_first():
        tokenized_datasets = load_and_preprocess_task(args.task_name, tokenizer, args.sample_size,
                                                      args.max_n_segments, args.num_mem_tokens,
                                                      args.use_length_filtering, args.reduce_eval)
    block_size = args.segment_size
    # TODO: add a switch for ARMT
    if args.num_mem_tokens is not None:
        block_size -= 2 * args.num_mem_tokens
        if args.d_mem is not None:
            # cause for ARMT we have mem tokens only at the end
            block_size -= args.num_mem_tokens
    history_size = args.sample_size - block_size

    # Check grouping in chunks, compare with TRL packing
    # It's almost the same, but maybe we need an additional separator?
    def group_texts(examples, block_size, history_size=None):
        if args.mask_non_completion:
            if "llama" in args.from_pretrained.lower() or "llama" in args.tokenizer_for_chat_template.lower():
                response_template = "<|start_header_id|>assistant<|end_header_id|>"
                response_template_end_idx = 128009
            elif "qwen" in args.from_pretrained.lower() or "qwen" in args.tokenizer_for_chat_template.lower():
                response_template = "<|im_start|>assistant\n"
                response_template_end_idx = 151645
            else:
                raise NotImplementedError("For this type of model or tokenizer non-completion masking is not supported")
            examples["labels"] = [mask_non_completion_multi(el, response_template, response_template_end_idx, tokenizer) for el in examples["input_ids"]]
        if args.append_concat_token:
            examples["input_ids"] = [el + [tokenizer.eos_token_id] for el in examples["input_ids"]]
            if "labels" in examples.keys():
                examples["labels"] = [el + [tokenizer.eos_token_id] for el in examples["labels"]]
            if "attention_mask" in examples.keys():
                examples["attention_mask"] = [el + [1] for el in examples["attention_mask"]]
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if history_size is None:
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        else:
            result = {
                k: [t[max({0, i - history_size}) : i + block_size] for i in range(history_size, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        if not("labels" in examples.keys()):
            result["labels"] = result["input_ids"].copy()
        return result
    def prepare_texts(examples, truncate_before):
        if args.mask_non_completion:
            if "llama" in args.from_pretrained.lower() or "llama" in args.tokenizer_for_chat_template.lower():
                response_template = "<|start_header_id|>assistant<|end_header_id|>"
                response_template_end_idx = 128009
            elif "qwen" in args.from_pretrained.lower() or "qwen" in args.tokenizer_for_chat_template.lower():
                response_template = "<|im_start|>assistant\n"
                response_template_end_idx = 151645
            else:
                raise NotImplementedError("For this type of model or tokenizer non-completion masking is not supported")
            examples["labels"] = [mask_non_completion_multi(el, response_template, response_template_end_idx, tokenizer) for el in examples["input_ids"]]
        #print(examples["labels"][0])
        result = {
            k: [el[truncate_before:] for el in t]
            for k, t in examples.items()
        }
        if not("labels" in examples.keys()):
            result["labels"] = result["input_ids"].copy()
        return result

    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if args.sliding_window:
        def collate_fn(batch):
            # deprecated
            input_ids = [torch.tensor(b['input_ids']) for b in batch]
            input_lens = [el.shape[-1] for el in input_ids]

            labels = [torch.tensor(b['labels']) for b in batch]
            attention_mask = [torch.tensor(b['attention_mask']) for b in batch]
            input_ids = pad_sequence(input_ids, padding_value=id_pad_value).T
            labels = pad_sequence(labels, padding_value=-100).T
            attention_mask = pad_sequence(attention_mask, padding_value=0).T

            # make sliding window att mask
            attention_mask = attention_mask[:, None, :].repeat(1, attention_mask.shape[1], 1)
            attention_mask = (torch.tril(attention_mask, 0) * (1 - torch.tril(attention_mask, -block_size)))

            collated = {'input_ids': input_ids,
                        'labels': labels, 
                        'attention_mask': attention_mask}

            if input_ids.shape[1] != block_size and args.num_mem_tokens is not None:
                # take only labels for last block (maybe use all labels during training?)
                labels_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for i, lens in enumerate(input_lens):
                    labels_mask[i, max(lens - block_size, 0): lens] = True
                collated['labels_mask'] = labels_mask

            return collated
    else:
        def collate_fn(batch):
            input_ids = [torch.tensor(b['input_ids']) for b in batch]
            if not("labels" in batch[0].keys()):
                labels = [torch.tensor(b['input_ids'].copy()) for b in batch]
            else:
                labels = [torch.tensor(b['labels']) for b in batch]
            attention_mask = [torch.ones_like(b, dtype=int) for b in input_ids]


            labels_mask = [torch.ones_like(b, dtype=int) for b in input_ids]
            
            if getattr(args, 'loss_from_last_seg_only', False):
                for m in labels_mask:
                    m[:-args.segment_size] = False

            if getattr(args, 'no_loss_from_first_segment', False):
                for m in labels_mask:
                    m[:args.segment_size] = False

            input_ids = pad_sequence(input_ids, padding_value=id_pad_value, batch_first=True)
            labels = pad_sequence(labels, padding_value=-100, batch_first=True)
            attention_mask = pad_sequence(attention_mask, padding_value=0, batch_first=True)
            labels_mask = pad_sequence(labels_mask, padding_value=0, batch_first=True)

            collated = {'input_ids': input_ids,
                        'labels': labels, 
                        'attention_mask': attention_mask,
                        }
            if args.num_mem_tokens is not None:
                # add labels mask only for RMT, ARMT
                collated['labels_mask'] = labels_mask.bool()
            # TODO: add masking for response only
            return collated
    if not(args.no_packing):
        with accelerator.main_process_first():
            train_dataset = tokenized_datasets["train"].map(lambda x: group_texts(x, block_size, history_size),
                                                            batched=True, desc=f"Grouping train in chunks of {block_size} and history {history_size}")
            valid_dataset = tokenized_datasets["validation"].map(lambda x: group_texts(x, block_size, history_size), 
                                                                batched=True, desc=f"Grouping valid in chunks of {block_size}")
            test_dataset = tokenized_datasets["test"].map(lambda x: group_texts(x, block_size, history_size),
                                                          batched=True, desc=f"Grouping test in chunks of {block_size}")
    else:
        with accelerator.main_process_first():
            train_dataset = tokenized_datasets["train"].map(lambda x: prepare_texts(x, args.truncate_before),
                                                            batched=True, desc=f"Preparing train")
            if args.truncate_only_train:
                args.truncate_before = 0
            valid_dataset = tokenized_datasets["validation"].map(lambda x: prepare_texts(x, args.truncate_before),
                                                                batched=True, desc=f"Preparing valid")
            test_dataset = tokenized_datasets["test"].map(lambda x: prepare_texts(x, args.truncate_before),
                                                          batched=True, desc=f"Preparing test")
            #train_dataset = tokenized_datasets["train"]
            #valid_dataset = tokenized_datasets["validation"]
            #test_dataset = tokenized_datasets["test"]
    # define model
    # TODO: move model building to separate function
    model_cls = get_cls_by_name(args.model_cls)
    logger.info(f'Using model class: {model_cls}')

    if args.use_adapter:
        model_cfg = AutoConfig.from_pretrained(args.from_pretrained)

        model_cfg.use_parallel_adapter = args.use_adapter
        model_cfg.parallel_adapter_mode = 'ffn'
        model_cfg.adapter_bottleneck_dim = args.adapter_bottleneck_dim
        model_cfg.adapter_dropout = args.adapter_dropout
        model_cfg.adapter_scale = args.adapter_scale

        model = model_cls(config=model_cfg)

        logger.info(f'Loading pretrained model: {args.from_pretrained}')
        base_model = model_cls.from_pretrained(args.from_pretrained, use_safetensors=False)

        model.load_state_dict(base_model.state_dict(), strict=False)
        del base_model
        logger.info(f'Added adapters')
    else:
        # TODO: fix if for Qwen and Llama
        if not args.from_pretrained:
            model_cfg = AutoConfig.from_pretrained(args.model_cfg)
            model = model_cls(config=model_cfg)
        else:
            logger.info(f'Loading pretrained model: {args.from_pretrained}')
            if "Qwen" in args.from_pretrained or "Llama" in args.from_pretrained:
                model = model_cls.from_pretrained(args.from_pretrained,
                                                  attn_implementation="flash_attention_2",
                                                  torch_dtype=torch.bfloat16,
                                                  #device_map="cuda",
                                                  trust_remote_code=True)
            else:
                model = model_cls.from_pretrained(args.from_pretrained, use_safetensors=False)
    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_attn_dim,
            lora_alpha=args.lora_attn_alpha,
            lora_dropout=args.lora_dropout
            )
        model = get_peft_model(model, peft_config)
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        model.print_trainable_parameters()
    ## load cpt of backbone model
    if args.backbone_cpt:
        backbone_cpt = os.path.join(args.backbone_cpt, "model_best.pth")
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'], strict=False)
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')
    # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
        recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
        logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')
        mem_cell_args = dict(
            base_model=model,
            num_mem_tokens=args.num_mem_tokens,
        )
        # additional parameters for ARMT model
        if args.d_mem is not None:
            mem_cell_args['d_mem'] = args.d_mem
            mem_cell_args['wrap_pos'] = args.wrap_pos
            mem_cell_args['correction'] = not(args.no_correction)
            mem_cell_args['use_lora'] = args.use_lora
        if args.layers_attr is not None:
            mem_cell_args['layers_attr'] = args.layers_attr
        if args.attend_to_previous_input:
            mem_cell_args['attend_to_previous_input'] = args.attend_to_previous_input
        cell = memory_cell_cls(**mem_cell_args)
        model = recurrent_wrapper_cls(cell, 
                                      segment_size=block_size,
                                      max_n_segments=args.max_n_segments, 
                                      vary_n_segments=args.vary_n_segments,
                                      k2=args.k2,
                                      attend_to_previous_input=args.attend_to_previous_input,
                                      return_all_logits=False,
        )
        ## load cpt of rmt
        if args.model_cpt:
            if "safetensors" in args.model_cpt:
                print(model)
                from safetensors.torch import load_model
                load_model(model, args.model_cpt, device="cuda:0")
            else:
                if ".bin" in args.model_cpt:
                    model_cpt = args.model_cpt
                elif "model_best" in os.listdir(args.model_cpt):
                    model_cpt = os.path.join(args.model_cpt, "model_best", "pytorch_model.bin")
                else:
                    dir_files = os.listdir(args.model_cpt)
                    checkpoint_dir = [el for el in dir_files if "checkpoint-" in el][0]
                    model_cpt = os.path.join(args.model_cpt, checkpoint_dir, "pytorch_model.bin")
                cpt = torch.load(model_cpt, map_location='cpu')
                model.load_state_dict(cpt, strict=False)
            logger.info(f'Loaded RMT state dict from: {args.model_cpt}')
        logger.info(f'Trainable parameters after adding RMT/ARMT:')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    if args.add_lora_to_armt:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_attn_dim,
            lora_alpha=args.lora_attn_alpha,
            lora_dropout=args.lora_dropout
            )
        # add LoRA only to the inner model
        model.memory_cell.model = get_peft_model(model.memory_cell.model, peft_config)
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        model.memory_cell.model.print_trainable_parameters()
        print(model)
    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            if 'memory' not in n and 'lora' not in n and 'adapter' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info(f'Frozen model weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    if args.tune_only_memory:
        for n, p in model.named_parameters():
            if 'memory_cell.memory' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info(f'Frozen model weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    if args.tune_only_armt:
        for n, p in model.named_parameters():
            if 'memory_cell.memory' not in n and 'W_mq' not in n and 'W_mk' not in n and 'W_mv' not in n and 'W_mb' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info(f'Frozen model weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    # fix the not-contiguous error
    def make_contiguous(module):
        with torch.no_grad():
            for param in module.parameters():
                param.set_(param.contiguous())
    make_contiguous(model)
    # now switch to HF trainer
    training_args_dict = {key: value for key, value in vars(args).items() if hasattr(SFTConfig('.'), key)}

    training_args_dict['remove_unused_columns'] = False
    training_args_dict['save_safetensors'] = False
    training_args_dict['bf16'] = True
    training_args_dict['label_names'] = ['labels']
    training_args_dict['evaluation_strategy'] = 'steps'
    if training_args_dict.get('per_device_train_batch_size') == 1:
        training_args_dict['per_device_eval_batch_size'] = training_args_dict.get('per_device_train_batch_size')
    else:
        training_args_dict['per_device_eval_batch_size'] = training_args_dict.get('per_device_train_batch_size') // 2
    training_args_dict['eval_accumulation_steps'] = 32
    if args.d_mem is None:
        # for now, gradient checkpointing doesn't supported for ARMT
        training_args_dict['gradient_checkpointing'] = True
        training_args_dict['gradient_checkpointing_kwargs'] = {'use_reentrant': False}
    training_args_dict['log_level'] = 'debug'
    training_args_dict['load_best_model_at_end'] = args.early_stopping_patience != -1
    if args.num_mem_tokens is not None:
        # fix max_seq_length warning
        training_args_dict["max_seq_length"] = args.sample_size
        model.to(torch.bfloat16)
    training_args = SFTConfig(**training_args_dict)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )
    logger.info(f"Trainer Gradient Checkpointing Enabled: {trainer.args.gradient_checkpointing}")
    if args.early_stopping_patience != -1:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )
        trainer.add_callback(early_stopping)
    start_metrics = trainer.evaluate()
    logger.info(f"Metrics of initial model: {start_metrics}")
    if not args.validate_only:
        trainer.train(resume_from_checkpoint=args.checkpoint)
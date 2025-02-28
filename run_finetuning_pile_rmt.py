import json
import logging
import os
import math
import random
import shutil
from pathlib import Path
from itertools import chain

# from dotenv import load_dotenv
import torch
import numpy as np
import accelerate
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, load_from_disk

from lm_experiments_tools import Trainer, TrainerArgs

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler

from peft import get_peft_model, LoraConfig, TaskType
# load_dotenv()

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

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--task_name', type=str, help="Task name, wikitext, ...")
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
parser.add_argument('--block_size', type=int, default=128, help='max size of language modeling block')
parser.add_argument('--history_size', type=int, default=0, help='max number of past tokens for each block')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

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

# Parallel Adapter args
parser.add_argument('--use_adapter', action='store_true', default=False, help='')
parser.add_argument('--adapter_bottleneck_dim', type=int, default=512, help='')
parser.add_argument('--adapter_dropout', type=float, default=0.1, help='')
parser.add_argument('--adapter_scale', type=float, default=4.0, help='')

# Dataset args
parser.add_argument('--pile_subset_names', type=str, default=None, help='use only these subsets of The PILE, separated by ;')
parser.add_argument('--min_tokens_in_document', type=int, default=None, help='do not use documents shorter than this value')
parser.add_argument('--max_tokens_in_document', type=int, default=None, help='do not use documents longer than this value')


if __name__ == '__main__':
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    from accelerate.logging import get_logger
    logger = get_logger('')

    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')

    if args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    # # create model path and save configuration
    # # todo: use prepare run
    # if accelerator.is_main_process and args.model_path is not None:
    #     model_path = Path(args.model_path)
    #     if not model_path.exists():
    #         Path(model_path).mkdir(parents=True)
    #     args_dict = collect_run_configuration(args)
    #     # todo: if model path exists and there is config file, write new config file aside
    #     json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)
    #     open(model_path / 'git.diff', 'w').write(get_git_diff())

    prepare_run(args, logger, logger_fmt)

    if not args.from_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    # Prepare datasets
    logger.info(f'preparing dataset for {args.task_name}')
    
    block_size = args.block_size if args.block_size is not None else args.input_size
    history_size = args.history_size

    def filter_by_subset(sample, subset_name):
        if not isinstance(subset_name, list):
            subset_name = [subset_name]
        
        return any([sn in sample['meta']['pile_set_name'] for sn in subset_name])

    def filter_by_len(sample, field, min_len=None, max_len=None):
        
        sample_text = sample[field]
        if min_len is not None and len(sample_text) < min_len:
            return False
        if max_len is not None and len(sample_text) > max_len:
            return False

        return True

    def tokenization(sample):
        return tokenizer.encode(sample["text"])

    class BlockDataLoader(DataLoader):
        def __init__(self, dataset, block_size, history_size, max_samples=None, shuffle=False, *args, **kwargs):
            super().__init__(dataset, *args, **kwargs)
            self.block_size = block_size
            self.history_size = history_size
            self.max_samples = max_samples
            self.shuffle = shuffle
                
        def get_samples(self, document):
            input_ids = tokenizer.encode(document['text'])
            # do not load short documents
            if len(input_ids) < self.block_size:
                return []
            if args.min_tokens_in_document is not None and len(input_ids) < args.min_tokens_in_document:
                return []
            if args.validate_only:
                samples = [input_ids[max({0, start - self.history_size - self.block_size}): start] for start in range(self.block_size, len(input_ids), self.block_size)]
            else:
                samples = [input_ids[max({0, start - self.history_size - self.block_size}): start] for start in range(self.block_size + self.history_size, len(input_ids), self.block_size)]
            
            
            # samples = [input_ids[start: start + self.block_size] for start in range(self.history_size, len(input_ids), self.block_size)]
            # print('samples', [len(s) for s in samples])
            # print('input_ids', len(input_ids))
            return samples
        
        def __iter__(self):
            inds = list(range(len(self.dataset)))

            if self.max_samples is not None:
                inds = inds[:self.max_samples]

            if self.shuffle: 
                np.random.shuffle(inds)

            doc_ind = 0
            samples = []

            while True:
                if doc_ind >= len(inds):
                    break
                try: 
                    while len(samples) < self.batch_size:
                        document = self.dataset[inds[doc_ind]]
                        doc_ind += 1
                        samples += self.get_samples(document)
                        if doc_ind >= len(inds):
                            raise(StopIteration)
                except(StopIteration):
                    pass

                batch, samples = samples[:self.batch_size], samples[self.batch_size:]
                # print('batch', [len(b) for b in batch])
                # print('samples', [len(b) for b in samples])
                yield self.collate_fn(batch)
                
            

    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    def collate_fn(batch):
        input_ids = labels = [torch.tensor(b) for b in batch]
        attention_mask = [torch.ones_like(b, dtype=int) for b in input_ids]
        labels_mask = [torch.ones_like(b, dtype=int) for b in input_ids]
        for m in labels_mask:
            m[:-block_size] = False
        
        input_ids = pad_sequence(input_ids, padding_value=id_pad_value, batch_first=True)
        labels = pad_sequence(labels, padding_value=-100, batch_first=True)
        attention_mask = pad_sequence(attention_mask, padding_value=0, batch_first=True)
        labels_mask = pad_sequence(labels_mask, padding_value=0, batch_first=True)

        collated = {'input_ids': input_ids,
                    'labels': labels, 
                    'attention_mask': attention_mask}
        
        if input_ids.shape[1] != block_size:
            collated['labels_mask'] = labels_mask.bool()

        if getattr(args, 'vary_n_segments', False):
            if getattr(args, 'sampling_prob', False):
                random_value = torch.rand((1)).item()
                if random_value > args.sampling_prob:
                    return collated
                
            n_segments = random.randint(0, args.max_n_segments)
            n_tokens = n_segments * block_size
            for k in collated:
                collated[k] = collated[k][:, -n_tokens:]
        
        # print('\n\n\ninput_ids', input_ids.shape)
        # print('collated', collated)
        return collated


    dataset = load_dataset('/home/jovyan/rmt/datasets/pile/data')

    if args.pile_subset_names is not None:
        subset_names = args.pile_subset_names.split(';')
        print(f'Leaving only following subsets: {subset_names}')
        dataset = dataset.filter(lambda sample: filter_by_subset(sample, subset_names))

    # min_tokens, max_tokens = args.min_tokens_in_document, args.max_tokens_in_document
    # max_chars = max_tokens * 128 if max_tokens is not None else None
    # if min_tokens is not None or max_chars is not None:
    #     dataset = dataset.filter(lambda sample: filter_by_len(sample, 'text', min_tokens, max_chars))

    # dataset = dataset.map(tokenization, batched=True)

    # if min_tokens is not None or max_tokens is not None:
    #     dataset = dataset.filter(lambda sample: filter_by_len(sample, 'input_ids', min_tokens, max_tokens))

    train_dataset, valid_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"]

    
    # shuffle train data each epoch (one loop over train_dataset)
    train_rnd_generator = torch.Generator()
    train_rnd_generator.manual_seed(args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
        
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = BlockDataLoader(train_dataset, batch_size=per_worker_batch_size,  #generator=train_rnd_generator,
                                    sampler=train_sampler,
                                    block_size=block_size, 
                                    history_size=history_size, 
                                    shuffle=True,
                                    collate_fn=collate_fn, **kwargs)

    # get validation dataset
    max_samples = 20000 if args.validate_only else 100
    # max_samples = 100
    valid_dataloader = None
    logger.info(f'preparing validation data')
    valid_sampler = DistributedSampler(valid_dataset, shuffle=True)
    valid_dataloader = BlockDataLoader(valid_dataset, batch_size=per_worker_batch_size,
                                    sampler=valid_sampler,
                                    block_size=block_size, 
                                    history_size=history_size, 
                                    shuffle=False,
                                    max_samples=max_samples,
                                    collate_fn=collate_fn, drop_last=True, **kwargs)
    
    # # get test dataset
    # test_sampler = DistributedSampler(test_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
    # test_dataloader = BlockDataLoader(test_dataset, batch_size=per_worker_batch_size, sampler=test_sampler,
    #                                 block_size=block_size, 
    #                                 history_size=history_size, 
    #                                 shuffle=False,
    #                                 max_samples=max_samples,
    #                                 collate_fn=collate_fn, drop_last=True, **kwargs)
    
    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    # define model
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
        if not args.from_pretrained:
            model_cfg = AutoConfig.from_pretrained(args.model_cfg)
            model = model_cls(config=model_cfg)
        else:
            logger.info(f'Loading pretrained model: {args.from_pretrained}')
            model = model_cls.from_pretrained(args.from_pretrained, use_safetensors=False)

    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_attn_dim, 
            lora_alpha=args.lora_attn_alpha, 
            lora_dropout=args.lora_dropout,
            layers_pattern=args.layers_pattern
            )
        model = get_peft_model(model, peft_config)
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        model.print_trainable_parameters()
    

    ## load cpt of backbone model
    if args.backbone_cpt:
        backbone_cpt = os.path.join(args.backbone_cpt, "pytorch_model.bin")
        # model = torch.load(backbone_cpt, map_location='cpu')
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt, strict=False)
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')

    # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
        recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
        logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')
        
        base_model_forward = args.base_model_forward
        if base_model_forward is not None:
            base_model_forward = get_cls_by_name(base_model_forward)
            logger.info(f"Using custom forward function: {base_model_forward}")
        cell = memory_cell_cls(model, args.num_mem_tokens)
        if args.segment_alignment not in {None, 'left'}:
            logger.info(f"Using custom segment alignment: {args.segment_alignment}")
        model = recurrent_wrapper_cls(cell, 
                                      segment_size=block_size,
                                      max_n_segments=args.max_n_segments, 
                                      vary_n_segments=args.vary_n_segments,
                                      segment_alignment=args.segment_alignment,
                                      k2=args.k2,
                                      base_model_forward_func=base_model_forward
        )
                                    

        ## load cpt of rmt
        if args.model_cpt:
            model_cpt = os.path.join(args.model_cpt, "pytorch_model.bin")
            cpt = torch.load(model_cpt, map_location='cpu')
            model.load_state_dict(cpt, strict=False)
            logger.info(f'Loaded RMT state dict from: {args.model_cpt}')

    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            if 'memory' not in n and 'lora' not in n and 'adapter' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info(f'Frozen moodel weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')

    # fix the not-contiguous error
    def make_contiguous(module):
        with torch.no_grad():
            for param in module.parameters():
                param.set_(param.contiguous())
    make_contiguous(model)

    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    logger.info(f'Using optimizer class: {optimizer_cls}')

    # todo: group optimizer params
    optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)    
    # if args.model_cpt or args.backbone_cpt:
    #     optimizer.load_state_dict(cpt['optimizer_state_dict'])

    # for encoder only classification
    def keep_for_metrics_fn(batch, output):
        # select data from batch and model output that would be used to compute metrics
        data = {}
        data['labels'] = batch['labels']
        data['loss'] = output['loss']
        if 'logits' in output:
            data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
        return data

    # HF datasets can compute metrics on each gpu process and then aggregate them on process with rank 0
    # synchronization is done by using temporay files on a shared filesystem
    # rank and number of workers is set by num_process and process_id params
    # BUT our Trainer aggregates all prediction from all gpus!
    #   this will lead to computing metrics for predictions repeated xN_GPUS times
    # need to try:
    # - keep_in_memory=True, may lead to OOM for large validation sets, after sync predictions and targets for the full
    #       validation set would be stored on each GPU -> xN_GPUs RAM
    #   - implemented currently
    # - compute metrics on batch lvl
    # - add support of HF metrics and turn off aggregation in case if metric has .add_batch method
    # scrolls_metric = datasets.load_metric(scrolls_metric_path, args.task_name, keep_in_memory=True)

    model, optimizer = accelerator.prepare(model, optimizer)
    # model, optimizer, _ = accelerator.prepare(model, optimizer, train_dataloader)

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        if 'predictions' in data:
            y, p = data['labels'], data['predictions']
            if args.show_valid_examples > 0:
                for i in range(min(args.show_valid_examples, len(y))):
                    logger.info(f'y: {y[i][:100]}')
                    logger.info(f'p: {p[i][:100]}')
                    # pred = p[i][p[i] != -100]
                    # lab = y[i][y[i] != -100]
                    # logger.info(f'pred: {pred}')
                    # logger.info(f'lab: {lab}')
                    pred = p[i]
                    lab = y[i]
                    try:
                        logger.info(f'y: {tokenizer.decode(lab)[:100]}')
                    except OverflowError as e:
                        logger.info(f"Error in decoding labels: {e}, {lab}")
                    try:
                        logger.info(f'y: {tokenizer.decode(pred)[:100]}')
                    except OverflowError as e:
                        logger.info(f"Error in decoding predictions: {e}, {pred}")
                    
                    logger.info('-' * 50)
        try:
            perplexity = math.exp(data["loss"].mean())
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        return metrics

    ### booydar
    batch_metrics_fn = lambda _, y: {key: y[key] for key in y.keys() if (('loss' in key) or ('!log' in key))}
    trainer = Trainer(args, accelerator, model, optimizer, train_dataloader, valid_dataloader,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
                      ###booydar
                      batch_metrics_fn=batch_metrics_fn,
                      generate_kwargs={})

    if not args.validate_only:
        # train loop
        trainer.train()
        # make sure all workers are done
        accelerator.wait_for_everyone()
        # run validation after training
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best')
            logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
        if valid_dataloader is not None:
            logger.info('Runnning validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False, split='valid')
        # if test_dataloader is not None:
        #     logger.info('Runnning validation on test data:')
        #     trainer.validate(test_dataloader, write_tb=True, split='test')
        trainer.save_metrics(save_path=args.model_path)
    else:
        # run validation, do not write to tensorboard
        # logger.info('Running validation on train set:')
        # trainer.validate(train_dataloader, split='train', write_tb=True)
        if valid_dataloader is not None:
            logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=True, split='valid')
        # if test_dataloader is not None:
        #     logger.info('Runnning validation on test data:')
        #     trainer.validate(test_dataloader, write_tb=False, split='test')

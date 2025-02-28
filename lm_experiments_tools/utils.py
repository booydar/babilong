import functools
import importlib
import inspect
import json
import logging
import os
import platform
import subprocess
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List

import torch
import transformers

import lm_experiments_tools.optimizers


def get_cls_by_name(name: str) -> type:
    """Get class by its name and module path.

    Args:
        name (str): e.g., transfomers:T5ForConditionalGeneration, modeling_t5:my_class

    Returns:
        type: found class for `name`
    """
    module_name, cls_name = name.split(':')
    return getattr(importlib.import_module(module_name), cls_name)


def get_git_hash_commit() -> str:
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        # no git installed or we are not in repository
        commit = ''
    return commit


def get_git_diff() -> str:
    try:
        diff = subprocess.check_output(['git', 'diff', 'HEAD', '--binary']).decode('utf8')
    except subprocess.CalledProcessError:
        # no git installed or we are not in repository
        diff = ''
    return diff


def get_fn_param_names(fn) -> List[str]:
    """get function parameters names except *args, **kwargs

    Args:
        fn: function or method

    Returns:
        List[str]: list of function parameters names
    """
    params = []
    for p in inspect.signature(fn).parameters.values():
        if p.kind not in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]:
            params += [p.name]
    return params


def get_optimizer(name: str):
    if ':' in name:
        return get_cls_by_name(name)
    if hasattr(lm_experiments_tools.optimizers, name):
        return getattr(lm_experiments_tools.optimizers, name)
    if hasattr(torch.optim, name):
        return getattr(torch.optim, name)
    if hasattr(transformers.optimization, name):
        return getattr(transformers.optimization, name)
    try:
        apex_opt = importlib.import_module('apex.optimizers')
        return getattr(apex_opt, name)
    except (ImportError, AttributeError):
        pass
    return None


def collect_run_configuration(args, env_vars=['CUDA_VISIBLE_DEVICES']):
    args_dict = dict(vars(args))
    args_dict['ENV'] = {}
    for env_var in env_vars:
        args_dict['ENV'][env_var] = os.environ.get(env_var, '')
    # hvd
    try:
        import horovod.torch as hvd
        args_dict['HVD_INIT'] = hvd.is_initialized()
        if hvd.is_initialized():
            args_dict['HVD_SIZE'] = hvd.size()
    except ImportError:
        pass
    # accelerate
    # todo: collect full accelerate config
    try:
        import accelerate
        args_dict['accelerate'] = {}
        args_dict['accelerate']['initialized'] = accelerate.PartialState().initialized
        if accelerate.PartialState().initialized:
            args_dict['accelerate']['num_processes'] = accelerate.PartialState().num_processes
            args_dict['accelerate']['backend'] = accelerate.PartialState().backend
            args_dict['accelerate']['distributed_type'] = accelerate.PartialState().distributed_type
    except ImportError:
        pass

    args_dict['MACHINE'] = platform.node()
    args_dict['COMMIT'] = get_git_hash_commit()
    return args_dict


def get_distributed_rank() -> int:
    try:
        import accelerate
        if accelerate.PartialState().initialized:
            return accelerate.PartialState().process_index
    except ImportError:
        pass

    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    try:
        import horovod.torch as hvd
        if hvd.is_initialized():
            return hvd.rank()
    except ImportError:
        pass

    return 0


def rank_0(fn):
    @functools.wraps(fn)
    def rank_0_wrapper(*args, **kwargs):
        if get_distributed_rank() == 0:
            return fn(*args, **kwargs)
        return None
    return rank_0_wrapper


def wait_for_everyone():
    try:
        import accelerate
        if accelerate.PartialState().initialized:
            accelerate.PartialState().wait_for_everyone()
    except ImportError:
        pass

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    try:
        import horovod.torch as hvd
        if hvd.is_initialized():
            hvd.barrier()
    except ImportError:
        pass


def prepare_run(args, logger=None, logger_fmt: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                add_file_logging=True):
    """creates experiment directory, saves configuration and git diff, setups logging

    Args:
        args: arguments parsed by argparser, model_path is a required field in args
        logger: python logger object
        logger_fmt (str): string with logging format
        add_file_logging (bool): whether to write logs into files or not
    """

    # create model path and save configuration
    rank = get_distributed_rank()
    if rank == 0 and args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        # todo: if model path exists and there is config file, write new config file aside
        json.dump(args_dict, open(model_path / 'config.json', 'w'), indent=4)
        open(model_path / 'git.diff', 'w').write(get_git_diff())

    # configure logging to a file
    if args.model_path is not None and logger is not None and add_file_logging:
        # sync workers to make sure that model_path is already created by worker 0
        wait_for_everyone()
        # RotatingFileHandler will keep logs only of a limited size to not overflow available disk space.
        # Each gpu worker has its own logfile.
        # todo: make logging customizable? reconsider file size limit?
        fh = RotatingFileHandler(Path(args.model_path) / f"{time.strftime('%Y.%m.%d_%H:%M:%S')}_rank_{rank}.log",
                                 mode='w', maxBytes=100*1024*1024, backupCount=2)
        logger_with_fh = logger
        if isinstance(logger, logging.LoggerAdapter):
            logger_with_fh = logger.logger
        fh.setLevel(logger_with_fh.level)
        fh.setFormatter(logging.Formatter(logger_fmt))
        logger_with_fh.addHandler(fh)

    if rank == 0 and args.model_path is None and logger is not None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')
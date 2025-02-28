from itertools import chain

def get_lm_datasets(raw_datasets, tokenizer, block_size):
    # copy-pasted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    # with accelerator.main_process_first():
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        # num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        # load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    # if args.block_size is None:
    # block_size = tokenizer.model_max_length
    # if block_size > 1024:
    #     logger.warning(
    #         "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
    #         " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
    #         " override this default with `--block_size xxx`."
    #     )
    # block_size = 1024
    # else:
    #     if args.block_size > tokenizer.model_max_length:
    #         logger.warning(
    #             f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
    #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
    #         )
    #     block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # with accelerator.main_process_first():
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        # num_proc=args.preprocessing_num_workers,
        # load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    valid_dataset = lm_datasets["validation"]

    return train_dataset, valid_dataset
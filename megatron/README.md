# Megatron-LM

## Readme
[Original Megatron-LM readme](https://github.com/NVIDIA/Megatron-LM/blob/e156d2fea7fc5c98e645f7742eb86b643956d840/README.md)

## Installation
```bash
cd ./data
make
```

## Data Preprocessing
[docs from Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/e156d2fea7fc5c98e645f7742eb86b643956d840/README.md#data-preprocessing)

changes in `preprocess_data.py`:
- `preprocess_data.py` script is moved to megatron folder
- supports tokenizers from HuggingFace Transformers
- `input` can be a folder with multiple json/jsonl files

example usage with HF Tokenizer:
```bash
python preprocess_data.py \
       --input ./train \
       --output-prefix ./train \
       --dataset-impl mmap \
       --tokenizer-type HFTokenizer \
       --tokenizer-name-or-path bert-base-uncased \
       --split-sentences --workers 8
```

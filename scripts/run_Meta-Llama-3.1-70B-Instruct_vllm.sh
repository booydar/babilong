#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES="" ./script_name.sh
set -e

RESULTS_FOLDER="./babilong_evals"
DATASET_NAME="RMT-team/babilong-1k-samples"
MODEL_NAME="meta-llama/Meta-Llama-3.1-70B-Instruct"
TOKENIZER="meta-llama/Meta-Llama-3.1-70B-Instruct"

# run model with vllm, e.g.:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --enable-chunked-prefill=False\
# --tensor-parallel-size 8 --enforce-eager --served-model-name meta-llama/Meta-Llama-3.1-70B-Instruct
# adjust parameters to your setup (e.g, set --max_model_len 40000)

TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k")

USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
API_URL="http://localhost:8000/v1/completions"

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python scripts/run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --tokenizer_name "$TOKENIZER" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    --api_url "$API_URL"

DATASET_NAME="RMT-team/babilong"
TASKS=("qa1" "qa2" "qa5" "qa3" "qa4")
LENGTHS=("64k" "128k")

USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
API_URL="http://localhost:8000/v1/completions"

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python scripts/run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --tokenizer_name "$TOKENIZER" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    --api_url "$API_URL"


# USE_CHAT_TEMPLATE=false
# USE_INSTRUCTION=false
# USE_EXAMPLES=false
# USE_POST_PROMPT=false
# API_URL=""

# echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

# python scripts/run_model_on_babilong.py \
#     --results_folder "$RESULTS_FOLDER" \
#     --dataset_name "$DATASET_NAME" \
#     --model_name "$MODEL_NAME" \
#     --tasks "${TASKS[@]}" \
#     --lengths "${LENGTHS[@]}" \
#     $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
#     $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
#     $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
#     $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
#     --api_url "$API_URL"

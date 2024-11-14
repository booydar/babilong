#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0,1 ./script_name.sh
set -e

RESULTS_FOLDER="./babilong_evals"
DATASET_NAME="RMT-team/babilong-1k-samples"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

TASKS=("qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k")

USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
API_URL=""

# or set API_URL and run model with vllm serve

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python scripts/run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
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

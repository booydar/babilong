#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES="" ./script_name.sh
set -e

RESULTS_FOLDER="./babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
MODEL_PATH="/home/jovyan/kuratov/models/Llama-3.2-1B-Instruct"

TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("128k" "64k")

USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
API_URL="http://localhost:8000/v1/completions"

# e.g., run locally with vllm (0.5.3.post1)
# CUDA_VISIBLE_DEVICES=0,1 vllm serve ./Llama-3.2-1B-Instruct --enable-chunked-prefill=False --tensor-parallel-size 2 \
# --served-model-name meta-llama/Llama-3.2-1B-Instruct

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python scripts/run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    --system_prompt "You are a helpful AI assistant." \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    --api_url "$API_URL"


DATASET_NAME="RMT-team/babilong-1k-samples"

TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k")

USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python scripts/run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    --system_prompt "You are a helpful AI assistant." \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    --api_url "$API_URL"

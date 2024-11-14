#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 ./script_name.sh
set -e

RESULTS_FOLDER="./babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME="THUDM/glm-4-9b-chat-1m"
TOKENIZER="THUDM/glm-4-9b-chat-1m"

# run model with vllm, e.g.:
# VLLM_ENGINE_ITERATION_TIMEOUT_S=300 # for 512k
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve ./glm-4-9b-chat-1m --served-model-name THUDM/glm-4-9b-chat-1m \
# --enable-chunked-prefill=False --tensor-parallel-size 4 --max_model_len 770192 --gpu-memory-utilization 0.99 \
# --trust_remote_code --enforce_eager

TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("512k" "128k" "64k")

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


TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k")

DATASET_NAME="RMT-team/babilong-1k-samples"
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

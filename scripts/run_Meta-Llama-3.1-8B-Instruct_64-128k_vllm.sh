#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES="" ./script_name.sh
set -e

RESULTS_FOLDER="./babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKENIZER="meta-llama/Meta-Llama-3.1-8B-Instruct"

# run model with vllm (0.5.3.post1), e.g.:
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --enable-chunked-prefill=False \
# --tensor-parallel-size 4 --served-model-name meta-llama/Meta-Llama-3.1-8B-Instruct

TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
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

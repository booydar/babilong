#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES="" ./script_name.sh
set -e

RESULTS_FOLDER="./babilong_evals"
DATASET_NAME="RMT-team/babilong-1k-samples"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
TOKENIZER="Qwen/Qwen2.5-72B-Instruct"

# run model with vllm (0.5.3.post1), e.g.:
# up to 16k
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen2.5-72B-Instruct --served-model-name Qwen/Qwen2.5-72B-Instruct \
# --enable-chunked-prefill=False --tensor-parallel-size 4 --gpu-memory-utilization 0.99 \
# --max_model_len 32768 --trust_remote_code --enforce_eager

# 32k, 64k update model config with (as recommended in https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
# "rope_scaling": {
#             "factor": 4.0,
#             "original_max_position_embeddings": 32768,
#             "type": "yarn"
#         }
# VLLM_ENGINE_ITERATION_TIMEOUT_S=300 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve ./Qwen2.5-72B-Instruct \
# --served-model-name Qwen/Qwen2.5-72B-Instruct --enable-chunked-prefill=False --tensor-parallel-size 8 \
# --gpu-memory-utilization 0.99 --max_model_len 131072 --trust_remote_code --enforce_eager

# 128k update model config with
# "rope_scaling": {
#             "factor": 5.0,
#             "original_max_position_embeddings": 32768,
#             "type": "yarn"
#         }
# VLLM_ENGINE_ITERATION_TIMEOUT_S=3000 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve ./Qwen2.5-72B-Instruct
# --served-model-name Qwen/Qwen2.5-72B-Instruct --enable-chunked-prefill=False --tensor-parallel-size 8 \
# --gpu-memory-utilization 0.99 --max_model_len 163840 --trust_remote_code --enforce_eager

TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k") #("32k")

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
    --system_prompt "You are a helpful assistant." \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    --api_url "$API_URL"


# TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
# LENGTHS=("128k") #("64k") # 128k with rope factor 5

# DATASET_NAME="RMT-team/babilong"
# USE_CHAT_TEMPLATE=true
# USE_INSTRUCTION=true
# USE_EXAMPLES=true
# USE_POST_PROMPT=true
# API_URL="http://localhost:8000/v1/completions"

# echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

# python scripts/run_model_on_babilong.py \
#     --results_folder "$RESULTS_FOLDER" \
#     --dataset_name "$DATASET_NAME" \
#     --model_name "$MODEL_NAME" \
#     --tokenizer_name "$TOKENIZER" \
#     --tasks "${TASKS[@]}" \
#     --lengths "${LENGTHS[@]}" \
#     --system_prompt "You are a helpful assistant." \
#     $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
#     $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
#     $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
#     $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
#     --api_url "$API_URL"

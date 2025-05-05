#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 TP=2 ./script_name.sh
set -e

# Function to check if the API server is ready
wait_for_server() {
    echo "Waiting for vLLM server to start..."
    while true; do
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "vLLM process failed to start!"
            exit 1
        fi
        if curl -s "${VLLM_API_URL}/completions" &>/dev/null; then
            echo "vLLM server is ready!"
            return 0
        fi
        sleep 1
    done
}

# Function to kill the vLLM server
cleanup() {
    echo "Stopping vLLM server..."
    pkill -f "vllm serve" || true
}

# API configuration
VLLM_API_HOST="${VLLM_API_HOST:-localhost}"
VLLM_API_PORT="${VLLM_API_PORT:-8000}"
VLLM_API_URL="${VLLM_API_URL:-http://${VLLM_API_HOST}:${VLLM_API_PORT}/v1}"

RESULTS_FOLDER="./babilong_evals"
MODEL_NAME="meta-llama/Llama-4-Scout-17B-16E-Instruct"
MODEL_PATH="/home/jovyan/kuratov/models/Llama-4-Scout-17B-16E-Instruct"

# Start the vLLM server in the background
# Comment this section if vLLM server is already running.
# 4xA100 80GB, vllm 0.8.4
echo "Starting vLLM server..."
VLLM_DISABLE_COMPILE_CACHE=1
vllm serve "$MODEL_PATH" --enable-chunked-prefill=False --tensor-parallel-size $TP \
    --served-model-name "$MODEL_NAME" --host "${VLLM_API_HOST}" --port "${VLLM_API_PORT}" --disable-log-requests \
    --max_model_len 42000 --override-generation-config='{"attn_temperature_tuning": true}' &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for the server to be ready
wait_for_server

# Set up trap to ensure cleanup on script exit
trap cleanup EXIT

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
    --system_prompt "You are a helpful assistant." \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    --api_url "${VLLM_API_URL}/completions"

# Cleanup will be automatically called by the trap
echo Done
#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=4
# CUDA_VISIBLE_DEVICES=0 ./script_name.sh
set -e

# Eval Llama3.2-1b base + IFT versions from various datasets
# use versions w/o chat template





# now ARMT on babilong, v3.1, 8s
# segment size increased to 1056 = 1024 + 2*16 for script compatibility
RESULTS_FOLDER="../babilong_evals/"
DATASET_NAME="RMT-team/babilong"
#"RMT-team/babilong"
# MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
# MODEL_CPT='../../data/pretrained_models/RMT-Llama-3.2-1B-Instruct-8x1024-mem16-lora-babilong-qa1-5_ct-v3.1/model.safetensors'
MODEL_NAME='/home/jovyan/kuratov/models/Llama-3.2-1B/'

MODEL_TITLE="llama3.2-1b"

# TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
# LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k" "64k")
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5" "qa6" "qa7" "qa8" "qa9" "qa10")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
# LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
API_URL=""

echo running $MODEL_NAME $MODEL_TITLE on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_title "$MODEL_TITLE" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    --api_url "$API_URL" \
    --segment_size 1056 \
    --max_n_segments 256 \
    --use_peft 0 \
    --add_question_prompt "Answer with a single word." \
    --segment_alignment right

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
MODEL_NAME='/home/jovyan/kuratov/models/Llama-3.2-1B-Instruct/'
MODEL_CPTS=(
    # "/home/jovyan/rmt/runs/test/babilong_multitask/meta-llama/Llama-3.2-1B-Instruct/lr_3e-04_d64_cosine_adamw_wd1e-03_1x1024_mem16_bs64_bptt--1_from_cpt_0-1_lora_ct-v3/run_1/checkpoint-6500/pytorch_model.bin" \
    # "/home/jovyan/rmt/runs/test/babilong_multitask/meta-llama/Llama-3.2-1B-Instruct/lr_3e-04_d64_cosine_adamw_wd1e-03_2x1024_mem16_bs64_bptt--1_from_cpt_1-2_lora_ct-v3/run_1/checkpoint-28000/pytorch_model.bin" \
    # "/home/jovyan/rmt/runs/test/babilong_multitask/meta-llama/Llama-3.2-1B-Instruct/lr_3e-04_d64_cosine_adamw_wd1e-03_4x1024_mem16_bs64_bptt--1_from_cpt_2-4_lora_ct-v3/run_1/checkpoint-24500/pytorch_model.bin" \
    "/home/jovyan/rmt/runs/test/babilong_multitask/meta-llama/Llama-3.2-1B-Instruct/lr_3e-04_d64_cosine_adamw_wd1e-03_8x1024_mem16_bs64_bptt--1_from_cpt_4-8_lora_ct-v3/run_1/checkpoint-30000/pytorch_model.bin" \
)

MODEL_TITLES=(
    # "armt-llama3.2-1b-1x1024-ct-v3-lora" \
    # "armt-llama3.2-1b-2x1024-ct-v3-lora" \
    # "armt-llama3.2-1b-4x1024-ct-v3-lora" \
    "armt-llama3.2-1b-8x1024-ct-v3-lora-lora-instruct_end-examples_end-ppt" \
)

for (( j=0; j<${#MODEL_CPTS[@]}; j++ ))
do

MODEL_CPT=${MODEL_CPTS[j]} 
MODEL_TITLE=${MODEL_TITLES[j]}

TASKS=("qa1" "qa2" "qa3" "qa4" "qa5" "qa6" "qa7" "qa8" "qa9" "qa10")
# TASKS=("qa6" "qa7" "qa8" "qa9" "qa10")
# LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k" "64k")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
# LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
API_URL=""

echo running $MODEL_NAME $MODEL_TITLE on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_babilong_ex_end.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_title "$MODEL_TITLE" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1056 \
    --max_n_segments 256 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --add_question_prompt "Answer with a single word." \
    --segment_alignment right

done
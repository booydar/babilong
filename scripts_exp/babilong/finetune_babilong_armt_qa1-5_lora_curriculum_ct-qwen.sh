#!/usr/bin/env bash
set -e
cd ../../

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1
MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.language_modeling:AssociativeRecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
NOISE_DATASET=pg19
METRIC=exact_match

for TASK_DATASET in "qa1_single-supporting-fact;qa2_two-supporting-facts;qa3_three-supporting-facts;qa4_two-arg-relations;qa5_three-arg-relations"
do

ITERS=1000
TBS=64
INPUT_SIZE=1024

MAX_N_SEGMENTSS=(2)
MEMORY_SIZE=16
D_MEM=64
for N in 2
do

MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
for MODEL_NAME in "Qwen/Qwen2.5-0.5B-Instruct"
do

ITERS=40000
TBS=64

for LR in 3e-04
do

for SEGMENT_SIZE in 1024 # size of one segment in tokens
do


# MAX_N_SEGMENTSS=(0 1 2 4 6 8 16 32)
# BSS=(16 8 8 4 4 2 2 1 1)

# MAX_N_SEGMENTSS=(0 1 2 4 8 16 32)
# BSS=(4 4 4 2 2 1 1)

MAX_N_SEGMENTSS=(0 1 2 4 8 16 32)
BSS=(4 4 4 2 2 1 1 1)

for (( j=2; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 
BS=${BSS[j]}

j1=$((j-1))
SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j1]}

j2=$((j-2))
SRC_SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j2]}


for MEMORY_SIZE in 16
do

SAMPLE_SIZE=$((MAX_N_SEGMENTS * SEGMENT_SIZE)) # length of task sample in tokens
GRAD_ACC_STEPS=$((TBS / (BS * NP))) # Calculate gradient accumulation steps

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
SCHEDULER=linear
MAX_N_FACTS=$((SAMPLE_SIZE/10))

for N in 1
do

K2=-1   # BPTT unroll length

NP=$NP

ACCEL_CONFIG=/home/jovyan/rmt/babilong/accel_configs/accelerate_bf16-$NP.yaml
accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29009 run_finetuning_babilong_armt_hf_trainer_ct.py \
        --task_dataset $TASK_DATASET \
        --noise_dataset $NOISE_DATASET \
        --babi_path /home/jovyan/rmt/babilong/data/tasks_1-20_v1-2/en-10k \
        --output_dir /home/jovyan/rmt/runs/test/babilong_multitask/$MODEL_NAME/lr_${LR}_d${D_MEM}_${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_N_SEGMENTS}-${MAX_N_SEGMENTS}_lora_ct-v3/run_$N \
        --model_cpt /home/jovyan/rmt/runs/test/babilong_multitask/$MODEL_NAME/lr_3e-04_d${D_MEM}_${SCHEDULER}_adamw_wd1e-03_${SRC_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_SRC_N_SEGMENTS}-${SRC_N_SEGMENTS}_lora_ct-v3/run_$N \
        --from_pretrained $MODEL_PATH \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --use_lora \
        --layers_attr base_model.base_model.layers \
        --max_n_segments $MAX_N_SEGMENTS\
        --max_n_facts $MAX_N_FACTS \
        --vary_n_segments \
        --per_device_train_batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --max_steps $ITERS \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --save_total_limit 1 \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --learning_rate  ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --logging_steps 50 --eval_steps 100 --save_steps 500 \
        --show_valid_examples 0 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --max_grad_norm 1.0 \
        --tokenizer_for_chat_template $MODEL_PATH \
        --d_mem $D_MEM \
        --report_to tensorboard
done
done
done
done
done
done
done
done
echo "done"

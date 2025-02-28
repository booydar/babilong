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

MEMORY_SIZE=16
D_MEM=64
for N in 1
do

MODEL_PATH="/home/jovyan/kuratov/models/Llama-3.2-1B-Instruct/"
for MODEL_NAME in "meta-llama/Llama-3.2-1B-Instruct"
do

ITERS=15000
TBS=64
INIT_BS=16

for LR in 3e-04
do

for SEGMENT_SIZE in 1024 # size of one segment in tokens
do

for MAX_N_SEGMENTS in 1
do

for MEMORY_SIZE in 16
do

SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens
BS=$(((INIT_BS/MAX_N_SEGMENTS)))

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
SCHEDULER=linear
MAX_N_FACTS=$((SAMPLE_SIZE/10))

for N in 1
do

K2=-1   # BPTT unroll length

NP=$NP
# ACCEL_CONFIG=/home/jovyan/rmt/babilong/accel_configs/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
# cd accel_configs/
# python create_config.py \
#         --bf16 \
#         --train_batch_size $TBS\
#         --train_micro_batch_size_per_gpu $BS\
#         --gradient_accumulation_steps $GRAD_ACC_STEPS\
#         --np $NP\
#         --gradient_clipping 1.0
# cd ..

# echo RUNNING: TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
# echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
# echo gradient accumulation steps $GRAD_ACC_STEPS


ACCEL_CONFIG=/home/jovyan/rmt/babilong/accel_configs/accelerate_bf16-$NP.yaml
accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29005 run_finetuning_babilong_armt_hf_trainer_ct_vary.py \
        --task_dataset $TASK_DATASET \
        --noise_dataset $NOISE_DATASET \
        --babi_path /home/jovyan/rmt/babilong/data/tasks_1-20_v1-2/en-10k \
        --output_dir /home/jovyan/rmt/runs/test/babilong_multitask/$MODEL_NAME/lr_${LR}_d${D_MEM}_${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}_from_cpt_0-1_lora_ct-v3-vary/run_$N \
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

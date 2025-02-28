#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_NAME=pg19
MODEL_NAME=gpt2

SEGMENT_ORDERING=regular
SCHEDULER=linear

ITERS=10000
TBS=128

INPUT_SIZE=512

MAX_N_SEGMENTSS=(1)
BSS=(16)

for MEMORY_SIZE in 0
do 

for N in 1
do

for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 
BLOCK_SIZE=$((INPUT_SIZE-2*MEMORY_SIZE))
HISTORY_SIZE=$(((MAX_N_SEGMENTS - 1) * BLOCK_SIZE))
K2=${MAX_N_SEGMENTS}
BS=${BSS[j]}

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
LR=1e-05

NP=$NP
ACCEL_CONFIG=/home/jovyan/rmt/babilong/accel_configs/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
cd accel_configs/
python create_config.py \
        --bf16 \
        --train_batch_size $TBS\
        --train_micro_batch_size_per_gpu $BS\
        --gradient_accumulation_steps $GRAD_ACC_STEPS\
        --np $NP\
        --gradient_clipping 1.0
cd ..


echo RUNNING: TASK_NAME MEMORY_SIZE INPUT_SIZE BLOCK_SIZE HISTORY_SIZE N_SEG  MODEL_NAME MODEL_CLS LR N
echo RUNNING: $TASK_NAME $MEMORY_SIZE $INPUT_SIZE $BLOCK_SIZE $HISTORY_SIZE $MAX_N_SEGMENTS $MODEL_NAME $MODEL_CLS  $LR $N
echo gradient accumulation steps $GRAD_ACC_STEPS
accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29500 run_finetuning_pg19_rmt.py \
        --task_name $TASK_NAME \
        --model_path /home/jovyan/rmt/runs/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${BLOCK_SIZE}-${EVAL_HISTORY_SIZE}-${EVAL_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_from_cpt_cv4_eval_32k+_20kdoc_${MAX_N_SEGMENTS}-${EVAL_N_SEGMENTS}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --block_size $BLOCK_SIZE \
        --history_size $HISTORY_SIZE \
        --input_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --num_training_steps $((ITERS*2)) \
        --validate_only \
        --iters $ITERS \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/20)) \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --clip_grad_norm 1.0
        
done
done
done
done
done
done
done
echo "done"

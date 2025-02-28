#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
BACKBONE_CLS=base_models.modeling_gpt_neox:GPTNeoXForCausalLM
TASK_NAME=pile

ITERS=30000
TBS=256

INPUT_SIZE=2048

MAX_N_SEGMENTSS=(2 3 4 5 6)
BSS=(16 8 8 4 4 2)

for MEMORY_SIZE in 2
do 

for N in 1
do

for MODEL_NAME in EleutherAI/pythia-70m-deduped
do

for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 
BLOCK_SIZE=$((INPUT_SIZE-2*MEMORY_SIZE))
HISTORY_SIZE=$(((MAX_N_SEGMENTS - 1) * BLOCK_SIZE))
BS=${BSS[j]}
K2=${MAX_N_SEGMENTS}

LR=1e-05

for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do


echo RUNNING: TASK_NAME MEMORY_SIZE INPUT_SIZE BLOCK_SIZE HISTORY_SIZE N_SEG  MODEL_NAME MODEL_CLS LR N
echo RUNNING: $TASK_NAME $MEMORY_SIZE $INPUT_SIZE $BLOCK_SIZE $HISTORY_SIZE $MAX_N_SEGMENTS $MODEL_NAME $MODEL_CLS  $LR $N
echo gradient accumulation steps $(($TBS/($BS*$NP)))
accelerate launch --config_file ./accel_configs/deepspeed.yaml --main_process_port 29511 run_finetuning_pile_rmt.py \
        --task_name $TASK_NAME \
        --model_path ../runs/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${BLOCK_SIZE}-${HISTORY_SIZE}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_from_cpt_cv2_sp${SAMPLING_PROB}_$((MAX_N_SEGMENTS-1))-${MAX_N_SEGMENTS}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_cpt ../runs/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${BLOCK_SIZE}-$(((MAX_N_SEGMENTS-2)*BLOCK_SIZE))-$((MAX_N_SEGMENTS-1))x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_from_cpt_cv2_sp${SAMPLING_PROB}_$((MAX_N_SEGMENTS-2))-$((MAX_N_SEGMENTS-1))/run_$N \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --block_size $BLOCK_SIZE \
        --history_size $HISTORY_SIZE \
        --sampling_prob $SAMPLING_PROB \
        --input_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --num_training_steps $((ITERS*2)) \
        --save_best \
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

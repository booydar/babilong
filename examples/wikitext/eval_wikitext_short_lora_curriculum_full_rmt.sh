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
TASK_NAME=wikitext-2-v1

ITERS=10000
TBS=128

TGT_LEN=128
INPUT_SIZE=128

MAX_N_SEGMENTSS=(1 2 3 4 5 6 7 )
BSS=(128 64 32 32 32 16)

MEMORY_SIZE=2

for N in 1
do

for MODEL_NAME in gpt2
do

for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 
INPUT_SEQ_LEN=$(((INPUT_SIZE-2*MEMORY_SIZE)*MAX_N_SEGMENTS))
BS=${BSS[j]}

for SOURCE_N_SEGMENTS in 1 2 3 4 5 6
do
K2=4

for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do

for LR in 1e-04
do


echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
accelerate launch --num_processes $NP --config_file ./accelerate.yaml run_finetuning_lm_rmt.py \
        --task_name $TASK_NAME \
        --model_path ../runs/lm/${TASK_NAME}/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${TGT_LEN}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_lora_freeze_from_cpt_full_$((SOURCE_N_SEGMENTS))-${MAX_N_SEGMENTS}_eval/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --model_cpt ../runs/lm/${TASK_NAME}/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_$(((INPUT_SIZE-2*MEMORY_SIZE)*SOURCE_N_SEGMENTS))-${TGT_LEN}-${SOURCE_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs32_${SEGMENT_ORDERING}_bptt-${K2}_lora_freeze_from_cpt_full_0-${SOURCE_N_SEGMENTS}/run_$N \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size $INPUT_SIZE \
        --target_seq_len $TGT_LEN \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --k1 -1 --k2 $K2 \
        --use_lora \
        --validate_only \
        --freeze_model_weights \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/10)) \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
        
done
done
done
done
done
done
done
echo "done"

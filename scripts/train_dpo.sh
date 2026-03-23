#!/bin/bash

# DPO Training Script
# Sử dụng: bash scripts/train_dpo.sh

# Set default values
MODEL_NAME=${MODEL_NAME:-"./outputs/sft"}  # Usually the SFT model
DATASET_NAME=${DATASET_NAME:-"HuggingFaceH4/ultrafeedback_binarized"}
DATASET_SPLIT=${DATASET_SPLIT:-"train_prefs"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/dpo"}
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LEARNING_RATE=${LEARNING_RATE:-5e-7}
BETA=${BETA:-0.1}
LOSS_TYPE=${LOSS_TYPE:-"sigmoid"}
MAX_LENGTH=${MAX_LENGTH:-1024}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
USE_LORA=${USE_LORA:-true}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}

echo "=================================================="
echo "DPO Training Configuration"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRAD_ACCUM"
echo "Learning Rate: $LEARNING_RATE"
echo "Beta: $BETA"
echo "Loss Type: $LOSS_TYPE"
echo "Max Length: $MAX_LENGTH"
echo "Max Prompt Length: $MAX_PROMPT_LENGTH"
echo "Use LoRA: $USE_LORA"
echo "LoRA Rank: $LORA_R"
echo "LoRA Alpha: $LORA_ALPHA"
echo "=================================================="

# Build command
CMD="python src/trainers/train_dpo.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --dataset_split $DATASET_SPLIT \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --beta $BETA \
    --loss_type $LOSS_TYPE \
    --max_length $MAX_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH"

# Add LoRA if enabled
if [ "$USE_LORA" = true ]; then
    CMD="$CMD --use_lora \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA"
fi

# Add optional arguments if set
if [ ! -z "$DATASET_CONFIG" ]; then
    CMD="$CMD --dataset_config $DATASET_CONFIG"
fi

if [ ! -z "$EVAL_DATASET" ]; then
    CMD="$CMD --eval_dataset_name $EVAL_DATASET"
fi

if [ ! -z "$REPORT_TO" ]; then
    CMD="$CMD --report_to $REPORT_TO"
fi

# Run training
echo ""
echo "Starting training..."
echo ""
eval $CMD

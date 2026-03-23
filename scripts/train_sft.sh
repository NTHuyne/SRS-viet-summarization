#!/bin/bash

# SFT Training Script
# Sử dụng: bash scripts/train_sft.sh

# Set default values
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-0.5B"}
DATASET_NAME=${DATASET_NAME:-"HuggingFaceH4/ultrachat_200k"}
DATASET_SPLIT=${DATASET_SPLIT:-"train_sft"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/sft"}
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}
USE_LORA=${USE_LORA:-true}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}

echo "=================================================="
echo "SFT Training Configuration"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRAD_ACCUM"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Sequence Length: $MAX_SEQ_LENGTH"
echo "Use LoRA: $USE_LORA"
echo "LoRA Rank: $LORA_R"
echo "LoRA Alpha: $LORA_ALPHA"
echo "=================================================="

# Build command
CMD="python src/trainers/train_sft.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --dataset_split $DATASET_SPLIT \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH"

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

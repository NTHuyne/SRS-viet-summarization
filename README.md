# LLM Training Framework with TRL

A complete framework for training LLMs using SFT, DPO, CPO, and KTO methods with TRL and Transformers.

## Installation

```bash
# Clone repository
git clone <repo_url>
cd SRS-viet-summarization

# Install dependencies
pip install -r requirements.txt

# Or with uv (faster)
uv pip install -r requirements.txt
```

## Directory Structure

```
.
├── src/
│   ├── configs/          # Training configurations
│   │   ├── sft_config.py
│   │   ├── dpo_config.py
│   │   ├── cpo_config.py
│   │   ├── kto_config.py
│   │   └── lora_config.py
│   ├── trainers/         # Training scripts
│   │   ├── train_sft.py
│   │   ├── train_dpo.py
│   │   └── ...
│   ├── data/            # Data utilities
│   │   └── dataset_loader.py
│   └── utils/           # Utilities
│       └── model_utils.py
├── scripts/             # Bash scripts
│   ├── train_sft.sh
│   ├── train_dpo.sh
│   └── ...
└── outputs/            # Model outputs
    ├── sft/
    ├── dpo/
    └── ...
```

## Usage

### 1. SFT Training

**Option 1: Using bash script (Recommended)**

```bash
# Basic training
bash scripts/train_sft.sh

# Custom parameters via environment variables
MODEL_NAME="Qwen/Qwen2.5-1.5B" \
DATASET_NAME="HuggingFaceH4/ultrachat_200k" \
OUTPUT_DIR="./outputs/my_sft_model" \
NUM_EPOCHS=5 \
BATCH_SIZE=8 \
USE_LORA=true \
bash scripts/train_sft.sh
```

**Option 2: Run Python script directly**

```bash
python src/trainers/train_sft.py \
    --model_name_or_path "Qwen/Qwen2.5-0.5B" \
    --dataset_name "HuggingFaceH4/ultrachat_200k" \
    --dataset_split "train_sft" \
    --output_dir "./outputs/sft" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --max_seq_length 2048 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32
```

### 2. DPO Training

**Option 1: Using bash script**

```bash
# Training with an SFT-trained model
MODEL_NAME="./outputs/sft" \
DATASET_NAME="HuggingFaceH4/ultrafeedback_binarized" \
OUTPUT_DIR="./outputs/dpo" \
BETA=0.1 \
bash scripts/train_dpo.sh
```

**Option 2: Python script**

```bash
python src/trainers/train_dpo.py \
    --model_name_or_path "./outputs/sft" \
    --dataset_name "HuggingFaceH4/ultrafeedback_binarized" \
    --dataset_split "train_prefs" \
    --output_dir "./outputs/dpo" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --loss_type "sigmoid" \
    --use_lora
```

## Configuration

### Environment Variables for Bash Scripts

#### SFT Training
- `MODEL_NAME`: Model path/name (default: "Qwen/Qwen2.5-0.5B")
- `DATASET_NAME`: Dataset name (default: "HuggingFaceH4/ultrachat_200k")
- `OUTPUT_DIR`: Output directory (default: "./outputs/sft")
- `NUM_EPOCHS`: Training epochs (default: 3)
- `BATCH_SIZE`: Batch size per device (default: 4)
- `GRAD_ACCUM`: Gradient accumulation steps (default: 4)
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `MAX_SEQ_LENGTH`: Max sequence length (default: 2048)
- `USE_LORA`: Use LoRA (default: true)
- `LORA_R`: LoRA rank (default: 16)
- `LORA_ALPHA`: LoRA alpha (default: 32)

#### DPO Training
- `MODEL_NAME`: SFT model path (default: "./outputs/sft")
- `DATASET_NAME`: Preference dataset (default: "HuggingFaceH4/ultrafeedback_binarized")
- `BETA`: DPO beta parameter (default: 0.1)
- `LOSS_TYPE`: Loss type (default: "sigmoid")
- `MAX_LENGTH`: Max sequence length (default: 1024)
- `MAX_PROMPT_LENGTH`: Max prompt length (default: 512)

### LoRA Presets

```python
# Small models (< 3B params)
LORA_R=8, LORA_ALPHA=16

# Medium models (3B-13B params)
LORA_R=16, LORA_ALPHA=32

# Large models (> 13B params)
LORA_R=32, LORA_ALPHA=64
```

## Datasets

### SFT Datasets
Expected format:
```json
{
  "text": "Complete training text..."
}
```

Or conversational format:
```json
{
  "messages": [
    {"role": "user", "content": "Question?"},
    {"role": "assistant", "content": "Answer."}
  ]
}
```

### Preference Datasets (DPO/CPO)
Expected format:
```json
{
  "prompt": "User question",
  "chosen": "Better response",
  "rejected": "Worse response"
}
```

### KTO Datasets
Expected format:
```json
{
  "prompt": "User question",
  "completion": "Model response",
  "label": true  // true for good, false for bad
}
```

## Tips

### Memory Optimization
1. **Reduce batch size**: `BATCH_SIZE=2`
2. **Increase gradient accumulation**: `GRAD_ACCUM=8`
3. **Reduce max sequence length**: `MAX_SEQ_LENGTH=1024`
4. **Use LoRA**: `USE_LORA=true`
5. **Gradient checkpointing**: Enabled automatically

### Training Tips
1. **SFT first, then DPO/CPO/KTO**
2. **Learning rate for DPO/CPO/KTO should be lower than SFT** (5e-7 vs 2e-5)
3. **Beta parameter**: 0.1 is a good default; increase it if you want less deviation from the reference model
4. **LoRA rank**: 8–16 for small models, 16–32 for medium/large models

### With WandB Logging

```bash
REPORT_TO="wandb" bash scripts/train_sft.sh
```

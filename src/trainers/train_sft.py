"""
SFT (Supervised Fine-Tuning) Trainer
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import HfArgumentParser
from trl import SFTTrainer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.sft_config import get_sft_config
from configs.lora_config import get_lora_config
from utils.model_utils import load_model, load_tokenizer
from data.dataset_loader import load_sft_dataset


@dataclass
class ScriptArguments:
    """Arguments for the training script"""
    
    # Model arguments
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code when loading model/tokenizer"}
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={"help": "Use Flash Attention 2"}
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "Torch dtype (auto, bfloat16, float16, float32)"}
    )
    
    # LoRA arguments
    use_lora: bool = field(
        default=False,
        metadata={"help": "Use LoRA for efficient fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    
    # Dataset arguments
    dataset_name: str = field(
        metadata={"help": "Dataset name or path"}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset configuration name"}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"}
    )
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Field name containing text in dataset"}
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Evaluation dataset name (optional)"}
    )
    eval_dataset_split: str = field(
        default="test",
        metadata={"help": "Evaluation dataset split"}
    )
    
    # Training arguments (overrides)
    output_dir: str = field(
        default="./outputs/sft",
        metadata={"help": "Output directory"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing"}
    )
    report_to: str = field(
        default="none",
        metadata={"help": "Where to report metrics (wandb, tensorboard, none)"}
    )


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    print("="*50)
    print("SFT Training with TRL")
    print("="*50)
    print(f"Model: {script_args.model_name_or_path}")
    print(f"Dataset: {script_args.dataset_name}")
    print(f"Output: {script_args.output_dir}")
    print(f"LoRA: {script_args.use_lora}")
    print("="*50)
    
    # Load model
    print("\n[1/5] Loading model...")
    model = load_model(
        model_name_or_path=script_args.model_name_or_path,
        use_flash_attention_2=script_args.use_flash_attention_2,
        torch_dtype=script_args.torch_dtype,
        trust_remote_code=script_args.trust_remote_code,
    )
    
    # Load tokenizer
    print("[2/5] Loading tokenizer...")
    tokenizer = load_tokenizer(
        model_name_or_path=script_args.model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
    )
    
    # Load dataset
    print("[3/5] Loading dataset...")
    train_dataset = load_sft_dataset(
        dataset_name=script_args.dataset_name,
        dataset_config=script_args.dataset_config,
        split=script_args.dataset_split,
        text_field=script_args.dataset_text_field,
    )
    
    eval_dataset = None
    if script_args.eval_dataset_name:
        eval_dataset = load_sft_dataset(
            dataset_name=script_args.eval_dataset_name,
            split=script_args.eval_dataset_split,
        )
    
    print(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Prepare LoRA config if needed
    print("[4/5] Preparing training configuration...")
    peft_config = None
    if script_args.use_lora:
        peft_config = get_lora_config(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
        )
        print(f"LoRA config: r={script_args.lora_r}, alpha={script_args.lora_alpha}")
    
    # Get training config
    training_args = get_sft_config(
        output_dir=script_args.output_dir,
        num_train_epochs=script_args.num_train_epochs,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        max_seq_length=script_args.max_seq_length,
        packing=script_args.packing,
        dataset_text_field=script_args.dataset_text_field,
        report_to=script_args.report_to,
    )
    
    # Initialize trainer
    print("[5/5] Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    trainer.train()
    
    # Save model
    print("\n" + "="*50)
    print("Saving model...")
    print("="*50)
    trainer.save_model(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)
    
    print(f"\n✓ Training completed! Model saved to {script_args.output_dir}")


if __name__ == "__main__":
    main()

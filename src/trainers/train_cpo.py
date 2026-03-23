"""
CPO (Contrastive Preference Optimization) Trainer
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import HfArgumentParser
from trl.experimental.cpo import CPOTrainer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.cpo_config import get_cpo_config
from configs.lora_config import get_lora_config
from utils.model_utils import load_model, load_tokenizer
from data.dataset_loader import load_preference_dataset


@dataclass
class ScriptArguments:
    """Arguments for CPO training script"""
    
    # Model arguments
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model (usually SFT model)"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code"}
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={"help": "Use Flash Attention 2"}
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "Torch dtype"}
    )
    
    # LoRA arguments
    use_lora: bool = field(
        default=False,
        metadata={"help": "Use LoRA"}
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    
    # Dataset arguments
    dataset_name: str = field(
        metadata={"help": "Preference dataset name"}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset config"}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split"}
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Eval dataset"}
    )
    eval_dataset_split: str = field(
        default="test",
        metadata={"help": "Eval split"}
    )
    
    # Training arguments
    output_dir: str = field(
        default="./outputs/cpo",
        metadata={"help": "Output directory"}
    )
    num_train_epochs: int = field(default=3, metadata={"help": "Epochs"})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Grad accum"})
    learning_rate: float = field(default=5e-7, metadata={"help": "Learning rate"})
    beta: float = field(default=0.1, metadata={"help": "CPO beta parameter"})
    loss_type: str = field(default="sigmoid", metadata={"help": "CPO loss type"})
    max_length: int = field(default=1024, metadata={"help": "Max length"})
    max_prompt_length: int = field(default=512, metadata={"help": "Max prompt length"})
    report_to: str = field(default="none", metadata={"help": "Reporting"})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    print("="*50)
    print("CPO Training with TRL")
    print("="*50)
    print(f"Model: {script_args.model_name_or_path}")
    print(f"Dataset: {script_args.dataset_name}")
    print(f"Output: {script_args.output_dir}")
    print(f"Beta: {script_args.beta}")
    print(f"Loss: {script_args.loss_type}")
    print("="*50)
    
    # Load model
    print("\n[1/4] Loading model...")
    model = load_model(
        model_name_or_path=script_args.model_name_or_path,
        use_flash_attention_2=script_args.use_flash_attention_2,
        torch_dtype=script_args.torch_dtype,
        trust_remote_code=script_args.trust_remote_code,
    )
    
    # Load tokenizer
    print("[2/4] Loading tokenizer...")
    tokenizer = load_tokenizer(
        model_name_or_path=script_args.model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
    )
    
    # Load dataset
    print("[3/4] Loading preference dataset...")
    train_dataset = load_preference_dataset(
        dataset_name=script_args.dataset_name,
        dataset_config=script_args.dataset_config,
        split=script_args.dataset_split,
    )
    
    eval_dataset = None
    if script_args.eval_dataset_name:
        eval_dataset = load_preference_dataset(
            dataset_name=script_args.eval_dataset_name,
            split=script_args.eval_dataset_split,
        )
    
    print(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Prepare configs
    print("[4/4] Preparing training configuration...")
    peft_config = None
    if script_args.use_lora:
        peft_config = get_lora_config(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
        )
    
    training_args = get_cpo_config(
        output_dir=script_args.output_dir,
        num_train_epochs=script_args.num_train_epochs,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        beta=script_args.beta,
        loss_type=script_args.loss_type,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
        report_to=script_args.report_to,
    )
    
    # Initialize trainer
    print("\nInitializing CPO trainer...")
    trainer = CPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    trainer.train()
    
    # Save
    print("\n" + "="*50)
    print("Saving model...")
    print("="*50)
    trainer.save_model(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)
    
    print(f"\n✓ Training completed! Model saved to {script_args.output_dir}")


if __name__ == "__main__":
    main()

"""
SFT (Supervised Fine-Tuning) configuration
"""
from trl import SFTConfig


def get_sft_config(
    output_dir: str = "./outputs/sft",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    max_seq_length: int = 2048,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    optim: str = "adamw_torch_fused",
    bf16: bool = True,
    fp16: bool = False,
    gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: dict = None,
    packing: bool = False,
    dataset_text_field: str = "text",
    max_steps: int = -1,
    save_total_limit: int = 3,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "loss",
    greater_is_better: bool = False,
    report_to: str = "none",
    **kwargs
):
    """
    Get SFT configuration for training
    
    Args:
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        warmup_ratio: Warmup ratio for learning rate scheduler
        lr_scheduler_type: Type of learning rate scheduler
        optim: Optimizer type
        bf16: Use bfloat16 precision
        fp16: Use float16 precision
        gradient_checkpointing: Enable gradient checkpointing
        gradient_checkpointing_kwargs: Additional kwargs for gradient checkpointing
        packing: Enable sequence packing for efficiency
        dataset_text_field: Field name containing text in dataset
        max_steps: Maximum number of training steps (overrides num_train_epochs)
        save_total_limit: Maximum number of checkpoints to keep
        load_best_model_at_end: Load best model at end of training
        metric_for_best_model: Metric to use for best model selection
        greater_is_better: Whether higher metric is better
        report_to: Where to report metrics (wandb, tensorboard, none)
        **kwargs: Additional arguments for SFTConfig
    
    Returns:
        SFTConfig object
    """
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": False}
    
    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        packing=packing,
        dataset_text_field=dataset_text_field,
        max_steps=max_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        report_to=report_to,
        **kwargs
    )
    
    return config


# Preset configurations
SFT_PRESETS = {
    "debug": {
        "num_train_epochs": 1,
        "max_steps": 10,
        "logging_steps": 1,
        "save_steps": 5,
        "eval_steps": 5,
    },
    "quick": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-5,
    },
    "standard": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
    },
    "long": {
        "num_train_epochs": 5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-5,
    },
}

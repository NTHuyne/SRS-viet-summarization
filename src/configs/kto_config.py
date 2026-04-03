"""
KTO (Kahneman-Tversky Optimization) configuration
"""
from trl.experimental.kto import KTOConfig


def get_kto_config(
    output_dir: str = "./outputs/kto",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-6,
    max_length: int = 1024,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    warmup_steps: float = 0.1,
    lr_scheduler_type: str = "cosine",
    optim: str = "adamw_torch_fused",
    bf16: bool = True,
    fp16: bool = False,
    gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: dict = {},
    beta: float = 0.1,
    loss_type: str = "kto",
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    max_steps: int = -1,
    save_total_limit: int = 3,
    report_to: str = "none",
    **kwargs
):
    """
    Get KTO configuration for training
    
    Args:
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device (minimum 4 recommended)
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate (5e-7 to 5e-6 recommended)
        max_length: Maximum sequence length
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        warmup_steps: Warmup ratio for learning rate scheduler
        lr_scheduler_type: Type of learning rate scheduler
        optim: Optimizer type
        bf16: Use bfloat16 precision
        fp16: Use float16 precision
        gradient_checkpointing: Enable gradient checkpointing
        gradient_checkpointing_kwargs: Additional kwargs for gradient checkpointing
        beta: KTO beta parameter (lower beta = less deviation from ref model)
        loss_type: KTO loss type (kto or apo_zero_unpaired)
        desirable_weight: Weight for desirable/positive examples
        undesirable_weight: Weight for undesirable/negative examples
        max_steps: Maximum number of training steps
        save_total_limit: Maximum number of checkpoints to keep
        report_to: Where to report metrics (wandb, tensorboard, none)
        **kwargs: Additional arguments for KTOConfig
    
    Returns:
        KTOConfig object
    """
    if gradient_checkpointing_kwargs is {}:
        gradient_checkpointing_kwargs = {"use_reentrant": False}
    
    config = KTOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_length=max_length,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        beta=beta,
        loss_type=loss_type,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        max_steps=max_steps,
        save_total_limit=save_total_limit,
        report_to=report_to,
        **kwargs
    )
    
    return config


# Preset configurations
KTO_PRESETS = {
    "debug": {
        "num_train_epochs": 1,
        "max_steps": 10,
        "logging_steps": 1,
        "save_steps": 5,
        "eval_steps": 5,
    },
    "balanced": {
        "learning_rate": 1e-6,
        "beta": 0.1,
        "desirable_weight": 1.0,
        "undesirable_weight": 1.0,
    },
    "more_positives": {
        "learning_rate": 1e-6,
        "beta": 0.1,
        "desirable_weight": 1.0,
        "undesirable_weight": 2.0,  # Upweight less common negatives
    },
    "more_negatives": {
        "learning_rate": 1e-6,
        "beta": 0.1,
        "desirable_weight": 2.0,  # Upweight less common positives
        "undesirable_weight": 1.0,
    },
}

# Common loss types for KTO
KTO_LOSS_TYPES = [
    "kto",                  # Standard KTO loss
    "apo_zero_unpaired",    # APO-zero for unpaired data
]

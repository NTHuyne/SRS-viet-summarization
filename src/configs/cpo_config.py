"""
CPO (Contrastive Preference Optimization) configuration
"""
from trl.experimental.cpo import CPOConfig

def get_cpo_config(
    output_dir: str = "./outputs/cpo",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-7,
    max_length: int = 1024,
    max_prompt_length: int = 512,
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
    loss_type: str = "sigmoid",
    cpo_alpha: float = 1.0,
    max_steps: int = -1,
    save_total_limit: int = 3,
    report_to: str = "none",
    **kwargs
):
    """
    Get CPO configuration for training
    
    Args:
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate (lower than SFT)
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
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
        beta: CPO beta parameter
        loss_type: CPO loss type (sigmoid, hinge, ipo, simpo, etc.)
        cpo_alpha: BC regularization weight (0 for SimPO)
        max_steps: Maximum number of training steps
        save_total_limit: Maximum number of checkpoints to keep
        report_to: Where to report metrics (wandb, tensorboard, none)
        **kwargs: Additional arguments for CPOConfig
    
    Returns:
        CPOConfig object
    """
    if gradient_checkpointing_kwargs is {}:
        gradient_checkpointing_kwargs = {"use_reentrant": False}
    
    config = CPOConfig(
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
        cpo_alpha=cpo_alpha,
        max_steps=max_steps,
        save_total_limit=save_total_limit,
        report_to=report_to,
        **kwargs
    )
    
    return config


# Preset configurations
CPO_PRESETS = {
    "debug": {
        "num_train_epochs": 1,
        "max_steps": 10,
        "logging_steps": 1,
        "save_steps": 5,
        "eval_steps": 5,
    },
    "cpo": {
        "loss_type": "sigmoid",
        "cpo_alpha": 1.0,
        "learning_rate": 5e-7,
    },
    "simpo": {
        "loss_type": "simpo",
        "cpo_alpha": 0.0,
        "learning_rate": 1e-6,
    },
    "cpo_simpo": {
        "loss_type": "simpo",
        "cpo_alpha": 1.0,
        "learning_rate": 5e-7,
    },
}

# Common loss types for CPO
CPO_LOSS_TYPES = [
    "sigmoid",      # Standard CPO loss
    "hinge",        # Hinge loss variant
    "ipo",          # Identity Policy Optimization
    "simpo",        # SimPO (no reference model)
    "alpha_simpo",  # AlphaPO variant
]

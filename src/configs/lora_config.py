"""
LoRA configuration for efficient fine-tuning
"""
from peft import LoraConfig


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
):
    """
    Get LoRA configuration for training
    
    Args:
        r: LoRA rank (8, 16, 32, 64)
        lora_alpha: LoRA scaling factor (usually r * 2)
        lora_dropout: Dropout probability
        target_modules: List of modules to apply LoRA
        bias: Bias type ("none", "all", "lora_only")
        task_type: Task type for PEFT
    
    Returns:
        LoraConfig object
    """
    if target_modules is None:
        # Default for Llama/Qwen/Mistral/Gemma architectures
        target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
    )


# Preset configurations for different model sizes
LORA_PRESETS = {
    "small": {  # For models < 3B parameters
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
    "medium": {  # For models 3B-13B parameters
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    },
    "large": {  # For models > 13B parameters
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
    },
}

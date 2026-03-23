"""
Model utilities for loading and preparing models
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training


def load_model(
    model_name_or_path: str,
    use_flash_attention_2: bool = True,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = False,
    use_4bit: bool = False,
    use_8bit: bool = False,
    **kwargs
):
    """
    Load a causal language model
    
    Args:
        model_name_or_path: Model name or path
        use_flash_attention_2: Use Flash Attention 2
        torch_dtype: Torch dtype (auto, bfloat16, float16, float32)
        device_map: Device map for model parallelism
        trust_remote_code: Trust remote code
        use_4bit: Use 4-bit quantization (QLoRA)
        use_8bit: Use 8-bit quantization
        **kwargs: Additional arguments for AutoModelForCausalLM
    
    Returns:
        Loaded model
    """
    # Prepare quantization config if needed
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Convert torch_dtype string to torch dtype
    if torch_dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch_dtype
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2" if use_flash_attention_2 else None,
        **kwargs
    )
    
    # Prepare for k-bit training if quantized
    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    return model


def load_tokenizer(
    model_name_or_path: str,
    padding_side: str = "right",
    trust_remote_code: bool = False,
    add_eos_token: bool = False,
    add_bos_token: bool = False,
    use_fast: bool = True,
    **kwargs
):
    """
    Load tokenizer with proper configuration
    
    Args:
        model_name_or_path: Model name or path
        padding_side: Padding side (left or right)
        trust_remote_code: Trust remote code
        add_eos_token: Add EOS token
        add_bos_token: Add BOS token
        use_fast: Use fast tokenizer
        **kwargs: Additional arguments for AutoTokenizer
    
    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side=padding_side,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
        **kwargs
    )
    
    # Set special tokens if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if add_eos_token:
        tokenizer.add_eos_token = True
    
    if add_bos_token:
        tokenizer.add_bos_token = True
    
    return tokenizer


def get_model_config(model_name_or_path: str):
    """
    Get model configuration to determine architecture-specific settings
    
    Args:
        model_name_or_path: Model name or path
    
    Returns:
        Dictionary with model info
    """
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    model_type = config.model_type.lower()
    
    # Determine recommended LoRA target modules based on architecture
    target_modules = None
    if model_type in ["llama", "mistral", "qwen2", "gemma", "phi"]:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    elif model_type in ["gpt2", "gpt_neo", "gpt_neox"]:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    elif model_type == "bloom":
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    
    return {
        "model_type": model_type,
        "hidden_size": config.hidden_size,
        "num_layers": getattr(config, "num_hidden_layers", None),
        "vocab_size": config.vocab_size,
        "recommended_lora_targets": target_modules,
    }

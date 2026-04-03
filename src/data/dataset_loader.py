"""
Dataset loading and preprocessing utilities
"""
from datasets import load_dataset, Dataset, DatasetDict
from typing import Optional, Union

from transformers import data


def load_sft_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    split: Optional[str] = None,
    text_field: str = "text",
    num_proc: Optional[int] = None,
    streaming: bool = False,
):
    """
    Load dataset for SFT training
    
    Args:
        dataset_name: Dataset name or path
        dataset_config: Dataset configuration name
        split: Dataset split to load
        text_field: Field name containing text
        num_proc: Number of processes for preprocessing
        streaming: Use streaming mode
    
    Returns:
        Loaded dataset
    """
    if dataset_config:
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming,
            num_proc=num_proc
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            num_proc=num_proc
        )

    return dataset


def load_preference_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    split: Optional[str] = None,
    num_proc: Optional[int] = None,
    streaming: bool = False,
):
    """
    Load preference dataset for DPO/CPO training
    
    Expected format:
    - Standard: {"prompt": str, "chosen": str, "rejected": str}
    - Conversational: {"prompt": List[dict], "chosen": List[dict], "rejected": List[dict]}
    
    Args:
        dataset_name: Dataset name or path
        dataset_config: Dataset configuration name
        split: Dataset split to load
        num_proc: Number of processes for preprocessing
        streaming: Use streaming mode
    
    Returns:
        Loaded dataset
    """
    if dataset_config:
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming,
            num_proc=num_proc
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            num_proc=num_proc
        )

    if not streaming:
        dataset = dataset.map(
            format_to_dpo_chat, 
        )

    dataset = dataset.shuffle(seed=42)
    
    return dataset


def load_kto_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    split: Optional[str] = None,
    num_proc: Optional[int] = None,
    streaming: bool = False,
    convert_from_preference: bool = False,
):
    """
    Load dataset cho KTO training và chuyển sang format hội thoại
    """
    if dataset_config:
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming,
            num_proc=num_proc
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            num_proc=num_proc
        )
    
    if convert_from_preference and not streaming:
        dataset = convert_preference_to_kto(dataset)
    
    if not streaming:
        dataset = dataset.map(format_to_kto_chat)
    
    return dataset

def convert_preference_to_kto(dataset: Union[Dataset, DatasetDict]):
    """
    Convert preference dataset to KTO format
    
    Converts:
    {"prompt": str, "chosen": str, "rejected": str}
    To:
    {"prompt": str, "completion": str, "label": bool}
    
    Args:
        dataset: Preference dataset
    
    Returns:
        KTO format dataset
    """
    if isinstance(dataset, DatasetDict):
        return DatasetDict({
            split: convert_preference_to_kto(ds)
            for split, ds in dataset.items()
        })
    
    # Create two rows for each preference pair
    kto_data = {
        "prompt": [],
        "completion": [],
        "label": [],
    }
    
    for example in dataset:
        # Add chosen as positive
        kto_data["prompt"].append(example["prompt"])
        kto_data["completion"].append(example["chosen"])
        kto_data["label"].append(True)
        
        # Add rejected as negative
        kto_data["prompt"].append(example["prompt"])
        kto_data["completion"].append(example["rejected"])
        kto_data["label"].append(False)
    
    return Dataset.from_dict(kto_data)


def prepare_conversational_dataset(
    dataset: Dataset,
    tokenizer,
    apply_chat_template: bool = True,
):
    """
    Prepare conversational dataset with chat template
    
    Args:
        dataset: Dataset with messages format
        tokenizer: Tokenizer with chat template
        apply_chat_template: Apply chat template
    
    Returns:
        Processed dataset
    """
    if not apply_chat_template:
        return dataset
    
    def apply_template(example):
        # Handle different message formats
        if "messages" in example:
            example["text"] = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
        elif "prompt" in example and isinstance(example["prompt"], list):
            example["prompt"] = tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=True
            )
        
        return example
    
    dataset = dataset.map(apply_template, num_proc=4)
    return dataset


def format_to_dpo_chat(example):
    """
    Chuyển đổi string prompt/chosen/rejected sang định dạng list of dict
    """
    # Đảm bảo dữ liệu là string và xóa khoảng trắng thừa
    prompt_str = str(example["prompt"]).strip()
    chosen_str = str(example["chosen"]).strip()
    rejected_str = str(example["rejected"]).strip()

    # Tạo format prompt (User)
    # Lưu ý: Nếu prompt của bạn đã là list rồi thì có thể bỏ qua bước bọc [ ]
    formatted_prompt = [{"role": "user", "content": prompt_str}]
    
    # Tạo format chosen/rejected (Assistant)
    formatted_chosen = [{"role": "assistant", "content": chosen_str}]
    formatted_rejected = [{"role": "assistant", "content": rejected_str}]
    
    return {
        "prompt": formatted_prompt,
        "chosen": formatted_chosen,
        "rejected": formatted_rejected
    }


def format_to_kto_chat(example):

    prompt_str = str(example["prompt"]).strip()
    completion_str = str(example["completion"]).strip()

    return {
        "prompt": [{"role": "user", "content": prompt_str}],
        "completion": [{"role": "assistant", "content": completion_str}],
        "label": example["label"]
    }
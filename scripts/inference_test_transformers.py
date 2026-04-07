#!/usr/bin/env python3
"""
Script để chạy inference trên test set sử dụng Transformers trực tiếp
Hỗ trợ batch processing và multithread để tăng tốc độ
"""

import argparse
import json
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ModelInference:
    """Wrapper class để thread-safe inference"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        print(f"Đang load model từ: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            device_map = "auto"
        else:
            self.device = device
            device_map = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map
        )
        
        # Lock để đảm bảo thread-safe khi gọi model.generate
        self.lock = threading.Lock()
        
        print(f"Model loaded on device: {self.device}")
    
    def generate_single(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int
    ) -> str:
        """Generate cho một prompt"""
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Use lock để đảm bảo thread-safe
        with self.lock:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        # Extract generated tokens (bỏ input tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        predicted = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        return predicted


def load_test_data(test_file: str) -> List[Dict]:
    """Load dữ liệu test từ file JSONL"""
    data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_prompt(document: str) -> str:
    """Tạo prompt theo template đã training"""
    template = "Hãy tóm tắt đoạn văn bản tiếng Việt sau đây một cách ngắn gọn.\n\nVăn bản cần tóm tắt:\n{document}\n\nTóm tắt:"
    return template.format(document=document)


def process_single_item(
    item: Dict,
    model_inference: ModelInference,
    temperature: float,
    top_p: float,
    max_new_tokens: int
) -> Dict:
    """Xử lý một sample"""
    input_text = item.get("input", "")
    output_label = item.get("output", "")
    
    prompt = create_prompt(input_text)
    
    try:
        predicted = model_inference.generate_single(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens
        )
    except Exception as e:
        print(f"\nLỗi khi generate: {e}")
        predicted = ""
    
    return {
        "input": input_text,
        "output": output_label,
        "predicted": predicted
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inference trên test set với Transformers (multithread)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Đường dẫn đến model đã fine-tune"
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--test-file",
        type=str,
        default="datasets-test/test.jsonl",
        help="Đường dẫn đến file test (default: datasets-test/test.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Thư mục lưu kết quả (default: results)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Tên file output (không có extension). Nếu không set, sẽ dùng tên model + timestamp"
    )
    
    # Generation arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature cho sampling (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus sampling) (default: 0.9)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Số tokens tối đa cho summary (default: 512)"
    )
    
    # Parallel processing arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Số samples xử lý trong một batch (default: 16)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Số threads song song (default: 4)"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device để chạy model: 'auto', 'cuda', 'cpu' (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tạo tên file output
    if args.output_name:
        output_file = os.path.join(args.output_dir, f"{args.output_name}.csv")
    else:
        model_name = os.path.basename(args.model_path.rstrip('/'))
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"{model_name}_{current_datetime}.csv")
    
    print("=" * 80)
    print("Cấu hình Inference với Transformers (Multithread)")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Test file: {args.test_file}")
    print(f"Output file: {output_file}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max workers: {args.max_workers}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load model
    print("\nĐang load model...")
    model_inference = ModelInference(args.model_path, args.device)
    
    # Load test data
    print("\nĐang load dữ liệu test...")
    test_data = load_test_data(args.test_file)
    print(f"Đã load {len(test_data)} samples")
    
    # Chạy inference với batch processing và multithread
    print("\nĐang chạy inference với multithread...")
    
    # Chia data thành batches
    batches = [
        test_data[i:i+args.batch_size] 
        for i in range(0, len(test_data), args.batch_size)
    ]
    
    all_results = []
    
    # Progress bar
    with tqdm(total=len(test_data), desc="Generating summaries") as pbar:
        for batch in batches:
            # Xử lý batch với multithread
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_item,
                        item,
                        model_inference,
                        args.temperature,
                        args.top_p,
                        args.max_new_tokens
                    ): item for item in batch
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        print(f"\nLỗi khi xử lý item: {e}")
                        pbar.update(1)
    
    # Lưu kết quả vào CSV
    print("\nĐang lưu kết quả vào CSV...")
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print("\n" + "=" * 80)
    print("Hoàn thành!")
    print("=" * 80)
    print(f"Đã xử lý: {len(all_results)} samples")
    print(f"Kết quả được lưu tại: {output_file}")
    print("=" * 80)
    
    # Hiển thị 2 ví dụ đầu tiên
    if len(all_results) > 0:
        print("\nVí dụ kết quả (2 samples đầu tiên):")
        print("-" * 80)
        for i, result in enumerate(all_results[:2], 1):
            print(f"\nSample {i}:")
            print(f"Input: {result['input'][:100]}...")
            print(f"Label: {result['output'][:100]}...")
            print(f"Predicted: {result['predicted'][:100]}...")
        print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script để chạy inference trên test set sử dụng vLLM OpenAI API
Hỗ trợ batch processing và parallel execution để tăng tốc độ
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def generate_summary(
    client: OpenAI,
    prompt: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int
) -> str:
    """Gọi API để sinh summary"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\nLỗi khi gọi API: {e}")
        return ""


def process_single_item(
    item: Dict,
    client: OpenAI,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int
) -> Dict:
    """Xử lý một sample và trả về kết quả"""
    # Lấy input (article)
    input_text = item.get("input", "")
    output_label = item.get("output", "")
    
    # Tạo prompt
    prompt = create_prompt(input_text)
    
    # Generate summary
    predicted = generate_summary(
        client=client,
        prompt=prompt,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    # Tạo result object
    result = {
        "input": input_text,
        "output": output_label,
        "predicted": predicted
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Inference trên test set với vLLM API")
    
    # API arguments
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL của vLLM API server (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Tên model để dùng cho inference (cần khớp với --served-model-name khi serve)"
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
        "--max-tokens",
        type=int,
        default=512,
        help="Số tokens tối đa cho summary (default: 512)"
    )
    
    # Batch processing arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Số samples xử lý đồng thời trong một batch (default: 16)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Số threads song song để gọi API (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tạo tên file output: model-name_YYYYMMDD.jsonl
    current_date = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(
        args.output_dir,
        f"{args.model_name}_{current_date}.jsonl"
    )
    
    print("=" * 80)
    print("Cấu hình Inference:")
    print("=" * 80)
    print(f"API URL: {args.api_url}")
    print(f"Model name: {args.model_name}")
    print(f"Test file: {args.test_file}")
    print(f"Output file: {output_file}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max workers: {args.max_workers}")
    print("=" * 80)
    
    # Khởi tạo OpenAI client
    client = OpenAI(
        base_url=args.api_url,
        api_key="EMPTY"  # vLLM không cần API key
    )
    
    # Load test data
    print("\nĐang load dữ liệu test...")
    test_data = load_test_data(args.test_file)
    print(f"Đã load {len(test_data)} samples")
    
    # Chạy inference với batch processing
    print("\nĐang chạy inference với batch processing...")
    
    # Chia data thành batches
    batches = [test_data[i:i+args.batch_size] for i in range(0, len(test_data), args.batch_size)]
    
    total_processed = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Progress bar cho tổng số samples
        with tqdm(total=len(test_data), desc="Generating summaries") as pbar:
            for batch in batches:
                # Xử lý batch song song với ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    # Submit tất cả tasks trong batch
                    future_to_item = {
                        executor.submit(
                            process_single_item,
                            item,
                            client,
                            args.model_name,
                            args.temperature,
                            args.top_p,
                            args.max_tokens
                        ): item for item in batch
                    }
                    
                    # Xử lý kết quả khi hoàn thành
                    for future in as_completed(future_to_item):
                        try:
                            result = future.result()
                            
                            # Ghi vào file ngay lập tức
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f.flush()
                            
                            total_processed += 1
                            pbar.update(1)
                            
                        except Exception as e:
                            print(f"\nLỗi khi xử lý item: {e}")
                            pbar.update(1)
    
    print("\n" + "=" * 80)
    print("Hoàn thành!")
    print("=" * 80)
    print(f"Đã xử lý: {total_processed}/{len(test_data)} samples")
    print(f"Kết quả được lưu tại: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script để serve model bằng vLLM với OpenAI-compatible API
Sử dụng TRL's vllm-serve command
Hỗ trợ cả full weights và LoRA adapter
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Serve model với vLLM OpenAI API")
    
    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Đường dẫn đến model đã fine-tune"
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default="qwen-sft",
        help="Tên model khi serve (default: qwen-sft)"
    )
    
    # Server arguments
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host để bind server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port để serve API (default: 8000)"
    )
    
    # Performance arguments
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Số GPU sử dụng cho tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Độ dài context tối đa (default: 4096)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type cho model weights (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Xây dựng command để chạy vLLM server với trl vllm-serve
    cmd = [
        "trl", "vllm-serve",
        "--model", args.model_path,
        "--served-model-name", args.served_model_name,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--dtype", args.dtype,
        "--trust-remote-code",
    ]
    
    print("=" * 80)
    print("Khởi động vLLM Server với TRL")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Model name: {args.served_model_name}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Max model length: {args.max_model_len}")
    print(f"Data type: {args.dtype}")
    print("=" * 80)
    print("\nCommand:")
    print(" ".join(cmd))
    print("=" * 80)
    print("\nĐang khởi động server...")
    print("Server sẽ chạy tại: http://{}:{}".format(args.host, args.port))
    print("API endpoint: http://{}:{}/v1/chat/completions".format(args.host, args.port))
    print("\nNhấn Ctrl+C để dừng server\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nĐang dừng server...")
        sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script đánh giá kết quả inference bằng ROUGE scores
Đọc file CSV với 3 cột: input, output (reference), predicted (candidate)
"""

import argparse
import json
import os
import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List


def calculate_rouge_scores(reference: str, candidate: str, scorer) -> Dict:
    """
    Tính ROUGE scores cho một cặp reference-candidate
    
    Args:
        reference: Text tham chiếu (ground truth)
        candidate: Text được sinh ra bởi model
        scorer: RougeScorer object
    
    Returns:
        Dictionary chứa các scores
    """
    scores = scorer.score(reference, candidate)
    
    result = {}
    for key, value in scores.items():
        result[f"{key}_precision"] = value.precision
        result[f"{key}_recall"] = value.recall
        result[f"{key}_fmeasure"] = value.fmeasure
    
    return result


def evaluate_dataset(input_file: str, output_dir: str) -> None:
    """
    Đánh giá toàn bộ dataset và lưu kết quả
    
    Args:
        input_file: Đường dẫn đến file CSV chứa predictions
        output_dir: Thư mục lưu kết quả
    """
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    # Đọc file CSV
    print(f"Đang đọc file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Kiểm tra các cột cần thiết
    required_columns = ['input', 'output', 'predicted']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"File CSV thiếu các cột: {missing_columns}")
    
    print(f"Đã load {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Khởi tạo ROUGE scorer
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    
    # Tính scores cho từng sample
    print("\nĐang tính ROUGE scores...")
    detailed_scores = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing ROUGE"):
        reference = str(row['output']).strip()
        candidate = str(row['predicted']).strip()
        
        # Tính scores
        scores = calculate_rouge_scores(reference, candidate, scorer)
        
        # Thêm thông tin sample
        scores['sample_id'] = idx
        scores['reference'] = reference
        scores['candidate'] = candidate
        
        detailed_scores.append(scores)
    
    # Tạo DataFrame với detailed scores
    detailed_df = pd.DataFrame(detailed_scores)
    
    # Tính average scores
    rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
    score_types = ['precision', 'recall', 'fmeasure']
    
    average_scores = {}
    for metric in rouge_metrics:
        average_scores[metric] = {}
        for score_type in score_types:
            col_name = f"{metric}_{score_type}"
            average_scores[metric][score_type] = detailed_df[col_name].mean()
    
    # Lưu detailed scores vào CSV
    detailed_output = os.path.join(output_dir, 'detailed_scores.csv')
    detailed_df.to_csv(detailed_output, index=False, encoding='utf-8')
    print(f"\nĐã lưu detailed scores: {detailed_output}")
    
    # Lưu average scores vào JSON
    average_output = os.path.join(output_dir, 'average_scores.json')
    with open(average_output, 'w', encoding='utf-8') as f:
        json.dump(average_scores, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu average scores: {average_output}")
    
    # In kết quả ra console
    print("\n" + "=" * 80)
    print("KẾT QUẢ ĐÁNH GIÁ ROUGE")
    print("=" * 80)
    print(f"Số samples: {len(df)}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print("\n" + "-" * 80)
    print("AVERAGE SCORES")
    print("-" * 80)
    
    for metric in rouge_metrics:
        print(f"\n{metric.upper()}:")
        print(f"  Precision: {average_scores[metric]['precision']:.4f}")
        print(f"  Recall:    {average_scores[metric]['recall']:.4f}")
        print(f"  F1 Score:  {average_scores[metric]['fmeasure']:.4f}")
    
    print("\n" + "=" * 80)
    
    # Tạo summary report
    summary_output = os.path.join(output_dir, 'summary_report.txt')
    with open(summary_output, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ROUGE EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Number of samples: {len(df)}\n")
        f.write("\n" + "-" * 80 + "\n")
        f.write("AVERAGE SCORES\n")
        f.write("-" * 80 + "\n")
        
        for metric in rouge_metrics:
            f.write(f"\n{metric.upper()}:\n")
            f.write(f"  Precision: {average_scores[metric]['precision']:.4f}\n")
            f.write(f"  Recall:    {average_scores[metric]['recall']:.4f}\n")
            f.write(f"  F1 Score:  {average_scores[metric]['fmeasure']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\nĐã lưu summary report: {summary_output}")
    
    # Tạo file với top và bottom samples
    print("\nĐang tìm top/bottom samples theo ROUGE-L F1...")
    
    # Sort by ROUGE-L F1
    detailed_df_sorted = detailed_df.sort_values('rougeL_fmeasure', ascending=False)
    
    # Top 10 samples
    top_samples = detailed_df_sorted.head(10)
    top_output = os.path.join(output_dir, 'top_10_samples.csv')
    top_samples.to_csv(top_output, index=False, encoding='utf-8')
    print(f"Đã lưu top 10 samples: {top_output}")
    
    # Bottom 10 samples
    bottom_samples = detailed_df_sorted.tail(10)
    bottom_output = os.path.join(output_dir, 'bottom_10_samples.csv')
    bottom_samples.to_csv(bottom_output, index=False, encoding='utf-8')
    print(f"Đã lưu bottom 10 samples: {bottom_output}")
    
    print("\n" + "=" * 80)
    print("HOÀN THÀNH!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá kết quả inference bằng ROUGE scores"
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Đường dẫn đến file CSV chứa predictions (3 cột: input, output, predicted)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/results",
        help="Thư mục lưu kết quả (default: evaluation/results)"
    )
    
    args = parser.parse_args()
    
    # Kiểm tra file input tồn tại
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Không tìm thấy file: {args.input_file}")
    
    # Chạy evaluation
    evaluate_dataset(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()

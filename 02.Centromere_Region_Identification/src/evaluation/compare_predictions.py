#!/usr/bin/env python3
"""
对比预测结果与真实标签
"""

import json
import csv
import os
import pandas as pd
from pathlib import Path

def load_ground_truth(bed_file):
    """加载真实标签BED文件"""
    gt_regions = {}
    with open(bed_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                seq_name = parts[0]
                if parts[1] != 'NA' and parts[2] != 'NA':
                    try:
                        start = int(parts[1])
                        end = int(parts[2])
                        gt_regions[seq_name] = (start, end)
                    except ValueError:
                        continue
    return gt_regions

def compare_predictions(results_dir, bed_file, output_file):
    """对比预测结果与真实标签"""
    # 加载真实标签
    gt_regions = load_ground_truth(bed_file)
    print(f"加载了 {len(gt_regions)} 个真实标签区域")
    
    # 加载评估结果
    eval_json = os.path.join(results_dir, 'evaluation_results.json')
    if not os.path.exists(eval_json):
        print(f"错误：找不到评估结果文件 {eval_json}")
        return
    
    with open(eval_json, 'r') as f:
        eval_data = json.load(f)
    
    # 加载最佳单区域预测
    best_single_csv = os.path.join(results_dir, 'test_best_single_region.csv')
    if os.path.exists(best_single_csv):
        best_single_df = pd.read_csv(best_single_csv)
    else:
        print(f"警告：找不到最佳单区域预测文件 {best_single_csv}")
        best_single_df = None
    
    # 生成对比报告
    comparison_results = []
    
    if 'test' in eval_data and 'chromosomes' in eval_data['test']:
        for chr_result in eval_data['test']['chromosomes']:
            chr_name = chr_result['chromosome']
            
            # 获取真实标签
            gt_start, gt_end = gt_regions.get(chr_name, (None, None))
            
            # 获取预测结果
            pred_start = chr_result.get('final_single_start', 0)
            pred_end = chr_result.get('final_single_end', 0)
            
            comparison_results.append({
                'chromosome': chr_name,
                'gt_start': gt_start if gt_start else 'NA',
                'gt_end': gt_end if gt_end else 'NA',
                'gt_length': (gt_end - gt_start) if (gt_start and gt_end) else 'NA',
                'pred_start': pred_start if pred_start > 0 else 'NA',
                'pred_end': pred_end if pred_end > 0 else 'NA',
                'pred_length': (pred_end - pred_start) if (pred_start > 0 and pred_end > 0) else 'NA',
                'precision': f"{chr_result.get('final_single_precision', 0):.4f}",
                'recall': f"{chr_result.get('final_single_recall', 0):.4f}",
                'f1': f"{chr_result.get('final_single_f1', 0):.4f}",
                'iou': f"{chr_result.get('final_single_iou', 0):.4f}",
                'accuracy': f"{chr_result.get('final_single_accuracy', 0):.4f}",
            })
    
    # 保存对比结果
    if comparison_results:
        df = pd.DataFrame(comparison_results)
        df.to_csv(output_file, index=False)
        print(f"\n对比结果已保存到: {output_file}")
        
        # 打印汇总统计
        print("\n" + "="*80)
        print("预测结果对比汇总")
        print("="*80)
        
        # 计算有真实标签的染色体数量
        has_gt = df[df['gt_start'] != 'NA']
        has_pred = df[df['pred_start'] != 'NA']
        has_both = df[(df['gt_start'] != 'NA') & (df['pred_start'] != 'NA')]
        
        print(f"\n总染色体数: {len(df)}")
        print(f"有真实标签的: {len(has_gt)}")
        print(f"有预测结果的: {len(has_pred)}")
        print(f"两者都有的: {len(has_both)}")
        
        if len(has_both) > 0:
            # 计算平均指标
            metrics_cols = ['precision', 'recall', 'f1', 'iou', 'accuracy']
            for col in metrics_cols:
                if col in has_both.columns:
                    values = pd.to_numeric(has_both[col], errors='coerce')
                    mean_val = values.mean()
                    print(f"\n平均 {col}: {mean_val:.4f}")
        
        # 打印整体指标
        if 'test' in eval_data and 'overall' in eval_data['test']:
            overall = eval_data['test']['overall']
            print("\n" + "-"*80)
            print("整体评估指标 (Final Single Prediction):")
            print("-"*80)
            print(f"Precision: {overall.get('final_single_precision', 0):.4f}")
            print(f"Recall:    {overall.get('final_single_recall', 0):.4f}")
            print(f"F1 Score:  {overall.get('final_single_f1', 0):.4f}")
            print(f"IoU:       {overall.get('final_single_iou', 0):.4f}")
            print(f"AUC:       {overall.get('auc', 0):.4f}")
            print(f"\n混淆矩阵:")
            print(f"  TP: {overall.get('tp', 0):,}")
            print(f"  FP: {overall.get('fp', 0):,}")
            print(f"  FN: {overall.get('fn', 0):,}")
            print(f"  TN: {overall.get('tn', 0):,}")
            print(f"  总样本数: {overall.get('total_samples', 0):,}")
            print(f"  正样本数: {overall.get('positive_samples', 0):,}")
        
        print("\n" + "="*80)
        print(f"详细对比结果请查看: {output_file}")
        print("="*80)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("使用方法: python compare_predictions.py <results_dir> <bed_file> [output_file]")
        print("示例: python compare_predictions.py inference/test_inference_results/20251215_151629 training/training_data/07.fasta_for_train/test/test/test.cenRegion.bed")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    bed_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else os.path.join(results_dir, 'prediction_comparison.csv')
    
    compare_predictions(results_dir, bed_file, output_file)





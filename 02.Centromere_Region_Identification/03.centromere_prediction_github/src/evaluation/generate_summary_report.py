#!/usr/bin/env python3
"""
生成平均指标汇总报告
"""

import pandas as pd
import numpy as np
import os
import sys

def generate_summary_report(results_dir, output_file=None):
    """生成汇总报告"""
    
    # 读取三个CSV文件
    comparison_file = os.path.join(results_dir, 'prediction_comparison.csv')
    best_single_file = os.path.join(results_dir, 'test_best_single_region.csv')
    metrics_file = os.path.join(results_dir, 'test_metrics.csv')
    
    if not all(os.path.exists(f) for f in [comparison_file, best_single_file, metrics_file]):
        print("错误：找不到所需的CSV文件")
        return
    
    df1 = pd.read_csv(comparison_file)
    df2 = pd.read_csv(best_single_file)
    df3 = pd.read_csv(metrics_file)
    
    # 收集所有指标
    summary_data = []
    
    # 从prediction_comparison.csv
    metrics = ['precision', 'recall', 'f1', 'iou', 'accuracy']
    for m in metrics:
        if m in df1.columns:
            values = pd.to_numeric(df1[m], errors='coerce').dropna()
            if len(values) > 0:
                summary_data.append({
                    'category': 'Final Single Prediction (from comparison)',
                    'metric': m,
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'count': len(values)
                })
    
    # 从test_best_single_region.csv
    metrics = ['final_single_precision', 'final_single_recall', 'final_single_f1', 
               'final_single_iou', 'final_single_accuracy']
    for m in metrics:
        if m in df2.columns:
            values = pd.to_numeric(df2[m], errors='coerce').dropna()
            if len(values) > 0:
                clean_name = m.replace('final_single_', '')
                summary_data.append({
                    'category': 'Final Single Prediction (from best_single_region)',
                    'metric': clean_name,
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'count': len(values)
                })
    
    # 从test_metrics.csv - Multi-Region
    metrics = ['precision', 'recall', 'f1', 'iou']
    for m in metrics:
        if m in df3.columns:
            values = pd.to_numeric(df3[m], errors='coerce').dropna()
            if len(values) > 0:
                summary_data.append({
                    'category': 'Multi-Region',
                    'metric': m,
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'count': len(values)
                })
    
    # 从test_metrics.csv - Final Single
    metrics = ['final_single_precision', 'final_single_recall', 'final_single_f1', 
               'final_single_iou']
    for m in metrics:
        if m in df3.columns:
            values = pd.to_numeric(df3[m], errors='coerce').dropna()
            if len(values) > 0:
                clean_name = m.replace('final_single_', '')
                summary_data.append({
                    'category': 'Final Single Prediction (from metrics)',
                    'metric': clean_name,
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'count': len(values)
                })
    
    # AUC
    if 'auc' in df3.columns:
        values = pd.to_numeric(df3['auc'], errors='coerce').dropna()
        if len(values) > 0:
            summary_data.append({
                'category': 'Other',
                'metric': 'auc',
                'mean': float(values.mean()),
                'std': float(values.std()),
                'median': float(values.median()),
                'min': float(values.min()),
                'max': float(values.max()),
                'count': len(values)
            })
    
    # 创建DataFrame并保存
    summary_df = pd.DataFrame(summary_data)
    
    if output_file is None:
        output_file = os.path.join(results_dir, 'average_metrics_summary.csv')
    
    summary_df.to_csv(output_file, index=False)
    
    # 打印报告
    print("\n" + "="*80)
    print("平均指标汇总报告")
    print("="*80)
    
    categories = summary_df['category'].unique()
    for cat in categories:
        cat_data = summary_df[summary_df['category'] == cat]
        print(f"\n【{cat}】")
        print("-" * 80)
        print(f"{'指标':<20s} {'平均值':<12s} {'标准差':<12s} {'中位数':<12s} {'最小值':<10s} {'最大值':<10s} {'样本数':<8s}")
        print("-" * 80)
        for _, row in cat_data.iterrows():
            print(f"{row['metric']:<20s} {row['mean']:>10.4f}  {row['std']:>10.4f}  "
                  f"{row['median']:>10.4f}  {row['min']:>8.4f}  {row['max']:>8.4f}  {int(row['count']):>6d}")
    
    print("\n" + "="*80)
    print(f"详细汇总已保存到: {output_file}")
    print("="*80)
    
    return summary_df

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法: python generate_summary_report.py <results_dir> [output_file]")
        print("示例: python generate_summary_report.py inference/test_inference_results/20251215_151629")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_summary_report(results_dir, output_file)




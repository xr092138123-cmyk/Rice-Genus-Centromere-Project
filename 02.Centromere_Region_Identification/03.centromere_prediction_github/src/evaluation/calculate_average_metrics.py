#!/usr/bin/env python3
"""
计算平均评估指标
"""

import pandas as pd
import numpy as np
import os
import sys

def calculate_averages_from_csv(csv_file, output_file=None):
    """从CSV文件计算平均指标"""
    if not os.path.exists(csv_file):
        print(f"错误：文件不存在 {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    
    # 找出所有数值列（排除染色体名称等非数值列）
    numeric_cols = []
    exclude_cols = ['chromosome', 'gt_start', 'gt_end', 'gt_length', 
                    'pred_start', 'pred_end', 'pred_length',
                    'final_single_start', 'final_single_end']
    
    for col in df.columns:
        if col not in exclude_cols:
            # 尝试转换为数值
            try:
                # 处理NA值
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isna().all():
                    numeric_cols.append(col)
            except:
                pass
    
    results = {}
    
    for col in numeric_cols:
        # 转换为数值，处理NA
        values = pd.to_numeric(df[col], errors='coerce')
        # 移除NA值
        valid_values = values.dropna()
        
        if len(valid_values) > 0:
            results[col] = {
                'mean': float(valid_values.mean()),
                'std': float(valid_values.std()),
                'median': float(valid_values.median()),
                'min': float(valid_values.min()),
                'max': float(valid_values.max()),
                'count': int(len(valid_values)),
                'total': int(len(values))
            }
    
    return results

def print_summary(results, title="平均指标汇总"):
    """打印汇总结果"""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    if not results:
        print("没有找到可计算的指标")
        return
    
    # 提取主要指标（去除文件名前缀）
    main_results = {}
    for key, value in results.items():
        # 移除文件名前缀
        clean_key = key
        if '_' in key:
            parts = key.split('_', 1)
            if len(parts) > 1:
                clean_key = parts[1]
        
        if clean_key not in main_results:
            main_results[clean_key] = []
        main_results[clean_key].append(value)
    
    # 按类别分组显示
    categories = {
        'Multi-Region Metrics': ['precision', 'recall', 'f1', 'iou', 'accuracy'],
        'Final Single Prediction Metrics': ['final_single_precision', 'final_single_recall', 
                                            'final_single_f1', 'final_single_iou', 
                                            'final_single_accuracy'],
        'Other Metrics': ['auc']
    }
    
    # 合并相同指标的结果（如果有多个文件）
    merged_results = {}
    for clean_key, values_list in main_results.items():
        # 如果有多个值，取平均
        if len(values_list) == 1:
            merged_results[clean_key] = values_list[0]
        else:
            # 合并多个文件的结果
            total_count = sum(v['count'] for v in values_list)
            total_valid = sum(v['count'] for v in values_list)
            if total_valid > 0:
                # 加权平均（基于count）
                weighted_mean = sum(v['mean'] * v['count'] for v in values_list) / total_valid
                merged_results[clean_key] = {
                    'mean': weighted_mean,
                    'std': np.mean([v['std'] for v in values_list]),
                    'median': np.mean([v['median'] for v in values_list]),
                    'min': min(v['min'] for v in values_list),
                    'max': max(v['max'] for v in values_list),
                    'count': total_valid,
                    'total': total_count
                }
    
    for category, keys in categories.items():
        found_keys = [k for k in keys if k in merged_results]
        if found_keys:
            print(f"\n{category}:")
            print("-" * 80)
            for key in found_keys:
                r = merged_results[key]
                print(f"  {key:30s}: Mean={r['mean']:8.4f}  Std={r['std']:8.4f}  "
                      f"Median={r['median']:8.4f}  Range=[{r['min']:.4f}, {r['max']:.4f}]  "
                      f"({r['count']}/{r['total']} valid)")
    
    # 显示其他未分类的指标
    all_categorized = []
    for keys in categories.values():
        all_categorized.extend(keys)
    
    other_keys = [k for k in merged_results.keys() if k not in all_categorized]
    if other_keys:
        print(f"\n其他指标:")
        print("-" * 80)
        for key in other_keys:
            r = merged_results[key]
            print(f"  {key:30s}: Mean={r['mean']:8.4f}  Std={r['std']:8.4f}  "
                  f"Median={r['median']:8.4f}  Range=[{r['min']:.4f}, {r['max']:.4f}]  "
                  f"({r['count']}/{r['total']} valid)")
    
    return merged_results

def main():
    if len(sys.argv) < 2:
        print("使用方法: python calculate_average_metrics.py <csv_file1> [csv_file2] ...")
        print("\n示例:")
        print("  python calculate_average_metrics.py prediction_comparison.csv")
        print("  python calculate_average_metrics.py test_best_single_region.csv test_metrics.csv")
        sys.exit(1)
    
    csv_files = sys.argv[1:]
    
    all_results = {}
    
    for csv_file in csv_files:
        print(f"\n处理文件: {csv_file}")
        results = calculate_averages_from_csv(csv_file)
        
        if results:
            # 添加文件名前缀以避免冲突
            file_basename = os.path.basename(csv_file).replace('.csv', '')
            for key, value in results.items():
                new_key = f"{file_basename}_{key}" if len(csv_files) > 1 else key
                all_results[new_key] = value
    
    # 打印汇总
    merged_results = print_summary(all_results, "平均指标汇总 - 所有文件")
    
    # 显示简化的主要指标摘要
    print("\n" + "="*80)
    print("主要指标摘要")
    print("="*80)
    
    main_metrics = ['precision', 'recall', 'f1', 'iou', 'accuracy',
                   'final_single_precision', 'final_single_recall', 
                   'final_single_f1', 'final_single_iou', 'final_single_accuracy',
                   'auc']
    
    print(f"\n{'指标':<35s} {'平均值':<12s} {'标准差':<12s} {'中位数':<12s} {'样本数':<10s}")
    print("-" * 80)
    
    for metric in main_metrics:
        if metric in merged_results:
            r = merged_results[metric]
            print(f"{metric:<35s} {r['mean']:>10.4f}  {r['std']:>10.4f}  "
                  f"{r['median']:>10.4f}  {r['count']:>8d}")
    
    # 保存汇总结果到CSV
    if merged_results:
        summary_data = []
        for metric, r in merged_results.items():
            summary_data.append({
                'metric': metric,
                'mean': r['mean'],
                'std': r['std'],
                'median': r['median'],
                'min': r['min'],
                'max': r['max'],
                'count': r['count'],
                'total': r['total']
            })
        
        summary_df = pd.DataFrame(summary_data)
        output_file = 'average_metrics_summary.csv'
        summary_df.to_csv(output_file, index=False)
        print(f"\n汇总结果已保存到: {output_file}")

if __name__ == '__main__':
    main()


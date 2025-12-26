#!/usr/bin/env python3
"""
处理训练CSV文件，计算coverage_matrix_1024的统计值

对于每个区域：
1. 将coverage_matrix_1024的1024个值分成10组（每组约102个值）
2. 对每组求和，得到10个和
3. 对这10个和取平均
4. 除以10240得到最终值

输出新的CSV包含：
- start: 区域开始位置
- end: 区域结束位置
- coverage_mean: 计算出的平均值
- has_cen: 是否是着丝粒的真值
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def calculate_coverage_statistic(coverage_str):
    """
    计算coverage_matrix_1024的统计值
    
    假设coverage_matrix_1024是一个10行x102列的矩阵（在CSV中被展平为1024个值）
    计算步骤：
    1. 将1024个值重新组织成10行x102列的矩阵
    2. 对每行求和，得到10个行和
    3. 对10个行和取平均
    4. 除以10240得到最终值
    
    参数:
        coverage_str: coverage_matrix_1024列的字符串值（逗号分隔的1024个整数）
    
    返回:
        计算出的平均值（10行和的平均值 / 10240）
    """
    if pd.isna(coverage_str) or coverage_str == '':
        return np.nan
    
    # 将字符串转换为整数列表
    try:
        values = [int(x.strip()) for x in str(coverage_str).split(',')]
    except:
        return np.nan
    
    if len(values) != 1024:
        # 如果长度不是1024，尝试处理
        if len(values) == 0:
            return np.nan
        # 如果长度不足，用0填充；如果超出，截断
        if len(values) < 1024:
            values.extend([0] * (1024 - len(values)))
        else:
            values = values[:1024]
    
    # 将1024个值分成10组（作为"10行"）
    # 1024 / 10 = 102.4，所以前4组每组103个值，后6组每组102个值
    num_groups = 10
    group_size_base = 1024 // num_groups  # 102
    remainder = 1024 % num_groups  # 4
    
    group_sums = []
    idx = 0
    
    for i in range(num_groups):
        if i < remainder:
            # 前4组，每组103个值
            group_size = group_size_base + 1  # 103
        else:
            # 后6组，每组102个值
            group_size = group_size_base  # 102
        
        group = values[idx:idx+group_size]
        idx += group_size
        
        group_sum = sum(group)
        group_sums.append(group_sum)
    
    # 对10个组和取平均
    mean_sum = np.mean(group_sums)
    
    # 除以10240
    result = mean_sum / 10240.0
    
    return result


def process_training_csv(input_file, output_file):
    """
    处理单个训练CSV文件
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
    """
    print(f"Processing: {input_file}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        
        # 检查必要的列是否存在
        required_cols = ['start', 'end', 'has_cen', 'coverage_matrix_1024']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing columns: {missing_cols}")
            return False
        
        # 计算coverage统计值
        print(f"  Calculating coverage statistics for {len(df)} rows...")
        df['coverage_mean'] = df['coverage_matrix_1024'].apply(calculate_coverage_statistic)
        
        # 处理has_cen列：将NaN填充为0，并转换为整数类型
        df['has_cen'] = df['has_cen'].fillna(0).astype(int)
        
        # 分离有coverage数据的行和没有coverage数据但has_cen=1的行
        rows_with_coverage = df[df['coverage_mean'].notna()].copy()
        rows_without_coverage_but_has_cen = df[(df['coverage_mean'].isna()) & (df['has_cen'] == 1)].copy()
        
        # 对于有coverage数据的行，检查它们是否落在has_cen=1的大区间内
        # 如果是，将它们的has_cen设置为1
        if len(rows_without_coverage_but_has_cen) > 0 and len(rows_with_coverage) > 0:
            print(f"  Found {len(rows_without_coverage_but_has_cen)} rows with has_cen=1 but no coverage data")
            print(f"  Checking {len(rows_with_coverage)} rows with coverage data for overlap...")
            
            for idx, cen_row in rows_without_coverage_but_has_cen.iterrows():
                cen_start = int(cen_row['start'])
                cen_end = int(cen_row['end'])
                
                # 找到所有落在该区间内的行
                mask = (rows_with_coverage['start'] >= cen_start) & (rows_with_coverage['end'] <= cen_end)
                rows_with_coverage.loc[mask, 'has_cen'] = 1
                
                overlap_count = mask.sum()
                if overlap_count > 0:
                    print(f"    Region {cen_start}-{cen_end}: updated {overlap_count} overlapping rows to has_cen=1")
        
        # 使用有coverage数据的行作为输出基础
        output_df = rows_with_coverage[['start', 'end', 'coverage_mean', 'has_cen']].copy()
        
        # 合并相同区间的行（start和end都相同）
        # 对于coverage_mean：取平均
        # 对于has_cen：取最大值（如果有1则保留1）
        if len(output_df) > 0:
            output_df = output_df.groupby(['start', 'end']).agg({
                'coverage_mean': 'mean',  # 取平均
                'has_cen': 'max'  # has_cen取最大值（如果有1则保留1）
            }).reset_index()
        
        # 保存输出文件
        output_df.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}")
        print(f"  Processed {len(output_df)} rows")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def find_all_training_csvs(root_dir):
    """
    查找所有training CSV文件
    
    参数:
        root_dir: 根目录路径
    
    返回:
        所有training CSV文件的路径列表
    """
    training_files = []
    root_path = Path(root_dir)
    
    for csv_file in root_path.rglob('*_training.csv'):
        training_files.append(csv_file)
    
    return sorted(training_files)


def find_all_coverage_stats_csvs(root_dir):
    """
    查找所有coverage_stats CSV文件
    
    参数:
        root_dir: 根目录路径
    
    返回:
        所有coverage_stats CSV文件的路径列表
    """
    stats_files = []
    root_path = Path(root_dir)
    
    for csv_file in root_path.rglob('*_coverage_stats.csv'):
        stats_files.append(csv_file)
    
    return sorted(stats_files)


def merge_coverage_statistics(csv_file, output_file=None):
    """
    合并coverage_stats CSV文件中的相同区间
    
    参数:
        csv_file: 输入CSV文件路径
        output_file: 输出CSV文件路径（如果为None，则覆盖原文件）
    
    返回:
        是否成功
    """
    try:
        print(f"Merging: {csv_file}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 检查必要的列
        required_cols = ['start', 'end', 'coverage_mean', 'has_cen']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing columns: {missing_cols}")
            return False
        
        original_rows = len(df)
        
        # 删除没有coverage数值的行
        df = df[df['coverage_mean'].notna()].copy()
        rows_after_filter = len(df)
        
        # 合并相同区间的行（start和end都相同），coverage_mean取平均
        if len(df) > 0:
            df_merged = df.groupby(['start', 'end']).agg({
                'coverage_mean': 'mean',  # 取平均
                'has_cen': 'max'  # has_cen取最大值（如果有1则保留1）
            }).reset_index()
        else:
            df_merged = df
        
        rows_after_merge = len(df_merged)
        
        # 确定输出文件路径
        if output_file is None:
            output_file = csv_file  # 覆盖原文件
        else:
            output_file = Path(output_file)
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存合并后的文件
        df_merged.to_csv(output_file, index=False)
        
        print(f"  Original rows: {original_rows}")
        print(f"  After filtering NaN: {rows_after_filter}")
        print(f"  After merging: {rows_after_merge}")
        print(f"  Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error merging {csv_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def merge_all_coverage_statistics(root_dir, output_dir=None, in_place=False):
    """
    批量合并所有coverage_stats CSV文件
    
    参数:
        root_dir: 根目录路径（包含所有coverage_stats CSV文件）
        output_dir: 输出目录路径（如果为None且in_place=False，则覆盖原文件）
        in_place: 是否在原位置覆盖文件
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: Directory not found: {root_path}")
        return
    
    stats_files = find_all_coverage_stats_csvs(root_path)
    print(f"Found {len(stats_files)} coverage_stats CSV files")
    
    if len(stats_files) == 0:
        print("No coverage_stats CSV files found!")
        return
    
    success_count = 0
    fail_count = 0
    
    for csv_file in stats_files:
        if output_dir and not in_place:
            # 保持相对路径结构
            rel_path = csv_file.relative_to(root_path)
            output_file = Path(output_dir) / rel_path.parent / csv_file.name
        else:
            output_file = None  # 覆盖原文件
        
        if merge_coverage_statistics(csv_file, output_file):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nMerging complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")


def plot_coverage_statistics(csv_file, output_file=None, figsize=(14, 6)):
    """
    绘制coverage统计值的折线图，并用浅绿色标记has_cen=1的区域
    
    参数:
        csv_file: 输入CSV文件路径（coverage_stats文件）
        output_file: 输出图片文件路径（如果为None，则在CSV同目录下生成）
        figsize: 图片大小（宽，高）
    
    返回:
        是否成功
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 检查必要的列
        required_cols = ['start', 'end', 'coverage_mean', 'has_cen']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing columns: {missing_cols}")
            return False
        
        # 处理has_cen列：将空值或NaN填充为0，并转换为整数类型
        df['has_cen'] = pd.to_numeric(df['has_cen'], errors='coerce').fillna(0).astype(int)
        
        # 过滤掉coverage_mean为NaN的行
        df = df[df['coverage_mean'].notna()].copy()
        
        if len(df) == 0:
            print(f"Warning: No valid data in {csv_file}")
            return False
        
        # 计算每个区域的中点位置（用于绘图）
        df['position'] = (df['start'] + df['end']) / 2
        
        # 按位置排序
        df = df.sort_values('position').reset_index(drop=True)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制折线图
        ax.plot(df['position'], df['coverage_mean'], 
                linewidth=0.8, color='#2E86AB', alpha=0.7, label='Coverage Mean')
        
        # 标记has_cen=1的区域（浅绿色背景）
        cen_regions = df[df['has_cen'] == 1]
        if len(cen_regions) > 0:
            for idx, row in cen_regions.iterrows():
                # 在has_cen=1的区域绘制浅绿色背景
                ax.axvspan(row['start'], row['end'], 
                          alpha=0.3, color='#90EE90', label='Centromere (has_cen=1)' if idx == cen_regions.index[0] else '')
        
        # 设置标签和标题
        ax.set_xlabel('Position (bp)', fontsize=12)
        ax.set_ylabel('Coverage Mean', fontsize=12)
        
        # 从文件名提取标题
        file_name = Path(csv_file).stem.replace('_coverage_stats', '')
        ax.set_title(f'Coverage Statistics: {file_name}', fontsize=14, fontweight='bold')
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加图例（只显示一次）
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if output_file is None:
            output_file = Path(csv_file).parent / (Path(csv_file).stem + '.png')
        else:
            output_file = Path(output_file)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error plotting {csv_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def plot_all_coverage_statistics(root_dir, output_dir=None):
    """
    批量绘制所有coverage_stats CSV文件的折线图
    
    参数:
        root_dir: 根目录路径（包含所有coverage_stats CSV文件）
        output_dir: 输出目录路径（如果为None，则在CSV同目录下生成）
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: Directory not found: {root_path}")
        return
    
    stats_files = find_all_coverage_stats_csvs(root_path)
    print(f"Found {len(stats_files)} coverage_stats CSV files")
    
    if len(stats_files) == 0:
        print("No coverage_stats CSV files found!")
        return
    
    success_count = 0
    fail_count = 0
    
    for csv_file in stats_files:
        print(f"Plotting: {csv_file}")
        
        if output_dir:
            # 保持相对路径结构
            rel_path = csv_file.relative_to(root_path)
            output_file = Path(output_dir) / rel_path.parent / (csv_file.stem + '.png')
        else:
            output_file = None  # 使用默认路径（CSV同目录）
        
        if plot_coverage_statistics(csv_file, output_file):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nPlotting complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")


def main():
    parser = argparse.ArgumentParser(
        description='处理训练CSV文件，计算coverage_matrix_1024的统计值'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        nargs='?',
        default=None,
        help='输入目录路径（包含所有training CSV文件）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录路径（如果未指定，将在输入文件同目录下创建）'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_coverage_stats.csv',
        help='输出文件后缀（默认: _coverage_stats.csv）'
    )
    parser.add_argument(
        '--single-file',
        type=str,
        default=None,
        help='只处理单个文件（用于测试）'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='绘制coverage统计值的折线图（需要先处理CSV文件）'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default=None,
        help='绘图输出目录（如果未指定，将在CSV文件同目录下生成）'
    )
    parser.add_argument(
        '--plot-single',
        type=str,
        default=None,
        help='只绘制单个coverage_stats CSV文件的图（用于测试）'
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        help='合并coverage_stats CSV文件中的相同区间（需要先处理CSV文件）'
    )
    parser.add_argument(
        '--merge-single',
        type=str,
        default=None,
        help='只合并单个coverage_stats CSV文件（用于测试）'
    )
    parser.add_argument(
        '--merge-output-dir',
        type=str,
        default=None,
        help='合并后的输出目录（如果未指定，将覆盖原文件）'
    )
    
    args = parser.parse_args()
    
    # 处理合并请求
    if args.merge_single:
        # 合并单个文件
        csv_file = Path(args.merge_single)
        if not csv_file.exists():
            print(f"Error: File not found: {csv_file}")
            return
        
        output_file = args.merge_output_dir if args.merge_output_dir else None
        merge_coverage_statistics(csv_file, output_file)
        return
    
    if args.merge:
        # 批量合并所有文件
        if not args.input_dir:
            parser.error("input_dir is required when using --merge")
        
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}")
            return
        
        merge_all_coverage_statistics(input_dir, args.merge_output_dir, in_place=(args.merge_output_dir is None))
        return
    
    # 处理绘图请求
    if args.plot_single:
        # 绘制单个文件
        csv_file = Path(args.plot_single)
        if not csv_file.exists():
            print(f"Error: File not found: {csv_file}")
            return
        
        output_file = args.plot_dir if args.plot_dir else None
        plot_coverage_statistics(csv_file, output_file)
        return
    
    if args.plot:
        # 批量绘制所有文件
        if not args.input_dir:
            parser.error("input_dir is required when using --plot")
        
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}")
            return
        
        plot_all_coverage_statistics(input_dir, args.plot_dir)
        return
    
    # 处理CSV文件
    if args.single_file:
        # 处理单个文件
        input_file = Path(args.single_file)
        if not input_file.exists():
            print(f"Error: File not found: {input_file}")
            return
        
        if args.output_dir:
            output_file = Path(args.output_dir) / (input_file.stem + args.output_suffix)
        else:
            output_file = input_file.parent / (input_file.stem + args.output_suffix)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        process_training_csv(input_file, output_file)
    else:
        # 处理所有文件
        if not args.input_dir:
            parser.error("input_dir is required when not using --single-file")
        
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}")
            return
        
        training_files = find_all_training_csvs(input_dir)
        print(f"Found {len(training_files)} training CSV files")
        
        success_count = 0
        fail_count = 0
        
        for input_file in training_files:
            if args.output_dir:
                # 保持相对路径结构
                rel_path = input_file.relative_to(input_dir)
                output_file = Path(args.output_dir) / rel_path.parent / (input_file.stem + args.output_suffix)
            else:
                output_file = input_file.parent / (input_file.stem + args.output_suffix)
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if process_training_csv(input_file, output_file):
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\nProcessing complete!")
        print(f"  Success: {success_count}")
        print(f"  Failed: {fail_count}")


if __name__ == '__main__':
    main()


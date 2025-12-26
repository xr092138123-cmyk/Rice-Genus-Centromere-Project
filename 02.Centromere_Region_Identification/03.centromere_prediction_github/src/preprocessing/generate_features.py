#!/usr/bin/env python3
"""
从FASTA文件和k-mer计数生成模型输入特征CSV
"""

import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import defaultdict
import argparse
import os
import sys

def load_kmer_counts(kmer_file):
    """
    加载k-mer计数文件（Jellyfish格式）
    
    格式：
    >kmer_sequence
    count_value
    """
    kmer_counts = {}
    print(f"  Loading {kmer_file}...")
    
    with open(kmer_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            
            if line.startswith('>'):
                kmer = line.strip()[1:]  # 去除 '>'
                count_line = f.readline()
                if count_line:
                    count = int(count_line.strip())
                    kmer_counts[kmer] = count
    
    print(f"    Loaded {len(kmer_counts)} unique k-mers")
    return kmer_counts

def calculate_bin_statistics(genome_file, kmer_counts_dict, bin_size=10000, chromosome=None):
    """
    计算每个bin的统计特征
    
    Args:
        genome_file: FASTA文件路径
        kmer_counts_dict: {k_value: kmer_counts}的字典
        bin_size: bin大小（默认10kb）
        chromosome: 只处理指定染色体（可选）
    
    Returns:
        DataFrame with features
    """
    results = []
    
    # 读取基因组序列
    for record in SeqIO.parse(genome_file, "fasta"):
        # 如果指定了染色体，只处理该染色体
        if chromosome and record.id != chromosome:
            continue
            
        print(f"Processing {record.id} (length: {len(record.seq):,} bp)...")
        seq = str(record.seq).upper()
        seq_len = len(seq)
        
        # 按bin分割
        num_bins = (seq_len + bin_size - 1) // bin_size
        for bin_idx in range(num_bins):
            start = bin_idx * bin_size
            end = min(start + bin_size, seq_len)
            bin_seq = seq[start:end]
            
            # 跳过太短的bin
            if len(bin_seq) < bin_size // 2:
                continue
            
            bin_features = {
                'chromosome': record.id,
                'start': start,
                'end': end,
                'has_cen': 0  # 推理时默认为0
            }
            
            # 计算每个k值的特征
            for k, kmer_counts in kmer_counts_dict.items():
                if len(bin_seq) < k:
                    bin_features[f'{k}_highlighted_percent'] = 0.0
                    bin_features[f'{k}_coverage_depth_avg'] = 0.0
                    continue
                
                # 提取bin中的所有k-mers
                bin_kmers = []
                for i in range(len(bin_seq) - k + 1):
                    kmer = bin_seq[i:i+k]
                    if 'N' not in kmer:  # 跳过含N的k-mer
                        bin_kmers.append(kmer)
                
                if not bin_kmers:
                    bin_features[f'{k}_highlighted_percent'] = 0.0
                    bin_features[f'{k}_coverage_depth_avg'] = 0.0
                    continue
                
                # 计算统计量
                counts = [kmer_counts.get(kmer, 0) for kmer in bin_kmers]
                
                # highlighted_percent: 高频k-mer的比例
                # 定义"高频"为大于中位数的2倍
                if counts:
                    median_count = np.median(counts)
                    threshold = median_count * 2
                    highlighted = sum(1 for c in counts if c > threshold)
                    highlighted_percent = highlighted / len(counts)
                    avg_depth = np.mean(counts)
                else:
                    highlighted_percent = 0.0
                    avg_depth = 0.0
                
                bin_features[f'{k}_highlighted_percent'] = highlighted_percent
                bin_features[f'{k}_coverage_depth_avg'] = avg_depth
            
            results.append(bin_features)
            
            # 进度提示
            if (bin_idx + 1) % 100 == 0:
                print(f"  Processed {bin_idx + 1}/{num_bins} bins", end='\r')
        
        print(f"  Completed {num_bins} bins for {record.id}" + " " * 20)
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(
        description='Generate feature CSV from FASTA and k-mer counts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # 基本用法
  python generate_features.py \\
      --genome genome.fasta \\
      --kmer-dir kmer_counts \\
      --output features.csv
  
  # 指定bin大小
  python generate_features.py \\
      --genome genome.fasta \\
      --kmer-dir kmer_counts \\
      --output features.csv \\
      --bin-size 5000
  
  # 只处理特定染色体
  python generate_features.py \\
      --genome genome.fasta \\
      --kmer-dir kmer_counts \\
      --output chr1_features.csv \\
      --chromosome chr1
        """
    )
    parser.add_argument('--genome', required=True, help='Input FASTA file')
    parser.add_argument('--kmer-dir', required=True, help='Directory containing k-mer count files')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--bin-size', type=int, default=10000, 
                       help='Bin size in base pairs (default: 10000)')
    parser.add_argument('--chromosome', type=str, default=None,
                       help='Process only this chromosome (optional)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[64, 128, 256, 512],
                       help='K-mer sizes to use (default: 64 128 256 512)')
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.genome):
        print(f"Error: Genome file not found: {args.genome}")
        sys.exit(1)
    
    if not os.path.exists(args.kmer_dir):
        print(f"Error: K-mer directory not found: {args.kmer_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("Feature Generation from FASTA")
    print("=" * 60)
    print(f"Genome file: {args.genome}")
    print(f"K-mer directory: {args.kmer_dir}")
    print(f"Output file: {args.output}")
    print(f"Bin size: {args.bin_size:,} bp")
    print(f"K values: {args.k_values}")
    if args.chromosome:
        print(f"Chromosome: {args.chromosome}")
    print("=" * 60)
    
    # 加载k-mer计数
    print("\n[Step 1/2] Loading k-mer counts...")
    kmer_counts_dict = {}
    for k in args.k_values:
        kmer_file = os.path.join(args.kmer_dir, f"{k}mer_counts.txt")
        if not os.path.exists(kmer_file):
            print(f"Warning: K-mer file not found: {kmer_file}, skipping k={k}")
            continue
        kmer_counts_dict[k] = load_kmer_counts(kmer_file)
    
    if not kmer_counts_dict:
        print("Error: No k-mer count files found!")
        sys.exit(1)
    
    print(f"Loaded k-mer counts for k values: {list(kmer_counts_dict.keys())}")
    
    # 计算bin统计量
    print("\n[Step 2/2] Calculating bin statistics...")
    df = calculate_bin_statistics(
        args.genome, 
        kmer_counts_dict, 
        args.bin_size,
        args.chromosome
    )
    
    # 保存结果
    print(f"\n[Saving] Writing to {args.output}...")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    
    # 总结
    print("\n" + "=" * 60)
    print("Feature generation completed!")
    print("=" * 60)
    print(f"Total bins: {len(df)}")
    print(f"Chromosomes: {df['chromosome'].nunique()}")
    print(f"\nOutput columns:")
    for col in df.columns:
        print(f"  - {col}")
    print(f"\nOutput file: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.1f} KB")
    print("=" * 60)
    
    # 显示前几行
    print("\nFirst 3 rows preview:")
    print(df.head(3).to_string())
    
    # 显示统计信息
    print("\nFeature statistics:")
    feature_cols = [col for col in df.columns if col not in ['chromosome', 'start', 'end', 'has_cen']]
    print(df[feature_cols].describe().to_string())

if __name__ == '__main__':
    main()


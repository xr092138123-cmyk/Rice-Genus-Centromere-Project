#!/usr/bin/env python3
"""
将预测结果转换为BED格式，方便在IGV等工具中可视化
"""

import json
import argparse
import os
import sys

def json_to_bed(json_file, bed_file, min_prob=0.5, top_n=None):
    """
    将预测结果转换为BED格式
    
    Args:
        json_file: 预测结果JSON文件
        bed_file: 输出BED文件
        min_prob: 最小概率阈值
        top_n: 只保留top N个区域（按概率排序）
    """
    print(f"Reading predictions from {json_file}...")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 处理单个或多个结果
    if isinstance(data, dict):
        results = [data]
    else:
        results = data
    
    bed_entries = []
    total_regions = 0
    filtered_regions = 0
    
    for result in results:
        csv_file = result.get('csv_file', 'unknown')
        # 尝试从文件名提取染色体信息
        chrom = result.get('chromosome', 'chr1')
        if chrom == 'chr1' and 'chr' in csv_file:
            # 尝试从文件名提取染色体
            import re
            match = re.search(r'chr[\dXY]+', csv_file)
            if match:
                chrom = match.group()
        
        regions = result.get('predicted_regions', [])
        total_regions += len(regions)
        
        for region in regions:
            if region['avg_prob'] >= min_prob:
                bed_entries.append({
                    'chrom': chrom,
                    'start': region['start_pos'],
                    'end': region['end_pos'],
                    'name': f"centromere_{region['start_pos']}_{region['end_pos']}",
                    'score': int(region['avg_prob'] * 1000),  # 转换为0-1000
                    'avg_prob': region['avg_prob'],
                    'max_prob': region['max_prob'],
                    'length': region['length_bp']
                })
                filtered_regions += 1
    
    # 按概率排序
    bed_entries.sort(key=lambda x: x['avg_prob'], reverse=True)
    
    # 如果指定了top_n，只保留前N个
    if top_n:
        bed_entries = bed_entries[:top_n]
    
    # 写入BED文件
    print(f"Writing to {bed_file}...")
    with open(bed_file, 'w') as f:
        # BED6格式头部（可选）
        f.write('track name="Centromeres" description="Predicted centromere regions" useScore=1\n')
        
        for entry in bed_entries:
            # BED6格式: chrom start end name score strand
            f.write(f"{entry['chrom']}\t{entry['start']}\t{entry['end']}\t"
                   f"{entry['name']}\t{entry['score']}\t.\n")
    
    print(f"\nConversion completed!")
    print(f"  Total predicted regions: {total_regions}")
    print(f"  Filtered by prob >= {min_prob}: {filtered_regions}")
    print(f"  Written to BED: {len(bed_entries)}")
    
    return bed_entries

def json_to_detailed_bed(json_file, bed_file, min_prob=0.5):
    """
    生成包含详细信息的BED文件（BED12+格式）
    """
    print(f"Reading predictions from {json_file}...")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        results = [data]
    else:
        results = data
    
    print(f"Writing detailed BED to {bed_file}...")
    with open(bed_file, 'w') as f:
        # 写入头部
        f.write('#chrom\tstart\tend\tname\tscore\tstrand\t'
               'avg_prob\tmax_prob\tlength_bp\tlength_bins\n')
        
        for result in results:
            chrom = result.get('chromosome', 'chr1')
            regions = result.get('predicted_regions', [])
            
            for region in regions:
                if region['avg_prob'] >= min_prob:
                    f.write(f"{chrom}\t{region['start_pos']}\t{region['end_pos']}\t"
                           f"centromere_{region['start_pos']}\t"
                           f"{int(region['avg_prob']*1000)}\t.\t"
                           f"{region['avg_prob']:.4f}\t{region['max_prob']:.4f}\t"
                           f"{region['length_bp']}\t{region['length_bins']}\n")
    
    print("Detailed BED file created!")

def main():
    parser = argparse.ArgumentParser(
        description='Convert prediction JSON to BED format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # 基本用法
  python predictions_to_bed.py predictions.json centromeres.bed
  
  # 设置最小概率阈值
  python predictions_to_bed.py predictions.json centromeres.bed --min-prob 0.7
  
  # 只保留top 5个区域
  python predictions_to_bed.py predictions.json centromeres.bed --top-n 5
  
  # 生成详细的BED文件
  python predictions_to_bed.py predictions.json centromeres_detail.bed --detailed
        """
    )
    parser.add_argument('json_file', help='Input JSON file with predictions')
    parser.add_argument('bed_file', help='Output BED file')
    parser.add_argument('--min-prob', type=float, default=0.5,
                       help='Minimum probability threshold (default: 0.5)')
    parser.add_argument('--top-n', type=int, default=None,
                       help='Keep only top N regions by probability (optional)')
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed BED with extra columns')
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("Predictions to BED Converter")
    print("=" * 60)
    print(f"Input: {args.json_file}")
    print(f"Output: {args.bed_file}")
    print(f"Min probability: {args.min_prob}")
    if args.top_n:
        print(f"Top N: {args.top_n}")
    print(f"Detailed format: {args.detailed}")
    print("=" * 60)
    print()
    
    # 转换
    if args.detailed:
        json_to_detailed_bed(args.json_file, args.bed_file, args.min_prob)
    else:
        bed_entries = json_to_bed(args.json_file, args.bed_file, args.min_prob, args.top_n)
        
        # 显示摘要
        if bed_entries:
            print("\nTop 5 regions:")
            for i, entry in enumerate(bed_entries[:5], 1):
                print(f"  {i}. {entry['chrom']}:{entry['start']}-{entry['end']} "
                     f"(prob: {entry['avg_prob']:.3f}, length: {entry['length']:,} bp)")
    
    print("\n" + "=" * 60)
    print(f"BED file created: {args.bed_file}")
    print("=" * 60)
    print("\nYou can now load this file in IGV or other genome browsers.")

if __name__ == '__main__':
    main()


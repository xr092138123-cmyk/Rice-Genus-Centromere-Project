#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
from shutil import which

def check_tool_exists(name):
    """检查所需命令行工具是否存在于系统PATH中"""
    if which(name) is None:
        print(f"错误: 必需的工具 '{name}' 未找到或不在您的系统PATH中。")
        print("请安装 samtools 并确保它在PATH中。")
        sys.exit(1)

def parse_pos_filename(filepath):
    """
    从坐标文件名中解析材料名、染色体和窗口信息。
    示例: AA_Ogla_hap1.Chr01.w100_top20_regions_list.txt -> ('AA_Ogla_hap1', 'Chr01', 'w100')
    """
    filename = os.path.basename(filepath)
    parts = filename.split('.')
    if len(parts) < 3:
        print(f"警告: 文件名格式不符合预期，跳过文件: {filename}")
        return None, None, None
        
    material_name = parts[0]
    chromosome = parts[1]
    # 从 'w100_top20_regions_list' 中提取 'w100'
    window_info = parts[2].split('_')[0]
    
    return material_name, chromosome, window_info

def extract_top1_regions(pos_file):
    """
    从坐标文件中读取并返回所有第二列为 'Top1' 的区域坐标。
    """
    top1_regions = []
    try:
        with open(pos_file, 'r') as f:
            # 跳过表头
            next(f)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                columns = line.split() # 按空格或制表符分割
                if len(columns) >= 2 and columns[1] == 'Top1':
                    region = columns[0]
                    top1_regions.append(region)
    except FileNotFoundError:
        print(f"错误: 坐标文件未找到: {pos_file}")
        return []
    except Exception as e:
        print(f"读取坐标文件 {pos_file} 时出错: {e}")
        return []
        
    return top1_regions

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="根据指定的坐标文件列表，从参考基因组中提取'Top1'区域的序列。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--pos_list', required=True, 
                        help="必填: 包含坐标文件路径的列表文件。\n"
                             "每行一个坐标文件的绝对或相对路径。")
    parser.add_argument('--ref_dir', required=True, 
                        help="必填: 包含所有材料参考基因组 (.fasta 和 .fasta.fai) 的目录。")
    parser.add_argument('--out_dir', required=True, 
                        help="必填: 用于存放输出 FASTA 文件的目录。")
    
    args = parser.parse_args()

    # 1. 检查依赖工具
    check_tool_exists('samtools')

    # 2. 检查并创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 3. 读取坐标文件列表
    try:
        with open(args.pos_list, 'r') as f_list:
            pos_files = [line.strip() for line in f_list if line.strip()]
    except FileNotFoundError:
        print(f"错误: 坐标列表文件未找到: {args.pos_list}")
        sys.exit(1)

    print(f"总共找到 {len(pos_files)} 个坐标文件进行处理。")

    # 4. 遍历并处理每个坐标文件
    for pos_file_path in pos_files:
        print(f"\n--- 正在处理: {os.path.basename(pos_file_path)} ---")
        
        # 解析文件名获取信息
        material, chrom, window = parse_pos_filename(pos_file_path)
        if not all([material, chrom, window]):
            continue

        # 提取需要抽提的Top1区域
        regions_to_extract = extract_top1_regions(pos_file_path)
        if not regions_to_extract:
            print(f"在 {os.path.basename(pos_file_path)} 中未找到 'Top1' 区域，跳过。")
            continue
        
        print(f"找到 {len(regions_to_extract)} 个 'Top1' 区域。")

        # 构建参考基因组和输出文件的路径
        ref_fasta_path = os.path.join(args.ref_dir, f"{material}.fasta")
        output_fasta_path = os.path.join(args.out_dir, f"{material}.{chrom}.{window}.Top1.fa")

        # 检查参考基因组文件是否存在
        if not os.path.exists(ref_fasta_path):
            print(f"警告: 找不到对应的参考基因组: {ref_fasta_path}，跳过。")
            continue
        
        # 检查FASTA索引文件是否存在
        if not os.path.exists(ref_fasta_path + '.fai'):
            print(f"警告: 找不到参考基因组索引文件: {ref_fasta_path}.fai")
            print("请先使用 'samtools faidx' 命令为该基因组创建索引。跳过。")
            continue

        # 构建 samtools 命令
        # 将所有区域作为参数传递给单次 samtools 调用，效率更高
        command = ['samtools', 'faidx', ref_fasta_path] + regions_to_extract
        
        print(f"执行命令提取序列...")
        try:
            # 执行命令并捕获输出
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True  # 如果samtools返回错误码，则抛出异常
            )
            
            # 将提取的序列写入输出文件
            with open(output_fasta_path, 'w') as f_out:
                f_out.write(result.stdout)
            
            print(f"成功提取序列并保存到: {output_fasta_path}")

        except subprocess.CalledProcessError as e:
            print(f"错误: 'samtools faidx' 命令执行失败。")
            print(f"返回码: {e.returncode}")
            print(f"标准错误输出:\n{e.stderr}")
        except Exception as e:
            print(f"执行 samtools 时发生未知错误: {e}")

    print("\n所有文件处理完成。")

if __name__ == "__main__":
    main()

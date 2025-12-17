#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from collections import Counter
import sys

def parse_region_string(region_str):
    """
    解析 'Chr:start-end' 格式的字符串。
    返回 (chromosome, start, end)。
    """
    match = re.match(r'([^:]+):(\d+)-(\d+)', region_str)
    if match:
        chrom, start, end = match.groups()
        return chrom, int(start), int(end)
    return None, None, None

def format_region_tuple(region_tuple):
    """
    将 (chromosome, start, end) 元组格式化为字符串。
    """
    return f"{region_tuple[0]}:{region_tuple[1]}-{region_tuple[2]}"

def analyze_moddotplot(bed_file, top_n, output_prefix):
    """
    主分析函数
    """
    print(f"--- 开始分析文件: {bed_file} ---")

    # 1. 读取并过滤数据 (Identity >= 86%)
    print("步骤 1/5: 读取并过滤输入文件 (Identity >= 86%)...")
    high_identity_records = []
    cen_info = None # 存储着丝粒区域信息 (chr, start, end)

    try:
        with open(bed_file, 'r') as f:
            for i, line in enumerate(f, 1):
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) < 7:
                    print(f"警告: 第 {i} 行格式不正确，已跳过: {line.strip()}", file=sys.stderr)
                    continue

                try:
                    identity = float(parts[6])
                    if identity >= 86.0:
                        if cen_info is None:
                            cen_info = parse_region_string(parts[0])
                            if cen_info[0] is None:
                                print(f"错误: 无法解析第一列的着丝粒区域格式: {parts[0]}", file=sys.stderr)
                                sys.exit(1)
                        
                        record = {
                            "q_name": parts[0], "q_start": int(parts[1]), "q_end": int(parts[2]),
                            "r_name": parts[3], "r_start": int(parts[4]), "r_end": int(parts[5]),
                            "identity": identity
                        }
                        high_identity_records.append(record)
                except (ValueError, IndexError):
                    print(f"警告: 第 {i} 行数据无法解析，已跳过: {line.strip()}", file=sys.stderr)
                    continue
    except FileNotFoundError:
        print(f"错误: 输入文件未找到: {bed_file}", file=sys.stderr)
        sys.exit(1)

    if not high_identity_records:
        print("错误: 文件中没有找到 Identity >= 86% 的有效数据行。", file=sys.stderr)
        sys.exit(1)
        
    cen_chr, cen_start, cen_end = cen_info
    cen_length = cen_end - cen_start + 1
    print(f"找到 {len(high_identity_records)} 条高相似度记录。")
    print(f"着丝粒区域: {format_region_tuple(cen_info)}, 总长度: {cen_length} bp")

    # 2. 转换坐标并生成结果文件1
    abs_coords_file = f"{output_prefix}_abs_coords.txt"
    print(f"\n步骤 2/5: 转换坐标并生成文件 -> {abs_coords_file}")
    
    abs_coord_pairs = []
    with open(abs_coords_file, 'w') as f_out:
        f_out.write("region1\tregion2\tidentity\n")
        for record in high_identity_records:
            abs_q_start = cen_start + record["q_start"] - 1
            abs_q_end = cen_start + record["q_end"] - 1
            abs_r_start = cen_start + record["r_start"] - 1
            abs_r_end = cen_start + record["r_end"] - 1
            q_region_tuple = (cen_chr, abs_q_start, abs_q_end)
            r_region_tuple = (cen_chr, abs_r_start, abs_r_end)
            abs_coord_pairs.append((q_region_tuple, r_region_tuple))
            f_out.write(f"{format_region_tuple(q_region_tuple)}\t{format_region_tuple(r_region_tuple)}\t{record['identity']:.4f}\n")
    print("坐标转换完成。")

    # 3. 迭代计算 Top N 重复
    print(f"\n步骤 3/5: 迭代计算 Top {top_n} 重复区域...")
    top_repeats_results = []
    all_top_family_regions = []
    
    remaining_pairs = list(abs_coord_pairs)
    cumulative_percentage_total = 0.0

    for i in range(1, top_n + 1):
        if not remaining_pairs:
            print(f"警告: 在找到 Top {i-1} 后已无剩余数据可分析。")
            break

        print(f"  -> 正在计算 Top {i}...")
        
        all_regions_in_scope = []
        for q_region, r_region in remaining_pairs:
            all_regions_in_scope.append(q_region)
            all_regions_in_scope.append(r_region)
        
        if not all_regions_in_scope:
            break 

        region_counts = Counter(all_regions_in_scope)
        # 仍然使用出现次数最多的区域作为"种子"，来识别出家族
        top_seed_tuple, _ = region_counts.most_common(1)[0]
        
        # 构建家族：包含种子和所有与它相关的区域
        current_family_regions = set()
        for q_region, r_region in remaining_pairs:
            if q_region == top_seed_tuple or r_region == top_seed_tuple:
                current_family_regions.add(q_region)
                current_family_regions.add(r_region)

        # <<< CHANGE START >>>
        # 修正：拷贝数现在定义为家族中独特区域的总数
        # 这将确保与 _regions_list.txt 文件中的行数一致
        copy_number = len(current_family_regions)
        # <<< CHANGE END >>>

        # 收集当前家族的区域信息用于输出文件3
        top_id_str = f"Top{i}"
        for region_tuple in current_family_regions:
            all_top_family_regions.append({'region': region_tuple, 'top_id': top_id_str})

        cumulative_length = sum(end - start + 1 for _, start, end in current_family_regions)
        
        percentage_of_cen = (cumulative_length / cen_length) * 100
        cumulative_percentage_total += percentage_of_cen
        
        result_line = {
            "region": format_region_tuple(top_seed_tuple),
            "copy_number": copy_number, # 使用修正后的拷贝数
            "cumulative_length": cumulative_length,
            "percentage_of_cen": percentage_of_cen,
            "cumulative_percentage": cumulative_percentage_total
        }
        top_repeats_results.append(result_line)

        # 移除已归类的家族，准备下一次迭代
        remaining_pairs = [
            (q, r) for q, r in remaining_pairs 
            if q not in current_family_regions and r not in current_family_regions
        ]
    
    print("Top N 计算完成。")

    # 4. 生成结果文件2
    top_repeats_file = f"{output_prefix}_top{top_n}_repeats.txt"
    print(f"\n步骤 4/5: 生成 Top N 统计文件 -> {top_repeats_file}")

    with open(top_repeats_file, 'w') as f_out:
        header = "Top\tSeed_Region\tCopy_Number\tFamily_Cumulative_Length\tPercentage_of_Centromere\tCumulative_Percentage\n"
        f_out.write(header)
        for i, result in enumerate(top_repeats_results, 1):
            f_out.write(
                f"Top{i}\t"
                f"{result['region']}\t"
                f"{result['copy_number']}\t"
                f"{result['cumulative_length']}\t"
                f"{result['percentage_of_cen']:.4f}%\t"
                f"{result['cumulative_percentage']:.4f}%\n"
            )

    # 5. 生成结果文件3
    regions_list_file = f"{output_prefix}_top{top_n}_regions_list.txt"
    print(f"\n步骤 5/5: 生成 Top N 区域列表文件 -> {regions_list_file}")

    all_top_family_regions.sort(key=lambda x: x['region'])

    with open(regions_list_file, 'w') as f_out:
        f_out.write("Region\tTop_Family\n")
        for item in all_top_family_regions:
            region_str = format_region_tuple(item['region'])
            f_out.write(f"{region_str}\t{item['top_id']}\n")


    print(f"\n--- 分析完成 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="统计 moddotplot 分析结果，识别着丝粒中的主要重复单元。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("bed_file", help="moddotplot 分析结果的 bed 文件。")
    parser.add_argument("top_n", type=int, help="要统计的 top N 个重复区域的数量，例如 10。")
    parser.add_argument("output_prefix", help="输出文件的前缀，例如 'Chr1_cen_analysis'。")
    args = parser.parse_args()
    analyze_moddotplot(args.bed_file, args.top_n, args.output_prefix)
import os
import argparse
import pandas as pd
import sys
from collections import defaultdict
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np

parser = argparse.ArgumentParser(description="处理hicpro输出的matrix文件")
parser.add_argument('--matrix', type=str, required=True, help='矩阵文件')
parser.add_argument('--abs', type=str, required=True, help='bins信息文件')
parser.add_argument('--cen', type=str, required=True, help='着丝粒区域文件')

args = parser.parse_args()

def filter_intra_chromosomal_contacts(matrix_file, abs_file, output_dir):
    """
    从Hi-C contact matrix中提取每条染色体内部的contact记录。

    Args:
        matrix_file (str): matrix文件的路径。
                           格式: bin1_id  bin2_id  contact_value (以制表符或空格分隔)
        abs_file (str): abs文件的路径。
                        格式: chr_name  start  end  bin_id (以制表符或空格分隔)
        output_dir (str): 输出文件夹的路径，用于存储结果。
    """
    # --- 步骤 1: 读取abs文件，构建 bin -> 染色体 的映射 ---
    print(f"正在读取abs文件: {abs_file}...")
    bin_to_chr = {}
    try:
        with open(abs_file, 'r') as f:
            for line in f:
                # 使用split()处理制表符或多个空格
                parts = line.strip().split()
                if len(parts) < 4:
                    continue  # 跳过格式不正确的行
                chrom = parts[0]
                bin_id = int(parts[3])
                bin_to_chr[bin_id] = chrom
    except FileNotFoundError:
        print(f"错误: abs文件未找到 -> {abs_file}", file=sys.stderr)
        return
    except ValueError:
        print(f"错误: abs文件中的bin号'{parts[3]}'不是一个有效的整数。", file=sys.stderr)
        return
        
    print(f"成功构建映射，包含 {len(bin_to_chr)} 个bins。")

    # --- 步骤 2: 创建输出目录 ---
    print(f"正在准备输出目录: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 步骤 3: 处理matrix文件并写入输出 ---
    print(f"正在处理matrix文件: {matrix_file}...")
    
    # 用于存储每个染色体对应的文件句柄，避免重复打开文件
    output_file_handlers = {}
    
    line_count = 0
    intra_contact_count = 0

    try:
        with open(matrix_file, 'r') as f_matrix:
            for line in f_matrix:
                line_count += 1
                if line_count % 5000000 == 0:  # 每处理500万行打印一次进度
                    print(f"  ...已处理 {line_count} 行...")
                    
                parts = line.strip().split()
                if len(parts) < 3:
                    continue # 跳过格式不正确的行

                try:
                    bin1 = int(parts[0])
                    bin2 = int(parts[1])
                except ValueError:
                    print(f"警告: 在第 {line_count} 行跳过无效的bin号: {line.strip()}", file=sys.stderr)
                    continue

                # 查询两个bin所属的染色体
                chr1 = bin_to_chr.get(bin1)
                chr2 = bin_to_chr.get(bin2)
                
                # 核心逻辑：如果两个bin都存在于映射中，并且属于同一染色体
                if chr1 and chr1 == chr2:
                    intra_contact_count += 1
                    output_chr = chr1
                    
                    # 如果该染色体的文件还未打开，则打开它
                    if output_chr not in output_file_handlers:
                        output_path = os.path.join(output_dir, f"{output_chr}.tsv")
                        # 以写入模式'w'打开文件
                        output_file_handlers[output_chr] = open(output_path, 'w')
                        print(f"  -> 已创建文件: {output_path}")

                    # 将原始行写入对应的文件中
                    output_file_handlers[output_chr].write(line)

    except FileNotFoundError:
        print(f"错误: matrix文件未找到 -> {matrix_file}", file=sys.stderr)
        return
    finally:
        # --- 步骤 4: 关闭所有已打开的输出文件 ---
        print("正在关闭所有输出文件...")
        for handler in output_file_handlers.values():
            handler.close()
            
    print("\n处理完成！")
    print(f"总共处理了 {line_count} 行matrix记录。")
    print(f"共找到并输出了 {intra_contact_count} 条染色体内部contact记录。")
    print(f"结果已保存在目录: {output_dir}")


def calculate_marginal_sums(input_dir):
    """
    遍历一个文件夹中的所有TSV文件，为每个文件计算每个bin的边缘和（Marginal Sum）。
    边缘和 = 该bin与同一染色体上所有其他bin（包括自身）的contact values之和。

    Args:
        input_dir (str): 包含contact TSV文件的文件夹路径。
                         输入文件格式: bin1  bin2  contact_value (制表符分隔)
    
    Returns:
        None. 结果会直接写入到输入文件夹中，
              文件名为 [原文件名].eachbin_contact.tsv。
    """
    # --- 步骤 1: 检查输入目录是否存在 ---
    if not os.path.isdir(input_dir):
        print(f"错误: 目录 '{input_dir}' 不存在。", file=sys.stderr)
        return

    print(f"开始处理目录: {input_dir}")

    # --- 步骤 2: 遍历目录中的所有文件 ---
    # 过滤掉已经是结果文件的文件，防止重复处理
    files_to_process = [
        f for f in os.listdir(input_dir)
        if f.endswith('.tsv') and not f.endswith('.eachbin_sum.tsv') and not f.endswith('.eachbin_contact.tsv')
    ]
    
    if not files_to_process:
        print("在目录中没有找到需要处理的 .tsv 文件。")
        return

    for filename in files_to_process:
        input_file_path = os.path.join(input_dir, filename)
        
        if not os.path.isfile(input_file_path):
            continue
            
        print(f"\n--- 正在处理文件: {filename} ---")

        # --- 步骤 3: 为每个文件计算边缘和 ---
        bin_contact_sums = defaultdict(float)

        try:
            with open(input_file_path, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    
                    if len(parts) < 3:
                        continue
                    
                    try:
                        bin1 = int(parts[0])
                        bin2 = int(parts[1])
                        contact_value = float(parts[2])
                    except ValueError:
                        print(f"  警告: 第 {i+1} 行数据类型错误，已跳过。")
                        continue

                    # 【核心逻辑】：累加交互值 (考虑矩阵对称性)
                    # 1. 为 bin1 累加
                    bin_contact_sums[bin1] += contact_value
                    
                    # 2. 为 bin2 累加 (排除对角线自身交互)
                    if bin1 != bin2:
                        bin_contact_sums[bin2] += contact_value
        
        except FileNotFoundError:
            print(f"  错误: 文件无法读取 {input_file_path}", file=sys.stderr)
            continue
            
        print(f"  ...读取完成，正在生成结果。")

        # --- 步骤 4: 准备输出数据 ---
        if not bin_contact_sums:
            print("  文件中没有有效数据，跳过生成输出文件。")
            continue

        results = []
        for bin_id, total_sum in bin_contact_sums.items():
            results.append((bin_id, total_sum))
            
        # 按bin号排序
        results.sort(key=lambda x: x[0])
        
        # --- 步骤 5: 写入到新的TSV文件 ---
        base_name = os.path.splitext(filename)[0]
        
        # 【修改点】：文件名后缀改为 .eachbin_contact.tsv
        output_filename = f"{base_name}.eachbin_contact.tsv" 
        output_file_path = os.path.join(input_dir, output_filename)
        
        try:
            with open(output_file_path, 'w') as f_out:
                # 表头保持为 marginal_sum 以准确描述数据含义，或者改为 contact_value 也可以
                f_out.write("bin_id\tmarginal_sum\n")
                
                # 写入数据
                for bin_id, sum_val in results:
                    f_out.write(f"{bin_id}\t{sum_val:.6f}\n")
            print(f"  -> 结果已成功保存到: {output_filename}")
        except IOError as e:
            print(f"  错误: 无法写入输出文件 {output_file_path}。原因: {e}", file=sys.stderr)

    print("\n所有文件处理完成！")

def map_bin_locations(input_dir, matrix_file):
    """
    将bin的平均contact值文件与bin的绝对位置文件进行合并。

    Args:
        input_dir (str): 包含'.eachbin_contact.tsv'文件的文件夹路径。
        matrix_file (str): 包含bin绝对位置信息的文件路径。
                           格式: chr start end bin_id (制表符或空格分隔)。
                           注意：根据您的描述，此参数名为'matrix'。
    """
    # --- 步骤 1: 检查输入路径 ---
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录 '{input_dir}' 不存在或不是一个目录。", file=sys.stderr)
        return
    if not os.path.isfile(matrix_file):
        print(f"错误: 位置文件 '{matrix_file}' 不存在或不是一个文件。", file=sys.stderr)
        return

    # --- 步骤 2: 读取位置文件，构建 bin_id -> [chr, start, end] 的映射 ---
    print(f"正在读取位置文件: {matrix_file} ...")
    bin_to_location = {}
    try:
        with open(matrix_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    # 将第四列(bin_id)作为key，前三列作为value
                    bin_id = int(parts[3])
                    location_info = parts[0:3]  # 这是一个列表 [chr, start, end]
                    bin_to_location[bin_id] = location_info
                except ValueError:
                    # 如果bin_id不是整数，则跳过该行
                    print(f"警告: 在位置文件中发现无效的bin_id，已跳过: '{line.strip()}'", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"错误: 读取位置文件时发生错误: {e}", file=sys.stderr)
        return
    
    if not bin_to_location:
        print("警告: 未从位置文件中加载任何数据。请检查文件格式和内容。", file=sys.stderr)
        return
        
    print(f"位置映射构建完成，共加载 {len(bin_to_location)} 个bins的信息。")

    # --- 步骤 3: 遍历目录，处理每个.eachbin_contact.tsv文件 ---
    print(f"\n开始扫描目录: {input_dir}")
    
    files_processed = 0
    for filename in os.listdir(input_dir):
        if filename.endswith(".eachbin_contact.tsv"):
            files_processed += 1
            input_path = os.path.join(input_dir, filename)
            print(f"--- 正在处理文件: {filename} ---")
            
            # 生成输出文件名
            # os.path.splitext可以安全地分割文件名和扩展名
            base_name = os.path.splitext(filename)[0] 
            output_filename = f"{base_name}.abs"
            output_path = os.path.join(input_dir, output_filename)

            try:
                # 使用 with 语句同时打开输入和输出文件
                with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
                    # 写入表头到输出文件
                    f_out.write("chr\tstart\tend\tavg_contact\tbin_id\n")
                    
                    # 跳过输入文件的表头
                    try:
                        next(f_in)
                    except StopIteration:
                        # 文件为空，跳过
                        print("  文件为空，已跳过。")
                        continue

                    # 逐行处理数据
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) < 2:
                            continue
                        
                        try:
                            bin_id = int(parts[0])
                            contact_val = parts[1]
                        except ValueError:
                            print(f"  警告: 在文件中发现无效数据行，已跳过: '{line.strip()}'", file=sys.stderr)
                            continue
                        
                        # 在映射中查找位置信息
                        location_info = bin_to_location.get(bin_id)
                        
                        if location_info:
                            # 如果找到，格式化并写入新行
                            chrom, start, end = location_info
                            f_out.write(f"{chrom}\t{start}\t{end}\t{contact_val}\t{bin_id}\n")
                        else:
                            # 如果未找到，打印警告
                            print(f"  警告: bin_id '{bin_id}' 在位置文件中未找到，已跳过。", file=sys.stderr)

                print(f"  -> 结果已成功保存到: {output_filename}")

            except Exception as e:
                print(f"  处理文件 {filename} 时发生错误: {e}", file=sys.stderr)

    if files_processed == 0:
        print("在目录中没有找到以 '.eachbin_contact.tsv' 结尾的文件。")
    
    print("\n所有文件处理完成！")

def split_chromosome_arms(cen_file, genome_size_file, output_tsv):
    """
    根据着丝粒位置将染色体臂划分为50个等份的bins。

    Args:
        cen_file (str): 记录着丝粒位置的BED文件路径。
                        格式: chr start end
        genome_size_file (str): samtools faidx生成的.fai文件路径。
                                格式: chr length ...
        output_tsv (str): 输出TSV文件的路径。
    """
    # --- 步骤 1: 读取并解析输入文件 ---
    print("步骤 1: 正在读取输入文件...")

    # 读取基因组大小文件 (.fai)
    genome_lengths = {}
    try:
        with open(genome_size_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    chrom = parts[0]
                    length = int(parts[1])
                    genome_lengths[chrom] = length
    except FileNotFoundError:
        print(f"错误: 基因组大小文件 '{genome_size_file}' 未找到。", file=sys.stderr)
        return
    except ValueError:
        print(f"错误: 基因组大小文件 '{genome_size_file}' 格式不正确。", file=sys.stderr)
        return
    print(f"  -> 成功加载 {len(genome_lengths)} 条染色体的长度信息。")

    # 读取着丝粒位置文件 (BED)
    centromeres = {}
    try:
        with open(cen_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    chrom = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    centromeres[chrom] = (start, end)
    except FileNotFoundError:
        print(f"错误: 着丝粒文件 '{cen_file}' 未找到。", file=sys.stderr)
        return
    except ValueError:
        print(f"错误: 着丝粒文件 '{cen_file}' 格式不正确。", file=sys.stderr)
        return
    print(f"  -> 成功加载 {len(centromeres)} 个着丝粒的位置信息。")

    # --- 步骤 2: 计算并生成所有区域 ---
    print("\n步骤 2: 正在计算染色体臂分区...")
    
    all_regions = []
    
    # 以genome_lengths为准，遍历所有染色体，并按染色体名称排序以保证输出顺序一致
    sorted_chroms = sorted(genome_lengths.keys())

    for chrom in sorted_chroms:
        if chrom not in centromeres:
            print(f"  警告: 染色体 '{chrom}' 在着丝粒文件中没有记录，已跳过。", file=sys.stderr)
            continue

        chr_len = genome_lengths[chrom]
        cen_start, cen_end = centromeres[chrom]
        
        print(f"  正在处理 {chrom}...")

        # --- 左臂 (p-arm) ---
        left_arm_len = cen_start
        if left_arm_len > 0:
            # 使用浮点数计算bin的大小以保证精度
            left_bin_size = left_arm_len / 50.0
            for i in range(50):
                # 计算每个bin的起始和结束位置
                start = int(i * left_bin_size)
                end = int((i + 1) * left_bin_size)
                # 最后一个bin的末端对齐到cen_start
                if i == 49:
                    end = cen_start
                
                region_id = f"left_arm_{i + 1}"
                all_regions.append((chrom, start, end, region_id))

        # --- 着丝粒 (Centromere) ---
        all_regions.append((chrom, cen_start, cen_end, "cen"))

        # --- 右臂 (q-arm) ---
        right_arm_len = chr_len - cen_end
        if right_arm_len > 0:
            right_bin_size = right_arm_len / 50.0
            for i in range(50):
                start = int(cen_end + (i * right_bin_size))
                end = int(cen_end + ((i + 1) * right_bin_size))
                # 确保最后一个bin的末端恰好是染色体的总长度
                if i == 49:
                    end = chr_len
                    
                region_id = f"right_arm_{i + 1}"
                all_regions.append((chrom, start, end, region_id))

    # --- 步骤 3: 写入输出文件 ---
    print(f"\n步骤 3: 正在将结果写入到 '{output_tsv}'...")
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_tsv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_tsv, 'w') as f_out:
            # 写入表头
            f_out.write("chr\tstart\tend\tregion_id\n")
            # 写入数据
            for region in all_regions:
                f_out.write(f"{region[0]}\t{region[1]}\t{region[2]}\t{region[3]}\n")
    except IOError as e:
        print(f"错误: 无法写入输出文件。原因: {e}", file=sys.stderr)
        return

    print("\n处理完成！")

def merge_abs_files(input_dir, output_file):
    """
    合并一个文件夹中所有以 '.eachbin_contact.abs' 结尾的文件。

    Args:
        input_dir (str): 包含'.eachbin_contact.abs'文件的文件夹路径。
        output_file (str): 合并后输出文件的完整路径和名称。
    """
    # --- 步骤 1: 检查输入目录是否存在 ---
    if not os.path.isdir(input_dir):
        print(f"错误: 目录 '{input_dir}' 不存在。", file=sys.stderr)
        return

    print(f"正在扫描目录: {input_dir}")
    suffix = ".eachbin_contact.abs"

    # --- 步骤 2: 查找所有符合条件的文件 ---
    files_to_merge = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(suffix) and os.path.isfile(os.path.join(input_dir, f))
    ]

    if not files_to_merge:
        print(f"在目录中没有找到以 '{suffix}' 结尾的文件。")
        return
    
    files_to_merge.sort()

    print(f"找到 {len(files_to_merge)} 个文件准备合并:")
    for f_path in files_to_merge:
        print(f"  - {os.path.basename(f_path)}")

    # --- 步骤 3: 使用传入的参数定义输出文件路径 ---
    output_filepath = output_file
    
    # 安全检查：如果要合并的文件列表中包含了输出文件本身，则跳过
    # os.path.abspath 用于获取绝对路径，确保比较的准确性
    if os.path.abspath(output_filepath) in [os.path.abspath(f) for f in files_to_merge]:
        print(f"警告: 输出文件 '{os.path.basename(output_filepath)}' 已存在于待合并列表中，将其移除。")
        # 从待合并列表中移除输出文件
        files_to_merge = [f for f in files_to_merge if os.path.abspath(f) != os.path.abspath(output_filepath)]
        if not files_to_merge:
            print("移除输出文件后，没有其他文件可供合并。")
            return

    print(f"\n合并后的文件将保存为: {output_filepath}")

    # --- 步骤 4: 执行合并操作 ---
    try:
        # 确保输出文件所在的目录存在
        output_dir = os.path.dirname(output_filepath)
        if output_dir: # 如果路径包含目录
            os.makedirs(output_dir, exist_ok=True)

        with open(output_filepath, 'w') as f_out:
            is_header_written = False
            
            for file_path in files_to_merge:
                with open(file_path, 'r') as f_in:
                    if not is_header_written:
                        shutil.copyfileobj(f_in, f_out)
                        is_header_written = True
                    else:
                        next(f_in) 
                        shutil.copyfileobj(f_in, f_out)
                
                # 在每个文件内容后追加换行符，以防源文件末尾没有换行符导致行合并
                f_out.write('\n')

    except IOError as e:
        print(f"错误: 读写文件时发生错误。原因: {e}", file=sys.stderr)
        return

    print("\n所有文件合并完成！")


def bin_chromosome_coordinates(contact_abs, cen, num_bins_per_arm=50):
    """
    以着丝粒中点为原点，将左右臂各划分为指定数量的windows，并计算每个window的平均contact值。

    Args:
        contact_abs (str): 输入TSV文件路径。
        cen (str): 着丝粒BED文件路径。
        num_bins_per_arm (int): 单侧臂的窗口数量，默认为50。总窗口数为 2 * num_bins_per_arm。
    """
    total_bins = num_bins_per_arm * 2
    
    # --- 步骤 1: 读取着丝粒信息 ---
    print("步骤 1: 读取着丝粒信息...")
    centromeres = {}
    if not os.path.isfile(cen):
        print(f"错误: 文件 '{cen}' 不存在。", file=sys.stderr)
        return

    try:
        with open(cen, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    chrom = parts[0]
                    start, end = int(parts[1]), int(parts[2])
                    cen_mid = (start + end) / 2.0
                    centromeres[chrom] = cen_mid
    except Exception as e:
        print(f"读取着丝粒文件出错: {e}", file=sys.stderr)
        return

    # --- 步骤 2: 扫描获取染色体长度 ---
    print("步骤 2: 扫描染色体长度...")
    chr_lengths = {}
    if not os.path.isfile(contact_abs):
        print(f"错误: 文件 '{contact_abs}' 不存在。", file=sys.stderr)
        return

    try:
        with open(contact_abs, 'r') as f:
            next(f) # 跳过表头
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                chrom = parts[0]
                try:
                    end_pos = int(parts[2])
                    if chrom not in chr_lengths or end_pos > chr_lengths[chrom]:
                        chr_lengths[chrom] = end_pos
                except ValueError: continue
    except Exception as e:
        print(f"读取contact文件出错: {e}", file=sys.stderr)
        return

    # --- 步骤 3: 聚合数据到Windows ---
    print(f"步骤 3: 正在将数据聚合到 {total_bins} 个窗口 (左{num_bins_per_arm} + 右{num_bins_per_arm})...")
    
    # 数据结构: chromosome_bins[chrom][bin_index] = [val1, val2, ...]
    # 使用 sum 和 count 来节省内存
    # bin_data[chrom][window_idx] = {'sum': 0.0, 'count': 0}
    bin_data = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'count': 0}))
    
    try:
        with open(contact_abs, 'r') as f_in:
            next(f_in) # 跳过表头
            
            for line in f_in:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                chrom = parts[0]
                # 跳过未知染色体
                if chrom not in centromeres or chrom not in chr_lengths:
                    continue

                try:
                    start, end = int(parts[1]), int(parts[2])
                    contact_val = float(parts[3]) # 假设 avg_contact 在第4列
                except ValueError:
                    continue

                cen_mid = centromeres[chrom]
                chr_len = chr_lengths[chrom]
                bin_midpoint = (start + end) / 2.0
                
                window_idx = -1

                # --- 窗口映射逻辑 ---
                if bin_midpoint < cen_mid:
                    # 左臂: [0, cen_mid) 映射到 index 0 ~ 49
                    # 比例 0.0 ~ 1.0
                    ratio = bin_midpoint / cen_mid if cen_mid > 0 else 0
                    # 计算索引
                    idx = int(ratio * num_bins_per_arm)
                    # 边界修正：如果正好是 cen_mid (ratio=1.0)，归入最后一个格子
                    if idx >= num_bins_per_arm:
                        idx = num_bins_per_arm - 1
                    window_idx = idx
                    
                else:
                    # 右臂: [cen_mid, chr_len] 映射到 index 50 ~ 99
                    right_len = chr_len - cen_mid
                    ratio = (bin_midpoint - cen_mid) / right_len if right_len > 0 else 0
                    
                    # 基础索引是 50
                    idx = num_bins_per_arm + int(ratio * num_bins_per_arm)
                    # 边界修正：如果正好是 chr_len (ratio=1.0)，归入最后一个格子
                    if idx >= total_bins:
                        idx = total_bins - 1
                    window_idx = idx

                # 累加数据
                if 0 <= window_idx < total_bins:
                    bin_data[chrom][window_idx]['sum'] += contact_val
                    bin_data[chrom][window_idx]['count'] += 1

    except Exception as e:
        print(f"处理数据时出错: {e}", file=sys.stderr)
        return

    # --- 步骤 4: 计算平均值并输出 ---
    base_name = os.path.splitext(os.path.basename(contact_abs))[0]
    #output_file = os.path.join(os.path.dirname(contact_abs), f"{base_name}.binned_{num_bins_per_arm}x2.tsv")
    output_file = os.path.join(os.path.dirname(contact_abs), f"{base_name}.normalized.tsv")

    print(f"步骤 4: 正在写入结果至 '{output_file}'...")

    try:
        with open(output_file, 'w') as f_out:
            # 写入新表头
            # Normalized_Coordinate: -1.0 到 0 到 1.0 用于绘图X轴
            f_out.write("Chrom\tWindow_Index\tNormalized_Coordinate\tAvg_Contact_Value\n")
            
            # 遍历每条染色体
            for chrom in sorted(bin_data.keys()):
                # 遍历所有可能的窗口 (0 到 99)
                # 即使某些窗口没有数据，也可以输出 0 或者跳过，这里选择输出 0
                for i in range(total_bins):
                    stats = bin_data[chrom].get(i, {'sum': 0.0, 'count': 0})
                    
                    avg_val = 0.0
                    if stats['count'] > 0:
                        avg_val = stats['sum'] / stats['count']
                    
                    # 计算归一化坐标用于绘图 (X轴)
                    # 0 -> -1.0
                    # 50 -> 0.0
                    # 100 -> 1.0
                    # 公式: (当前索引 - 半数索引) / 半数索引
                    norm_coord = (i - num_bins_per_arm) / float(num_bins_per_arm)
                    
                    f_out.write(f"{chrom}\t{i}\t{norm_coord:.4f}\t{avg_val:.6f}\n")
                    
    except Exception as e:
        print(f"写入文件出错: {e}", file=sys.stderr)
        return

    print("处理完成！")


def plot_chromosome_contacts(input_abs_contact, output_dir):
    """
    为TSV文件中的每条染色体绘制contact profile折线图。
    
    输入文件格式: Chrom, Window_Index, Normalized_Coordinate, Avg_Contact_Value

    Args:
        input_abs_contact (str): TSV文件路径。
        output_dir (str): 输出图片文件夹。
    """
    # --- 步骤 1: 验证输入并创建输出目录 ---
    if not os.path.isfile(input_abs_contact):
        print(f"错误: 输入文件 '{input_abs_contact}' 不存在。", file=sys.stderr)
        return
        
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录 '{output_dir}' 已准备就绪。")
    except OSError as e:
        print(f"错误: 无法创建输出目录 '{output_dir}'. 原因: {e}", file=sys.stderr)
        return

    # --- 步骤 2: 读取并处理数据 ---
    print(f"正在读取数据文件: {input_abs_contact}")
    try:
        df = pd.read_csv(input_abs_contact, sep='\t')
        df['Normalized_Coordinate'] = pd.to_numeric(df['Normalized_Coordinate'])
        df['Avg_Contact_Value'] = pd.to_numeric(df['Avg_Contact_Value'])
    except Exception as e:
        print(f"错误: 读取或解析文件时出错。请检查文件格式。错误信息: {e}", file=sys.stderr)
        return

    # --- 步骤 3: 按染色体分组并循环绘图 ---
    chromosomes = df['Chrom'].unique()
    print(f"在文件中找到 {len(chromosomes)} 条染色体，开始绘图...")

    for chrom in chromosomes:
        chrom_df = df[df['Chrom'] == chrom].copy()
        chrom_df.sort_values('Normalized_Coordinate', inplace=True)
        
        print(f"  -> 正在绘制: {chrom}")

        # --- 设置绘图环境 ---
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            plt.style.use('default') 

        fig, ax = plt.subplots(figsize=(10, 5)) 

        # --- 绘制主折线图 ---
        sns.lineplot(
            x='Normalized_Coordinate', 
            y='Avg_Contact_Value', 
            data=chrom_df, 
            ax=ax,
            linewidth=2.5,
            color='black' 
        )
        
        # --- 美化图形 ---
        ax.set_title(f"{chrom}", fontsize=14, weight='bold')
        ax.set_xlabel(None)
        ax.set_ylabel("Average Contact Value", fontsize=12)
        
        # 移除图例
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        # 设置坐标轴范围
        ax.set_xlim(-1.1, 1.1)
        
        # Y轴范围
        y_max = chrom_df['Avg_Contact_Value'].max()
        ax.set_ylim(bottom=0, top=y_max * 1.1)
        
        # 【修改点】：设置特定的X轴刻度和自定义标签
        ax.set_xticks([-1.0, 0.0, 1.0])
        ax.set_xticklabels(['TEL', 'CEN', 'TEL'], fontsize=12, fontweight='bold')

        # --- 步骤 4: 保存图片 ---
        base_filename = os.path.join(output_dir, chrom)
        plt.savefig(f"{base_filename}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{base_filename}.pdf", bbox_inches='tight')

        plt.close(fig)

    print("\n所有绘图任务完成！")



def batch_create_heatmaps(input_dir: str, output_dir: str):
    """
    批量处理一个文件夹中的所有TSV文件，为每个文件生成一个热图。
    热图的值将进行 log10(x+1) 转换以压缩动态范围。

    每个TSV文件应包含三列: bin1, bin2, contact_value。
    函数会为每个输入文件生成一个PNG和一个PDF格式的热图，保存在输出文件夹中。

    Args:
        input_dir (str): 包含TSV文件的输入文件夹路径。
        output_dir (str): 用于保存生成的热图的输出文件夹路径。
    """
    # 1. 验证输入目录是否存在
    if not os.path.isdir(input_dir):
        print(f"ERROR: 输入文件夹 '{input_dir}' 不存在。")
        return

    # 2. 创建输出目录（如果不存在）
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"INFO: 输出将被保存到文件夹 '{output_dir}'")
    except OSError as e:
        print(f"ERROR: 无法创建输出文件夹 '{output_dir}'. 错误: {e}")
        return

    # 3. 查找所有 .tsv 或 .txt 文件
    search_pattern = os.path.join(input_dir, '*.tsv')
    tsv_files = glob.glob(search_pattern)
    
    if not tsv_files:
        search_pattern = os.path.join(input_dir, '*.txt')
        tsv_files = glob.glob(search_pattern)

    if not tsv_files:
        print(f"WARNING: 在 '{input_dir}' 中没有找到任何 .tsv 或 .txt 文件。")
        return
        
    print(f"INFO: 找到了 {len(tsv_files)} 个文件进行处理。")

    # 4. 循环处理每个文件
    for input_file_path in tsv_files:
        try:
            # 提取文件名（不含扩展名）
            base_filename = os.path.splitext(os.path.basename(input_file_path))[0]
            print(f"\n--- 正在处理: {base_filename}.tsv ---")

            # --- 核心绘图逻辑 ---
            
            # 使用 pandas 读取数据
            df = pd.read_csv(input_file_path, sep='\s+', header=None, names=['bin1', 'bin2', 'value'])
            
            if df.empty:
                print(f"WARNING: 文件 '{input_file_path}' 为空，已跳过。")
                continue

            # 使用 pivot 将数据重塑为矩阵
            heatmap_df = df.pivot(index='bin2', columns='bin1', values='value')
            # 排序以保证坐标轴连续
            heatmap_df = heatmap_df.sort_index(axis=0).sort_index(axis=1)
            # 填充缺失值为0
            heatmap_df = heatmap_df.fillna(0)

            # 【修改点】：对数据取对数
            # 使用 log10(x + 1) 是标准做法。
            # +1 是为了防止 log(0) 导致的负无穷大错误 (log10(1) = 0)。
            # 这能更好地展示 Hi-C 数据，因为 Hi-C 数据的动态范围通常很大。
            heatmap_log_values = np.log10(heatmap_df.values + 1)

            # 创建图形和坐标轴
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # 绘制热图 (使用取log后的数据)
            # 建议将 cmap 改为 'Reds' 或 'YlOrRd' 等渐变色，更能体现深浅变化
            ax.imshow(heatmap_log_values, cmap='Reds', origin='lower', aspect='auto')
            
            # 移除所有坐标轴相关的元素
            ax.axis('off')
            
            # 构建输出文件路径
            output_png = os.path.join(output_dir, f"{base_filename}.png")
            output_pdf = os.path.join(output_dir, f"{base_filename}.pdf")

            # 保存为PNG格式
            fig.savefig(output_png, bbox_inches='tight', pad_inches=0, dpi=300)
            print(f"SUCCESS: 已保存PNG图像至 '{output_png}'")

            # 保存为PDF格式
            fig.savefig(output_pdf, bbox_inches='tight', pad_inches=0)
            print(f"SUCCESS: 已保存PDF图像至 '{output_pdf}'")

        except Exception as e:
            print(f"ERROR: 处理文件 '{input_file_path}' 时发生错误: {e}")
        finally:
            # 关闭图形释放内存
            if 'fig' in locals():
                plt.close(fig)
    
    print("\nINFO: 所有文件处理完毕。")

def main(matrix, abs, cen):
    matrix_prefix = matrix.split('.')[0]
    # 提取出每条染色体内部的contact值
    filter_intra_chromosomal_contacts(matrix, abs, matrix_prefix)
    # 对每条染色体内部的contact值绘制热图
    batch_create_heatmaps(matrix_prefix, f"{matrix_prefix}_heatmap")
    # 计算某条染色体中，每个bins与其他所有bins的contact值的平均值
    calculate_marginal_sums(matrix_prefix)
    # 将实际位置映射回.eachbin_contact.tsv文件
    map_bin_locations(matrix_prefix, abs)
    # 将带有实际位置信息，bin信息，contact信息的文件合并
    merge_abs_files(matrix_prefix, f"{matrix_prefix}.abs_contact")
    # 坐标均一化
    #normalize_chromosome_coordinates(f"{matrix_prefix}.abs_contact", cen)
    bin_chromosome_coordinates(f"{matrix_prefix}.abs_contact", cen)
    # 绘制图片
    plot_chromosome_contacts(f"{matrix_prefix}.normalized.tsv", f"{matrix_prefix}_plot")



main(args.matrix, args.abs, args.cen)

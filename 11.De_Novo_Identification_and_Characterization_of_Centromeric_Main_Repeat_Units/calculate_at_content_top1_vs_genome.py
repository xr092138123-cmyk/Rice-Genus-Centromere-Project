#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from collections import defaultdict

# --- 全局绘图参数设置 ---
# 确保输出的PDF文本可被Illustrator编辑
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# 设置字体为Arial，字号为7
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 7
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="计算基因组和着丝粒Top1重复序列的AT含量，并进行比较和可视化。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-g', '--genome_dir',
        required=True,
        help="必需：参考基因组目录。\n文件命名格式: {material}.fasta"
    )
    parser.add_argument(
        '-t', '--top1_dir',
        required=True,
        help="必需：着丝粒区域Top1重复序列目录。\n文件命名格式: {material}.{chromosome}.{...}.Top1.fa"
    )
    parser.add_argument(
        '-o', '--output_dir',
        required=True,
        help="必需：结果输出目录。"
    )
    return parser.parse_args()


def fasta_reader(fasta_file):
    """一个简单的FASTA文件解析器，返回 (header, sequence)"""
    header, sequence = None, []
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    yield header, ''.join(sequence)
                header = line[1:].split()[0] # 取>后的第一个词作为ID
                sequence = []
            else:
                sequence.append(line)
        if header:
            yield header, ''.join(sequence)


def calculate_at_content(sequence):
    """计算单条序列的AT含量"""
    seq_upper = sequence.upper()
    a_count = seq_upper.count('A')
    t_count = seq_upper.count('T')
    g_count = seq_upper.count('G')
    c_count = seq_upper.count('C')
    total_bases = a_count + t_count + g_count + c_count
    if total_bases == 0:
        return 0.0
    return (a_count + t_count) / total_bases


def process_files(genome_dir, top1_dir, output_dir):
    """主处理函数：计算AT含量并生成结果文件"""
    print("开始处理文件...")
    
    results = []
    genome_cache = {}  # 缓存已读取的基因组序列，避免重复读取大文件

    top1_files = [f for f in os.listdir(top1_dir) if f.endswith(('.fa', '.fasta'))]
    if not top1_files:
        print(f"错误：在目录 '{top1_dir}' 中未找到FASTA文件。")
        return None

    for top1_filename in sorted(top1_files):
        # 解析文件名获取材料和染色体信息
        # AA_Ogla_hap1.Chr01.w100.Top1.fa
        match = re.match(r'(.+?)\.(Chr\d+)\.', top1_filename)
        if not match:
            print(f"警告：跳过文件名格式不正确的文件: {top1_filename}")
            continue
        
        material, chromosome = match.groups()
        print(f"正在处理: 材料={material}, 染色体={chromosome}")

        # --- 1. 计算着丝粒Top1重复序列的平均AT含量 ---
        top1_file_path = os.path.join(top1_dir, top1_filename)
        at_contents = []
        for _, seq in fasta_reader(top1_file_path):
            at_contents.append(calculate_at_content(seq))
        
        if not at_contents:
            print(f"警告：文件 '{top1_filename}' 中没有序列，跳过。")
            continue
        top1_at_mean = sum(at_contents) / len(at_contents)

        # --- 2. 计算对应基因组染色体的AT含量 ---
        genome_file_path = os.path.join(genome_dir, f"{material}.fasta")
        
        # 检查基因组文件是否存在
        if not os.path.exists(genome_file_path):
            print(f"警告：找不到材料 '{material}' 对应的基因组文件: {genome_file_path}，跳过。")
            continue

        # 从缓存或文件中读取基因组
        if material not in genome_cache:
            print(f"  读取并缓存基因组: {genome_file_path}")
            genome_cache[material] = {header: seq for header, seq in fasta_reader(genome_file_path)}

        # 查找染色体并计算AT含量
        if chromosome in genome_cache[material]:
            genome_chr_seq = genome_cache[material][chromosome]
            genome_at = calculate_at_content(genome_chr_seq)
        else:
            print(f"警告：在基因组 '{material}.fasta' 中找不到染色体 '{chromosome}'，跳过。")
            continue

        # --- 3. 保存结果 ---
        results.append({
            'Material': material,
            'Chromosome': chromosome,
            'Genome_AT_Content': genome_at,
            'Centromere_Top1_AT_Content': top1_at_mean
        })

    if not results:
        print("处理完成，但没有生成任何有效数据。")
        return None

    # 将结果转换为DataFrame并保存
    df = pd.DataFrame(results)
    output_tsv_path = os.path.join(output_dir, 'at_content_statistics.tsv')
    df.to_csv(output_tsv_path, sep='\t', index=False, float_format='%.4f')
    print(f"统计结果已保存至: {output_tsv_path}")
    
    return df


def format_p_value(p):
    """格式化p值并返回显著性星号"""
    if p < 0.001:
        return f"p < 0.001 (***)"
    elif p < 0.01:
        return f"p = {p:.3f} (**)"
    elif p < 0.05:
        return f"p = {p:.3f} (*)"
    else:
        return f"p = {p:.3f} (ns)"


def create_plot(df, output_dir):
    """创建箱线图并进行统计检验"""
    if df is None or df.empty:
        print("没有数据可供绘图。")
        return

    print("开始生成箱线图...")
    
    # 将宽数据转换为长数据，方便seaborn绘图
    df_long = pd.melt(df, id_vars=['Material', 'Chromosome'],
                      value_vars=['Genome_AT_Content', 'Centromere_Top1_AT_Content'],
                      var_name='Type', value_name='AT_Content')
    
    # 转换Type列的名称
    df_long['Type'] = df_long['Type'].replace({
        'Genome_AT_Content': 'Genome',
        'Centromere_Top1_AT_Content': 'Centromere Top1'
    })

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(df['Material'].unique())), 5))

    # 绘制箱线图和散点图
    sns.boxplot(
        data=df_long, x='Material', y='AT_Content', hue='Type', ax=ax,
        palette={'Genome': '#66c2a5', 'Centromere Top1': '#fc8d62'},
        width=0.6, fliersize=0  # 不显示异常值，因为我们用散点图表示
    )
    sns.stripplot(
        data=df_long, x='Material', y='AT_Content', hue='Type', ax=ax,
        dodge=True, jitter=True, size=3,
        palette={'Genome': '#333333', 'Centromere Top1': '#333333'},
        alpha=0.7
    )
    
    # --- 添加统计检验和显著性标注 ---
    materials = sorted(df['Material'].unique())
    for i, material in enumerate(materials):
        sub_df = df[df['Material'] == material]
        genome_vals = sub_df['Genome_AT_Content']
        top1_vals = sub_df['Centromere_Top1_AT_Content']

        # 至少需要几对数据才能进行检验
        if len(genome_vals) < 3:
            continue
            
        # 配对 Wilcoxon 检验
        stat, p_val = wilcoxon(genome_vals, top1_vals)

        # 准备在图上标注
        y_max = max(genome_vals.max(), top1_vals.max())
        
        # 使用 df_long 计算全局的y轴范围，以确定合适的偏移量
        y_offset = (df_long['AT_Content'].max() - df_long['AT_Content'].min()) * 0.05
        
        # 绘制标注线
        line_y = y_max + y_offset
        ax.plot([i - 0.15, i + 0.15], [line_y, line_y], color='black', lw=0.8)
        
        # 标注p值
        p_text = format_p_value(p_val)
        ax.text(i, line_y, p_text, ha='center', va='bottom', fontsize=6)

    # --- 图形美化 ---
    ax.set_title('AT Content Comparison: Genome vs. Centromere Top1 Repeats', fontsize=9)
    ax.set_xlabel('Material')
    ax.set_ylabel('AT Content')
    
    # 调整图例
    handles, labels = ax.get_legend_handles_labels()
    # 只保留箱线图的图例 (前两个)
    ax.legend(handles[:2], labels[:2], title='Region Type', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # 调整布局为图例留出空间

    # 保存图像
    output_pdf_path = os.path.join(output_dir, 'AT_content_comparison.pdf')
    fig.savefig(output_pdf_path, dpi=300, bbox_inches='tight')
    print(f"箱线图已保存至: {output_pdf_path}")
    plt.close()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 核心处理流程
    results_df = process_files(args.genome_dir, args.top1_dir, args.output_dir)
    
    # 绘图
    create_plot(results_df, args.output_dir)
    
    print("所有任务完成！")


if __name__ == "__main__":
    main()
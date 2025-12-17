import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import argparse
import re
import sys

def plot_top1_dominance(data_file, config_file, output_file, width=None, height=None):
    """
    根据Top1重复占比数据和配置文件，绘制一个水平的矩阵条形图。
    样式精确模仿用户提供的最终范例图。
    支持自定义图像尺寸。
    """
    # --- 1. 设置Matplotlib参数 ---
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8
    plt.rcParams['figure.autolayout'] = False

    # --- 2. 加载和预处理数据 ---
    try:
        config_df = pd.read_csv(config_file, sep='\t')
        config_df.columns = ['Species', 'Color', 'Species_name']
        config_df = config_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        data_df = pd.read_csv(data_file, sep='\t')
    except Exception as e:
        print(f"错误: 读取文件失败。请检查文件路径和格式 (应为制表符分隔)。- {e}", file=sys.stderr)
        sys.exit(1)

    # ... (数据处理部分与之前相同) ...
    def safe_split(material_string):
        if '_' in material_string: return material_string.rsplit('_', 1)
        else: return [material_string, 'N/A']
    split_data = data_df['Material'].apply(safe_split).tolist()
    data_df[['Species', 'Haplotype']] = pd.DataFrame(split_data, index=data_df.index)
    merged_df = pd.merge(data_df, config_df, on='Species', how='left')
    merged_df.dropna(subset=['Species_name'], inplace=True)
    if merged_df.empty:
        print("错误: 没有可用于绘图的数据。", file=sys.stderr)
        sys.exit(1)

    # --- 3. 确定绘图顺序 ---
    species_order = config_df['Species'].tolist()
    merged_df['Species'] = pd.Categorical(merged_df['Species'], categories=species_order, ordered=True)
    merged_df.sort_values(by=['Species', 'Haplotype'], inplace=True)
    material_order = merged_df['Material'].unique().tolist()
    
    y_labels_map = {}
    for _, row in merged_df.drop_duplicates(subset=['Material']).iterrows():
        hap_label = row['Haplotype'] if row['Haplotype'] != 'N/A' else ''
        species_name_formatted = ' '.join([f'$\\it{{{part}}}$' for part in row['Species_name'].split()])
        y_labels_map[row['Material']] = f"{species_name_formatted} {hap_label}".strip()
    
    merged_df['Chr_num'] = merged_df['Chromosome'].apply(lambda x: int(re.search(r'\d+', str(x)).group()))
    chromosome_order = sorted(merged_df['Chromosome'].unique(), key=lambda x: int(re.search(r'\d+', str(x)).group()))

    # --- 4. 创建绘图网格 ---
    n_rows = len(material_order)
    n_cols = len(chromosome_order)
    
    dynamic_width = max(8, n_cols * 0.8)
    dynamic_height = max(5, n_rows * 0.25)
    fig_width = width if width is not None else dynamic_width
    fig_height = height if height is not None else dynamic_height
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), 
                             sharex=True, sharey=True, gridspec_kw={'hspace': 0.1, 'wspace': 0.2})
    
    if n_rows == 1 and n_cols == 1: axes = [[axes]]
    elif n_rows == 1: axes = [axes]
    elif n_cols == 1: axes = [[ax] for ax in axes]

    color_low_ows = '#C44E52' # Red (<= 500)
    color_high_ows = '#4C72B0'  # Blue (> 500)

    # --- 5. 循环绘图 ---
    for i, material in enumerate(material_order):
        for j, chromosome in enumerate(chromosome_order):
            ax = axes[i][j]
            point_data = merged_df[(merged_df['Material'] == material) & (merged_df['Chromosome'] == chromosome)]
            
            if not point_data.empty:
                dominance = point_data['Dominance_at_OWS_Percent'].iloc[0]
                ows = point_data['OWS_Final'].iloc[0]
                bar_color = color_high_ows if ows > 500 else color_low_ows
                ax.barh(0, dominance, color=bar_color, height=0.75, zorder=3)

            # --- 6. 格式化每个子图 (ax) ---
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_facecolor('none')
            for spine in ax.spines.values(): spine.set_visible(False)
            
            # *** 修改点 3: 只在60%处画一条点状虚线 ***
            ax.axvline(x=60, color='grey', linestyle=':', linewidth=0.8, zorder=0)

            if j == 0: ax.set_ylabel(y_labels_map[material], rotation=0, ha='right', va='center', fontsize=8, labelpad=5)
            if i == 0: ax.set_title(chromosome, fontsize=9, pad=10)

            # --- 修改点 1 & 2: 调整X轴刻度尺样式 ---
            if i == n_rows - 1:
                # 绘制一条清晰的黑色基线 (从0到100)
                ax.spines['bottom'].set_visible(True)
                ax.spines['bottom'].set_color('black')
                ax.spines['bottom'].set_linewidth(0.8)
                
                # 设置刻度位置，包括100，以确保基线末端有标记
                ax.set_xticks([0, 40, 80, 100])
                # 只为0, 40, 80提供标签，100的标签为空字符串
                ax.set_xticklabels(['0', '40', '80', ''])
                ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize=8, length=2, pad=2)
            else:
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # --- 7. 添加全局元素 ---
    fig.suptitle('Dominance of Top-1 Repeat in Centromeres', fontsize=12, y=0.99)
    legend_elements = [
        Patch(facecolor=color_low_ows, label='Optimal Window Size <= 500'),
        Patch(facecolor=color_high_ows, label='Optimal Window Size > 500')]
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.01), fontsize=8, frameon=True, edgecolor='black')
    
    fig.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])

    # --- 8. 保存图像 ---
    try:
        plt.savefig(output_file, format='pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f"绘图成功！结果已保存至: {output_file}")
    except Exception as e:
        print(f"错误: 保存文件时发生错误 - {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="绘制Top1重复占比的水平矩阵条形图，样式模仿指定范例。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('data_file', type=str, help="各材料的Top1重复占比结果文件 (制表符分隔)。")
    parser.add_argument('config_file', type=str, help="作图的配置文件 (制表符分隔)。")
    parser.add_argument('output_file', type=str, help="输出的PDF文件名。")
    parser.add_argument('--width', type=float, default=None, help="可选: 设置输出图像的宽度 (英寸)。")
    parser.add_argument('--height', type=float, default=None, help="可选: 设置输出图像的高度 (英寸)。")
    
    args = parser.parse_args()
    
    plot_top1_dominance(args.data_file, args.config_file, args.output_file, args.width, args.height)

if __name__ == '__main__':
    main()
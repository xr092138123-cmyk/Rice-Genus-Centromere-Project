#!/usr/bin/env python3
import argparse
import sys
from Bio import SeqIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from matplotlib.patches import Patch


def get_kmers(seq, k=17):
    seq_str = str(seq).upper()
    if len(seq_str) < k:
        return set()
    return {seq_str[i:i + k] for i in range(len(seq_str) - k + 1)}


def jaccard_similarity(set1, set2):
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return len(set1 & set2) / union


def classify_species(seq_id):
    """第一层：按物种/亚种分类（新增 5SrDNA 作为独立功能位点类别）"""
    sid = seq_id.upper()
    if "AA_OSAT_JAP" in sid:
        return "AA_Osat_jap"
    if "AA_OSAT_IND" in sid:
        return "AA_Osat_ind"
    if "AA_OGLA" in sid:
        return "AA_Ogla"
    if "AA_ORUF" in sid:
        return "AA_Oruf"
    if "AA_ONIV" in sid:
        return "AA_Oniv"
    if "AA_OLON" in sid:
        return "AA_Olon"
    if "AA_OGLU" in sid:
        return "AA_Oglu"
    if "BB_OPUN" in sid:
        return "BB_Opun"
    if "CC_OOFF" in sid:
        return "CC_Ooff"
    if "EE_OAUS" in sid:
        return "EE_Oaus"
    if "FF_OBRA" in sid:
        return "FF_Obra"
    if "GG" in sid:
        return "GG"
    if "CRM" in sid:
        return "CRM_species"
    if "5SRDNA" in sid:
        return "5SrDNA_locus"  # 独立类别，用于物种层
    return "Other"


def classify_cen(seq_id):
    """第二层：按 CEN/CRM 类型分类"""
    sid = seq_id.upper()
    if "CEN155" in sid:
        return "CEN155"
    if "CEN165" in sid:
        return "CEN165"
    if "CEN154" in sid:
        return "CEN154"
    if "CEN126" in sid:
        return "CEN126"
    if "CRM" in sid:
        return "CRM_cen"
    if "5SRDNA" in sid:
        return "5SrDNA"
    if "NUMT" in sid:
        return "NUMT"
    if "NUPT" in sid:
        return "NUPT"
    return "Other"


def main():
    parser = argparse.ArgumentParser(description="Generate clustered heatmap with dual annotation: species and CEN type (including 5SrDNA in both layers).")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("-k", "--kmer", type=int, default=17, help="k-mer length (default: 17)")
    parser.add_argument("-o", "--output", default="dual_annotated_heatmap.pdf", help="Output PDF file")
    args = parser.parse_args()

    try:
        records = list(SeqIO.parse(args.fasta, "fasta"))
    except FileNotFoundError:
        sys.exit(f"Error: FASTA file '{args.fasta}' not found.")
    if len(records) < 2:
        sys.exit("Error: At least 2 sequences required.")

    print(f"Loaded {len(records)} sequences from '{args.fasta}'")

    kmer_sets = []
    labels = []
    species_cats = []
    cen_cats = []
    for rec in records:
        kmers = get_kmers(rec.seq, args.kmer)
        kmer_sets.append(kmers)
        labels.append(rec.id)
        species_cats.append(classify_species(rec.id))
        cen_cats.append(classify_cen(rec.id))

    n = len(labels)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = jaccard_similarity(kmer_sets[i], kmer_sets[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
    dist_matrix = 1.0 - sim_matrix
    condensed_dist = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method='average')

    # === 颜色映射 ===
    species_colors_map = {
        "AA_Osat_jap": "#59AC6E",
        "AA_Osat_ind": "#CBE54E",
        "AA_Ogla":     "#76D273",
        "AA_Oruf":     "#215A20",
        "AA_Oniv":     "#3BA738",
        "AA_Olon":     "#51C54E",
        "AA_Oglu":     "#3D8347",
        "BB_Opun":     "#F2AE2C",
        "CC_Ooff":     "#684E94",
        "EE_Oaus":     "#4E84C3",
        "FF_Obra":     "#D55F6F",
        "GG":     "#9D5427",
        "CRM_species": "#595959",
        "5SrDNA_locus": "#DF46CA",  # 金色：物种层的 5SrDNA
        "Other":       "#CCCCCC"
    }

    cen_colors_map = {
        "CEN155": "#be0f2d",
        "CEN165": "#4E84C3",
        "CEN154": "#F2AE2C",
        "CEN126": "#3D8347",
        "CRM_cen": "#D89BE7",
        "NUMT": "#64288B",
        "NUPT": "#4643CF",
        "5SrDNA": "#C6F35E",        # 亮绿色：CEN 层的 5SrDNA
        "Other":   "#EEEEEE"
    }

    # 生成颜色列表
    species_colors = [species_colors_map.get(cat, "#CCCCCC") for cat in species_cats]
    cen_colors = [cen_colors_map.get(cat, "#EEEEEE") for cat in cen_cats]

    # 双层颜色注释
    row_colors = [species_colors, cen_colors]
    col_colors = [species_colors, cen_colors]

    # 绘图
    cg = sns.clustermap(
        df,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap="RdPu",
        figsize=(14, 12),
        cbar_kws={'label': f'{args.kmer}-mer Jaccard Similarity'},
        annot=False,
        xticklabels=False,
        yticklabels=False
    )

    # === 添加双图例，统一处理标签后缀 ===
    unique_species = sorted(set(species_cats))
    handles1 = [
        Patch(
            facecolor=species_colors_map[cat],
            label=cat.replace("_species", "").replace("_locus", "")
        )
        for cat in unique_species if cat in species_colors_map
    ]

    unique_cen = sorted(set(cen_cats))
    handles2 = [
        Patch(
            facecolor=cen_colors_map[cat],
            label=cat.replace("_cen", "")
        )
        for cat in unique_cen if cat in cen_colors_map
    ]

    # 使用 figure 级图例，避免覆盖
    fig = cg.fig
    fig.legend(
        handles=handles1,
        title="Species / Haplotype / Locus",
        bbox_to_anchor=(0.98, 0.95),
        loc='upper left',
        frameon=True,
        fontsize='small'
    )
    fig.legend(
        handles=handles2,
        title="Centromere Type",
        bbox_to_anchor=(0.98, 0.75),
        loc='upper left',
        frameon=True,
        fontsize='small'
    )

    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Dual-annotated heatmap saved to: {args.output}")


if __name__ == "__main__":
    main()
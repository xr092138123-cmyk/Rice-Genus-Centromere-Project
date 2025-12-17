#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_centromere_dynamics_v18_final.py

This final, comprehensive script identifies the Optimal Window Size (OWS) using
the log-normalized contribution score. It integrates a multi-evidence system
to automatically classify centromere types, generates detailed output files
(with NA for missing values), and produces publication-quality PDF plots.
"""

import os
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Publication Quality Settings ---
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
FONT_SIZE = 7
TITLE_FONT_SIZE = 8

# === Helper functions for classification and data retrieval ===

def get_value_at_window(df, window_size, col_name):
    """Generic function to get a value from a specific column at a given window size."""
    if pd.isna(window_size):
        return np.nan
    value = df[df['WindowSize'] == window_size][col_name]
    return value.iloc[0] if not value.empty else np.nan

def classify_scale(ows):
    if pd.isna(ows): return "Unknown"
    if ows <= 300: return "Short"
    if ows < 1500: return "Mid"
    return "Long"

def classify_clarity(score):
    if pd.isna(score): return "Unknown"
    if score > 5.0: return "Strong"
    if score > 1.0: return "Moderate"
    return "Weak"

def classify_density(ws_for_coverage):
    if pd.isna(ws_for_coverage): return "Sparse"
    if ws_for_coverage <= 500: return "Dense"
    if ws_for_coverage < 2000: return "Intermediate"
    return "Sparse"

def classify_dominance_ratio(ratio):
    if pd.isna(ratio): return "Unknown"
    if ratio < 1.2: return "Short-Dominant"
    if ratio < 2.0: return "Balanced"
    return "Long-Dominant"

def predict_type(scale, clarity, density, dom_ratio_class):
    """Predicts centromere type based on a majority vote system."""
    dense_votes = 0
    dispersed_votes = 0

    if scale == "Short": dense_votes += 1
    elif scale == "Long": dispersed_votes += 1
    
    if clarity == "Strong": dense_votes += 1
    elif clarity == "Weak": dispersed_votes += 1
    
    if density == "Dense": dense_votes += 1
    elif density == "Sparse": dispersed_votes += 1

    if dom_ratio_class == "Short-Dominant": dense_votes += 1
    elif dom_ratio_class == "Long-Dominant": dispersed_votes += 1

    if dense_votes >= 3: return "Dense (High Confidence)"
    if dispersed_votes >= 3: return "Dispersed (High Confidence)"
    if dense_votes == 2 and dispersed_votes <= 1: return "Likely Dense"
    if dispersed_votes == 2 and dense_votes <= 1: return "Likely Dispersed"
    return "Hybrid/Ambiguous"


def analyze_material_with_classification(input_file, material_name, output_dir):
    """
    Final analysis function with integrated classification system and publication-quality plotting.
    """
    # --- 1. Load data ---
    print(f"--- Step 1: Loading data for material '{material_name}' ---")
    df = pd.read_csv(input_file, sep='\t')
    df_material = df[df['Material'] == material_name].copy()
    if df_material.empty:
        print(f"ERROR: No data found for material '{material_name}'.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Prepare output ---
    print(f"--- Step 2: Preparing output directory '{output_dir}' ---")
    os.makedirs(output_dir, exist_ok=True)
    all_summary_metrics = []
    all_contribution_tables = []

    # --- 3. Analyze each chromosome ---
    print(f"--- Step 3: Analyzing each chromosome with classification ---")
    chromosomes = sorted(df_material['Chromosome'].unique())

    for chrom in chromosomes:
        print(f"  -> Processing {chrom}...")
        df_chrom_all_tops = df_material[df_material['Chromosome'] == chrom]
        df_top1 = df_chrom_all_tops[df_chrom_all_tops['Top'] == 'Top1'].sort_values('WindowSize').reset_index(drop=True)
        df_top5 = df_chrom_all_tops[df_chrom_all_tops['Top'] == 'Top5'].sort_values('WindowSize')

        if len(df_top1) < 2:
            print(f"  WARNING: Fewer than 2 data points for {chrom}. Skipping.", file=sys.stderr)
            continue
            
        # --- a) Calculate log-contribution score to find OWS ---
        contribution_results = []
        for i in range(len(df_top1) - 1):
            w1_row = df_top1.iloc[i]
            w2_row = df_top1.iloc[i+1]
            window_diff = w2_row['WindowSize'] - w1_row['WindowSize']
            perc_diff = w2_row['Percentage_of_Centromere'] - w1_row['Percentage_of_Centromere']
            if window_diff > 1:
                contribution_score = perc_diff / np.log10(window_diff)
            else:
                contribution_score = 0
            
            contribution_results.append({
                'Chromosome': chrom, 'w1(bp)': w1_row['WindowSize'], 'w2(bp)': w2_row['WindowSize'],
                'Window_Diff(bp)': window_diff, 'Perc_w1(%)': w1_row['Percentage_of_Centromere'],
                'Perc_w2(%)': w2_row['Percentage_of_Centromere'], 'Perc_Diff(%)': perc_diff,
                'Contribution_Score': contribution_score
            })
        
        contribution_df = pd.DataFrame(contribution_results)
        all_contribution_tables.append(contribution_df)
        
        if contribution_df.empty: continue
        
        if (contribution_df['Contribution_Score'] > 0).any():
            ows_row_from_contribution = contribution_df.loc[contribution_df['Contribution_Score'].idxmax()]
        else:
            ows_row_from_contribution = contribution_df.loc[contribution_df['Contribution_Score'].idxmax()]

        ows_final = int(ows_row_from_contribution['w2(bp)'])
        peak_score = ows_row_from_contribution['Contribution_Score']
        dominance_at_ows = get_value_at_window(df_top1, ows_final, 'Percentage_of_Centromere')
        seed_region_at_ows = get_value_at_window(df_top1, ows_final, 'Seed_Region')

        # --- b) Calculate all metrics for classification ---
        saturated_rows = df_top5[df_top5['Cumulative_Percentage'] >= 80]
        ws_top5_80 = saturated_rows['WindowSize'].min() if not saturated_rows.empty else np.nan
        perc_at_ws_top5_80 = get_value_at_window(df_top5, ws_top5_80, 'Cumulative_Percentage')
        
        dom_200 = get_value_at_window(df_top1, 200, 'Percentage_of_Centromere')
        dom_2000 = get_value_at_window(df_top1, 2000, 'Percentage_of_Centromere')
        
        dominance_ratio_value = np.nan
        if pd.notna(dom_200) and pd.notna(dom_2000) and dom_200 > 0:
            dominance_ratio_value = dom_2000 / dom_200

        # --- c) Generate evidence and prediction ---
        evidence_scale = classify_scale(ows_final)
        evidence_clarity = classify_clarity(peak_score)
        evidence_density = classify_density(ws_top5_80)
        evidence_dom_ratio_class = classify_dominance_ratio(dominance_ratio_value)
        predicted_type = predict_type(evidence_scale, evidence_clarity, evidence_density, evidence_dom_ratio_class)

        # --- d) Store all metrics ---
        all_summary_metrics.append({
            'Material': material_name,
            'Chromosome': chrom,
            'Predicted_Type': predicted_type,
            'OWS_Final': ows_final,
            'OWS_Top1_Region': seed_region_at_ows,
            'Dominance_at_OWS_Percent': dominance_at_ows,
            'Peak_Contribution_Score': peak_score,
            'WS_Top5_80pct_WS': ws_top5_80,
            'Top5_Coverage_at_WS_80pct': perc_at_ws_top5_80,
            'Top1_Dom_at_200bp': dom_200,
            'Top1_Dom_at_2000bp': dom_2000,
            'Evidence_Scale': evidence_scale,
            'Evidence_Clarity': evidence_clarity,
            'Evidence_Density': evidence_density,
            'Evidence_Dominance_Ratio': evidence_dom_ratio_class,
            'Dominance_Ratio_Value': dominance_ratio_value,
        })

        # --- e) Generate plot ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        fig.suptitle(f'Centromere Signal Analysis: {material_name} - {chrom}', fontsize=TITLE_FONT_SIZE)
        ax1.plot(df_top1['WindowSize'], df_top1['Percentage_of_Centromere'], 'o-', color='tab:blue', markersize=3, linewidth=1, label='Top1 Dominance (%)')
        ax1.scatter(ows_final, dominance_at_ows, s=60, facecolors='none', edgecolors='purple', lw=1.5, label=f'Final OWS: {ows_final} bp')
        ax1.set_ylabel('Top1 Dominance (%)', fontsize=FONT_SIZE)
        ax1.grid(True, which="both", ls="--", linewidth=0.5)
        ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        ax1.legend(fontsize=FONT_SIZE)
        ax2.plot(contribution_df['w2(bp)'], contribution_df['Contribution_Score'], 'o-', color='tab:green', markersize=3, linewidth=1, label='Contribution Score')
        ax2.scatter(ows_final, peak_score, s=40, facecolors='none', edgecolors='purple', lw=1.5)
        ax2.axvline(x=ows_final, color='purple', linestyle='-.', lw=1, label=f'Peak Score at {ows_final} bp')
        ax2.set_ylabel('Contribution Score', fontsize=FONT_SIZE)
        ax2.set_xlabel('Window Size (bp) [Log Scale]', fontsize=FONT_SIZE)
        ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        for idx, row in contribution_df.iterrows():
            vertical_alignment = 'bottom' if idx % 2 == 0 else 'top'
            y_pos = row['Contribution_Score']
            offset = (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.03
            if vertical_alignment == 'top': offset = -offset
            ax2.text(row['w2(bp)'], y_pos + offset, f"w{int(row['w2(bp)'])}", fontsize=5, ha='center', va=vertical_alignment, color='grey')
        ax2.legend(fontsize=FONT_SIZE)
        plt.xscale('log')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f"{material_name}_{chrom}_dynamics_classified.pdf"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, format='pdf', dpi=300)
        plt.close(fig)
        print(f"     - Classified plot saved to {plot_path}")

    # --- 5. Save the final output files ---
    print(f"\n--- Step 5: Saving the final output files ---")

    # Output 1: Main classified summary file
    summary_df = pd.DataFrame(all_summary_metrics)
    final_cols = [
        'Material', 'Chromosome', 'Predicted_Type', 'OWS_Final', 'OWS_Top1_Region',
        'Dominance_at_OWS_Percent', 'Peak_Contribution_Score', 'WS_Top5_80pct_WS',
        'Top5_Coverage_at_WS_80pct', 'Top1_Dom_at_200bp', 'Top1_Dom_at_2000bp',
        'Evidence_Scale', 'Evidence_Clarity', 'Evidence_Density',
        'Evidence_Dominance_Ratio', 'Dominance_Ratio_Value'
    ]
    summary_df = summary_df.reindex(columns=final_cols)
    summary_filename = f"{material_name}_summary_metrics_classified.tsv"
    summary_path = os.path.join(output_dir, summary_filename)
    # ==================== CHANGE: Added na_rep='NA' ====================
    summary_df.to_csv(summary_path, sep='\t', index=False, float_format='%.4f', na_rep='NA')
    print(f"Final classified summary saved to: {summary_path}")
    print("\nClassified Summary Table Preview:")
    print(summary_df.to_string())

    # Output 2: Detailed contribution analysis file
    if all_contribution_tables:
        full_contribution_df = pd.concat(all_contribution_tables, ignore_index=True)
        contribution_filename = f"{material_name}_contribution_details.tsv"
        contribution_path = os.path.join(output_dir, contribution_filename)
        # ==================== CHANGE: Added na_rep='NA' ====================
        full_contribution_df.to_csv(contribution_path, sep='\t', index=False, float_format='%.4f', na_rep='NA')
        print(f"\nContribution details saved to: {contribution_path}")

    print(f"\n--- Analysis for '{material_name}' complete! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final centromere analysis with classification and publication-quality plotting.")
    parser.add_argument("input_file", help="Path to the aggregated TSV file.")
    parser.add_argument("material_name", help="Name of the material to analyze.")
    parser.add_argument("output_dir", help="Directory for output plots and summary tables.")
    args = parser.parse_args()
    analyze_material_with_classification(args.input_file, args.material_name, args.output_dir)
#!/usr/bin/env python3
"""
Evaluate predictions using top 5 bin_start and bin_end with highest probabilities.
Selects the farthest start and end windows (within 1200 window limit) and compares with ground truth.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob

def calculate_iou(pred_start, pred_end, gt_start, gt_end):
    """Calculate Intersection over Union (IoU) between predicted and ground truth regions."""
    # Calculate intersection
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    if intersection_start >= intersection_end:
        intersection = 0
    else:
        intersection = intersection_end - intersection_start
    
    # Calculate union
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start
    
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_precision_recall(pred_start, pred_end, gt_regions):
    """
    Calculate precision and recall.
    gt_regions: list of (start, end) tuples for ground truth regions
    """
    if len(gt_regions) == 0:
        return 0.0, 0.0, 0, 0, 0
    
    # Calculate total ground truth length
    gt_total = 0
    for gt_start, gt_end in gt_regions:
        gt_total += (gt_end - gt_start)
    
    # Calculate predicted length
    pred_length = pred_end - pred_start
    
    # Calculate intersection (TP)
    tp = 0
    for gt_start, gt_end in gt_regions:
        intersection_start = max(pred_start, gt_start)
        intersection_end = min(pred_end, gt_end)
        if intersection_start < intersection_end:
            tp += (intersection_end - intersection_start)
    
    # Calculate false positives (FP) - predicted but not in ground truth
    fp = pred_length - tp
    
    # Calculate false negatives (FN) - ground truth but not in predicted
    fn = gt_total - tp
    
    # Precision = TP / (TP + FP)
    if (tp + fp) == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    # Recall = TP / (TP + FN)
    if (tp + fn) == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    return precision, recall, int(tp), int(fp), int(fn)

def find_prediction_region(df, max_window_distance=1200):
    """
    Find prediction region using top 5 bin_start and bin_end with highest probabilities.
    Returns the farthest start and end windows within the window distance limit.
    """
    # Get top 5 unique bin_start with highest start_prob (group by bin_start, take max, then top 5)
    start_max = df.groupby('bin_start')['start_prob'].max().reset_index()
    top5_starts = start_max.nlargest(5, 'start_prob')
    top5_start_values = top5_starts['bin_start'].values
    
    # Get top 5 unique bin_end with highest end_prob (group by bin_end, take max, then top 5)
    end_max = df.groupby('bin_end')['end_prob'].max().reset_index()
    top5_ends = end_max.nlargest(5, 'end_prob')
    top5_end_values = top5_ends['bin_end'].values
    
    # Find the minimum start and maximum end
    min_start = top5_start_values.min()
    max_end = top5_end_values.max()
    
    # Check window distance constraint
    # We need to find the index positions (using the 'index' column) to calculate window distance
    start_rows = df[df['bin_start'] == min_start]
    end_rows = df[df['bin_end'] == max_end]
    
    if len(start_rows) == 0 or len(end_rows) == 0:
        # Fallback: use the values directly if we can't find them
        return min_start, max_end
    
    start_idx = start_rows['index'].iloc[0]
    end_idx = end_rows['index'].iloc[0]
    window_distance = abs(end_idx - start_idx)
    
    # If window distance exceeds limit, adjust
    if window_distance > max_window_distance:
        # Try to find a valid combination within the limit
        best_start = None
        best_end = None
        best_distance = 0
        
        for start_val in top5_start_values:
            for end_val in top5_end_values:
                if end_val > start_val:  # End should be after start
                    start_rows = df[df['bin_start'] == start_val]
                    end_rows = df[df['bin_end'] == end_val]
                    
                    if len(start_rows) > 0 and len(end_rows) > 0:
                        start_idx = start_rows['index'].iloc[0]
                        end_idx = end_rows['index'].iloc[0]
                        window_dist = abs(end_idx - start_idx)
                        
                        if window_dist <= max_window_distance:
                            region_distance = end_val - start_val
                            if region_distance > best_distance:
                                best_distance = region_distance
                                best_start = start_val
                                best_end = end_val
        
        if best_start is not None and best_end is not None:
            return best_start, best_end
        else:
            # If no valid combination found, use the original but warn
            print(f"Warning: Window distance {window_distance} exceeds limit {max_window_distance}, using original values")
            return min_start, max_end
    
    return min_start, max_end

def get_ground_truth_regions(df):
    """Extract all ground truth regions from the dataframe."""
    gt_df = df[df['is_ground_truth'] == 1]
    if len(gt_df) == 0:
        return []
    
    # Get unique bin_start and bin_end pairs for ground truth
    gt_regions = gt_df[['bin_start', 'bin_end']].drop_duplicates().values
    return [(int(start), int(end)) for start, end in gt_regions]

def process_chromosome(prob_file_path, output_dir=None):
    """Process a single chromosome's probability file."""
    print(f"Processing: {prob_file_path}")
    
    # Read the probabilities file
    df = pd.read_csv(prob_file_path)
    
    # Find prediction region
    pred_start, pred_end = find_prediction_region(df, max_window_distance=1200)
    
    # Get ground truth regions
    gt_regions = get_ground_truth_regions(df)
    
    # Calculate metrics
    if len(gt_regions) > 0:
        # Merge ground truth regions for IoU calculation (use min start and max end)
        gt_merged_start = min([r[0] for r in gt_regions])
        gt_merged_end = max([r[1] for r in gt_regions])
        
        iou = calculate_iou(pred_start, pred_end, gt_merged_start, gt_merged_end)
        precision, recall, tp, fp, fn = calculate_precision_recall(pred_start, pred_end, gt_regions)
    else:
        iou = 0.0
        precision = 0.0
        recall = 0.0
        tp = 0
        fp = pred_end - pred_start
        fn = 0
    
    # Create visualization
    if output_dir:
        chrom_name = Path(prob_file_path).parent.name
        detail_path = os.path.join(output_dir, f"{chrom_name}_detail.png")
        create_detail_plot(df, pred_start, pred_end, gt_regions, precision, recall, iou, detail_path, chrom_name)
    
    return {
        'pred_start': pred_start,
        'pred_end': pred_end,
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'gt_regions': gt_regions
    }

def create_detail_plot(df, pred_start, pred_end, gt_regions, precision, recall, iou, output_path, chrom_name):
    """Create a detailed visualization comparing prediction with ground truth."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Get chromosome range
    all_positions = np.concatenate([df['bin_start'].values, df['bin_end'].values])
    min_pos = all_positions.min()
    max_pos = all_positions.max()
    
    # Plot 1: Probabilities
    ax1 = axes[0]
    ax1.plot(df['bin_start'], df['start_prob'], 'b-', alpha=0.6, label='Start Probability', linewidth=1)
    ax1.plot(df['bin_end'], df['end_prob'], 'r-', alpha=0.6, label='End Probability', linewidth=1)
    ax1.plot(df['bin_start'], df['region_prob'], 'g-', alpha=0.4, label='Region Probability', linewidth=0.5)
    ax1.axvline(pred_start, color='blue', linestyle='--', linewidth=2, label=f'Predicted Start: {pred_start:,}')
    ax1.axvline(pred_end, color='red', linestyle='--', linewidth=2, label=f'Predicted End: {pred_end:,}')
    for gt_start, gt_end in gt_regions:
        ax1.axvspan(gt_start, gt_end, alpha=0.2, color='yellow', label='Ground Truth' if gt_regions.index((gt_start, gt_end)) == 0 else '')
    ax1.set_xlabel('Genomic Position')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'{chrom_name} - Probability Distribution')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Region comparison
    ax2 = axes[1]
    y_pred = 1
    y_gt = 0
    
    # Draw predicted region
    ax2.barh(y_pred, pred_end - pred_start, left=pred_start, height=0.3, 
             color='blue', alpha=0.7, label=f'Predicted: {pred_start:,} - {pred_end:,}')
    
    # Draw ground truth regions
    for i, (gt_start, gt_end) in enumerate(gt_regions):
        ax2.barh(y_gt, gt_end - gt_start, left=gt_start, height=0.3,
                color='green', alpha=0.7, 
                label='Ground Truth' if i == 0 else '')
    
    ax2.set_xlim(min_pos, max_pos)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_yticks([y_gt, y_pred])
    ax2.set_yticklabels(['Ground Truth', 'Predicted'])
    ax2.set_xlabel('Genomic Position')
    ax2.set_title(f'{chrom_name} - Region Comparison')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Zoomed view around prediction
    ax3 = axes[2]
    zoom_margin = (pred_end - pred_start) * 0.5
    zoom_start = max(min_pos, pred_start - zoom_margin)
    zoom_end = min(max_pos, pred_end + zoom_margin)
    
    ax3.plot(df['bin_start'], df['start_prob'], 'b-', alpha=0.6, label='Start Probability', linewidth=1.5)
    ax3.plot(df['bin_end'], df['end_prob'], 'r-', alpha=0.6, label='End Probability', linewidth=1.5)
    ax3.axvline(pred_start, color='blue', linestyle='--', linewidth=2, label=f'Predicted Start: {pred_start:,}')
    ax3.axvline(pred_end, color='red', linestyle='--', linewidth=2, label=f'Predicted End: {pred_end:,}')
    for gt_start, gt_end in gt_regions:
        if gt_end >= zoom_start and gt_start <= zoom_end:
            ax3.axvspan(gt_start, gt_end, alpha=0.3, color='yellow', label='Ground Truth' if gt_regions.index((gt_start, gt_end)) == 0 else '')
    ax3.set_xlim(zoom_start, zoom_end)
    ax3.set_xlabel('Genomic Position')
    ax3.set_ylabel('Probability')
    ax3.set_title(f'{chrom_name} - Zoomed View (Prediction Region)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Add metrics text
    metrics_text = f'Precision: {precision:.4f} | Recall: {recall:.4f} | IoU: {iou:.4f}'
    fig.suptitle(f'{chrom_name} - Evaluation Results\n{metrics_text}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved detail plot: {output_path}")

def process_all_chromosomes(base_dir):
    """Process all chromosomes in the base directory."""
    results = []
    
    # Find all probability files
    prob_files = glob.glob(os.path.join(base_dir, '**', '*_probabilities.csv'), recursive=True)
    
    if not prob_files:
        print(f"No probability files found in {base_dir}")
        return
    
    # Group by sample
    samples = {}
    for prob_file in prob_files:
        parts = Path(prob_file).parts
        # Find sample name (directory before chromosomes)
        for i, part in enumerate(parts):
            if part == 'chromosomes':
                sample_name = parts[i-1]
                if sample_name not in samples:
                    samples[sample_name] = []
                samples[sample_name].append(prob_file)
                break
    
    # Process each sample
    for sample_name, prob_files in samples.items():
        print(f"\n{'='*60}")
        print(f"Processing sample: {sample_name}")
        print(f"{'='*60}")
        
        sample_results = []
        
        for prob_file in sorted(prob_files):
            chrom_dir = Path(prob_file).parent
            result = process_chromosome(prob_file, output_dir=str(chrom_dir))
            
            chrom_name = chrom_dir.name
            result['sample'] = sample_name
            result['chromosome'] = chrom_name
            sample_results.append(result)
            results.append(result)
            
            print(f"{chrom_name}: Pred=[{result['pred_start']:,}, {result['pred_end']:,}], "
                  f"Precision={result['precision']:.4f}, Recall={result['recall']:.4f}, IoU={result['iou']:.4f}")
        
        # Calculate overall metrics for sample
        total_tp = sum(r['tp'] for r in sample_results)
        total_fp = sum(r['fp'] for r in sample_results)
        total_fn = sum(r['fn'] for r in sample_results)
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
        
        print(f"\nSample {sample_name} Overall:")
        print(f"  Precision: {overall_precision:.4f}")
        print(f"  Recall: {overall_recall:.4f}")
        print(f"  IoU: {overall_iou:.4f}")
        print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        
        # Add overall row
        results.append({
            'sample': sample_name,
            'chromosome': 'OVERALL',
            'pred_start': None,
            'pred_end': None,
            'precision': overall_precision,
            'recall': overall_recall,
            'iou': overall_iou,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'gt_regions': []
        })
    
    # Save summary CSV
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(base_dir, 'top5_evaluation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")
    
    return results

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = '/home/centromere_area_prediction_v1/inference/output_past_public_data'
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        sys.exit(1)
    
    process_all_chromosomes(base_dir)


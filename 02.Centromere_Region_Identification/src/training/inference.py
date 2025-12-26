#!/usr/bin/env python3
"""
推理脚本 - 着丝粒预测
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional

from config import get_config, Config
from model import CentromereTransformer, create_model
from dataset import ChromosomeDataset


def load_model(checkpoint_path: str, device: str = 'cpu') -> Tuple[CentromereTransformer, Dict, Config]:
    """
    加载训练好的模型
    
    Returns:
        (model, feature_stats, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 重建配置
    config = get_config()
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        for key, value in cfg.get('model', {}).items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
        for key, value in cfg.get('inference', {}).items():
            if hasattr(config.inference, key):
                setattr(config.inference, key, value)
    
    # 创建模型
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 获取特征统计
    feature_stats = checkpoint.get('feature_stats', {'mean': np.zeros(8), 'std': np.ones(8)})
    if isinstance(feature_stats['mean'], list):
        feature_stats['mean'] = np.array(feature_stats['mean'])
        feature_stats['std'] = np.array(feature_stats['std'])
    
    print(f"模型加载成功: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"  验证F1: {metrics.get('f1', 'N/A'):.4f}")
    
    return model, feature_stats, config


def load_csv_for_inference(csv_path: str, feature_stats: Dict) -> Tuple[torch.Tensor, np.ndarray, List[Dict]]:
    """
    加载CSV文件用于推理
    
    Returns:
        (features_tensor, positions, rows_info)
    """
    FEATURE_COLUMNS = [
        '64_highlighted_percent', '64_coverage_depth_avg',
        '128_highlighted_percent', '128_coverage_depth_avg',
        '256_highlighted_percent', '256_coverage_depth_avg',
        '512_highlighted_percent', '512_coverage_depth_avg'
    ]
    
    features_list = []
    positions_list = []
    rows_info = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = int(row['start'])
            end = int(row['end'])
            positions_list.append([start, end])
            
            feat = []
            for col in FEATURE_COLUMNS:
                feat.append(float(row[col]))
            features_list.append(feat)
            
            rows_info.append({
                'start': start,
                'end': end,
                'has_cen': int(row.get('has_cen', -1))
            })
    
    features = np.array(features_list, dtype=np.float32)
    positions = np.array(positions_list, dtype=np.int64)
    
    # 归一化
    mean = feature_stats['mean']
    std = feature_stats['std']
    features = (features - mean) / std
    
    features_tensor = torch.from_numpy(features).unsqueeze(0)  # (1, seq_len, 8)
    
    return features_tensor, positions, rows_info


def find_centromere_regions(
    probs: np.ndarray,
    positions: np.ndarray,
    threshold: float = 0.5,
    min_bins: int = 3
) -> List[Dict]:
    """
    从概率预测中找出着丝粒区域
    
    Args:
        probs: (seq_len,) 每个位置的预测概率
        positions: (seq_len, 2) 每个bin的起止位置
        threshold: 分类阈值
        min_bins: 最小区域长度（bin数）
    
    Returns:
        预测区域列表
    """
    binary = (probs > threshold).astype(int)
    regions = []
    
    # 找连续的1区间
    in_region = False
    region_start_idx = 0
    
    for i in range(len(binary)):
        if binary[i] == 1 and not in_region:
            in_region = True
            region_start_idx = i
        elif binary[i] == 0 and in_region:
            in_region = False
            region_end_idx = i - 1
            
            if region_end_idx - region_start_idx + 1 >= min_bins:
                region = {
                    'start_bin': region_start_idx,
                    'end_bin': region_end_idx,
                    'start_pos': int(positions[region_start_idx, 0]),
                    'end_pos': int(positions[region_end_idx, 1]),
                    'length_bins': region_end_idx - region_start_idx + 1,
                    'length_bp': int(positions[region_end_idx, 1] - positions[region_start_idx, 0]),
                    'avg_prob': float(probs[region_start_idx:region_end_idx+1].mean()),
                    'max_prob': float(probs[region_start_idx:region_end_idx+1].max())
                }
                regions.append(region)
    
    # 处理末尾的区域
    if in_region:
        region_end_idx = len(binary) - 1
        if region_end_idx - region_start_idx + 1 >= min_bins:
            region = {
                'start_bin': region_start_idx,
                'end_bin': region_end_idx,
                'start_pos': int(positions[region_start_idx, 0]),
                'end_pos': int(positions[region_end_idx, 1]),
                'length_bins': region_end_idx - region_start_idx + 1,
                'length_bp': int(positions[region_end_idx, 1] - positions[region_start_idx, 0]),
                'avg_prob': float(probs[region_start_idx:region_end_idx+1].mean()),
                'max_prob': float(probs[region_start_idx:region_end_idx+1].max())
            }
            regions.append(region)
    
    # 按平均概率排序
    regions.sort(key=lambda x: x['avg_prob'], reverse=True)
    
    return regions


def predict_single_chromosome(
    model: CentromereTransformer,
    csv_path: str,
    feature_stats: Dict,
    config: Config,
    device: str = 'cpu'
) -> Dict:
    """
    预测单条染色体
    
    Returns:
        预测结果字典
    """
    # 加载数据
    features, positions, rows_info = load_csv_for_inference(csv_path, feature_stats)
    features = features.to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        position_probs, range_scores = model(features)
    
    probs = position_probs.squeeze().cpu().numpy()  # (seq_len,)
    
    # 找出预测区域
    regions = find_centromere_regions(
        probs,
        positions,
        threshold=config.inference.threshold,
        min_bins=config.inference.min_region_bins
    )
    
    # 计算真实标签（如果有）
    true_labels = np.array([r['has_cen'] for r in rows_info])
    has_ground_truth = (true_labels >= 0).all()
    
    result = {
        'csv_file': csv_path,
        'seq_len': len(probs),
        'predictions': probs.tolist(),
        'positions': positions.tolist(),
        'predicted_regions': regions,
        'num_predicted_regions': len(regions)
    }
    
    if has_ground_truth:
        # 找出真实区域
        true_regions = find_centromere_regions(
            true_labels.astype(float),
            positions,
            threshold=0.5,
            min_bins=1
        )
        result['true_regions'] = true_regions
        result['num_true_regions'] = len(true_regions)
        
        # 计算指标
        binary_preds = (probs > config.inference.threshold).astype(int)
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, binary_preds, average='binary', zero_division=0
        )
        try:
            auc = roc_auc_score(true_labels, probs)
        except:
            auc = 0.0
        
        result['metrics'] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc)
        }
    
    return result


def predict_batch(
    model: CentromereTransformer,
    csv_files: List[str],
    feature_stats: Dict,
    config: Config,
    output_dir: str,
    device: str = 'cpu'
):
    """批量预测"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for i, csv_path in enumerate(csv_files):
        print(f"预测 {i+1}/{len(csv_files)}: {os.path.basename(csv_path)}")
        
        try:
            result = predict_single_chromosome(model, csv_path, feature_stats, config, device)
            all_results.append(result)
            
            # 打印结果摘要
            print(f"  序列长度: {result['seq_len']} bins")
            print(f"  预测区域数: {result['num_predicted_regions']}")
            
            if 'metrics' in result:
                m = result['metrics']
                print(f"  F1: {m['f1']:.4f}, Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")
            
            if result['predicted_regions']:
                top_region = result['predicted_regions'][0]
                print(f"  最可能区域: {top_region['start_pos']:,} - {top_region['end_pos']:,} "
                      f"(概率: {top_region['avg_prob']:.4f})")
            
        except Exception as e:
            print(f"  错误: {e}")
            continue
    
    # 保存所有结果
    results_path = os.path.join(output_dir, 'predictions.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n预测结果已保存: {results_path}")
    
    # 生成汇总CSV
    summary_path = os.path.join(output_dir, 'predictions_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'seq_len', 'num_regions', 'top_region_start', 'top_region_end', 
                        'top_region_prob', 'f1', 'precision', 'recall', 'auc'])
        
        for result in all_results:
            row = [
                os.path.basename(result['csv_file']),
                result['seq_len'],
                result['num_predicted_regions']
            ]
            
            if result['predicted_regions']:
                top = result['predicted_regions'][0]
                row.extend([top['start_pos'], top['end_pos'], f"{top['avg_prob']:.4f}"])
            else:
                row.extend(['', '', ''])
            
            if 'metrics' in result:
                m = result['metrics']
                row.extend([f"{m['f1']:.4f}", f"{m['precision']:.4f}", 
                           f"{m['recall']:.4f}", f"{m['auc']:.4f}"])
            else:
                row.extend(['', '', '', ''])
            
            writer.writerow(row)
    
    print(f"汇总CSV已保存: {summary_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='着丝粒预测推理')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件或目录')
    parser.add_argument('--output', type=str, default='./predictions', help='输出目录')
    parser.add_argument('--threshold', type=float, default=None, help='分类阈值')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model, feature_stats, config = load_model(args.checkpoint, device)
    
    # 覆盖配置
    if args.threshold:
        config.inference.threshold = args.threshold
    
    # 查找输入文件
    if os.path.isfile(args.input):
        csv_files = [args.input]
    else:
        csv_files = []
        for root, dirs, files in os.walk(args.input):
            for f in files:
                if f.endswith('_multi_k_summary.csv'):
                    csv_files.append(os.path.join(root, f))
        csv_files.sort()
    
    if not csv_files:
        print(f"未找到CSV文件: {args.input}")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 批量预测
    predict_batch(model, csv_files, feature_stats, config, args.output, device)


if __name__ == "__main__":
    main()


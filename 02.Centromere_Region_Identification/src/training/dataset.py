#!/usr/bin/env python3
"""
数据集模块 - 加载多k-mer CSV数据用于着丝粒预测
"""

import os
import glob
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import random


class ChromosomeDataset(Dataset):
    """
    染色体数据集
    每个样本是一条完整的染色体，包含所有bin的多k-mer特征和标签
    """
    
    # CSV列名到索引的映射
    FEATURE_COLUMNS = [
        '64_highlighted_percent', '64_coverage_depth_avg',
        '128_highlighted_percent', '128_coverage_depth_avg',
        '256_highlighted_percent', '256_coverage_depth_avg',
        '512_highlighted_percent', '512_coverage_depth_avg'
    ]
    
    def __init__(
        self,
        csv_files: List[str],
        normalize: bool = True,
        feature_stats: Optional[Dict] = None
    ):
        """
        Args:
            csv_files: CSV文件路径列表
            normalize: 是否对特征进行归一化
            feature_stats: 特征统计信息（用于归一化），格式为 {'mean': array, 'std': array}
        """
        self.csv_files = csv_files
        self.normalize = normalize
        self.feature_stats = feature_stats
        
        # 预加载所有数据
        self.data = []
        self.metadata = []
        
        for csv_file in csv_files:
            features, labels, positions, meta = self._load_csv(csv_file)
            if features is not None:
                self.data.append({
                    'features': features,
                    'labels': labels,
                    'positions': positions
                })
                self.metadata.append(meta)
        
        # 计算或使用提供的归一化统计
        if self.normalize:
            if self.feature_stats is None:
                self.feature_stats = self._compute_stats()
            self._normalize_data()
    
    def _load_csv(self, csv_file: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """加载单个CSV文件"""
        meta = {
            'file': csv_file,
            'name': os.path.basename(csv_file)
        }
        
        try:
            features_list = []
            labels_list = []
            positions_list = []
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # 提取位置信息
                    start = int(row['start'])
                    end = int(row['end'])
                    positions_list.append([start, end])
                    
                    # 提取标签
                    has_cen = int(row['has_cen'])
                    labels_list.append(has_cen)
                    
                    # 提取特征
                    feat = []
                    for col in self.FEATURE_COLUMNS:
                        val = float(row[col])
                        feat.append(val)
                    features_list.append(feat)
            
            if len(features_list) == 0:
                print(f"Warning: 空文件 {csv_file}")
                return None, None, None, meta
            
            features = np.array(features_list, dtype=np.float32)
            labels = np.array(labels_list, dtype=np.float32)
            positions = np.array(positions_list, dtype=np.int64)
            
            meta['seq_len'] = len(features)
            meta['pos_count'] = int(labels.sum())
            meta['neg_count'] = len(labels) - meta['pos_count']
            
            return features, labels, positions, meta
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            return None, None, None, meta
    
    def _compute_stats(self) -> Dict:
        """计算所有数据的特征统计信息"""
        all_features = []
        for item in self.data:
            all_features.append(item['features'])
        
        if len(all_features) == 0:
            return {'mean': np.zeros(8), 'std': np.ones(8)}
        
        all_features = np.concatenate(all_features, axis=0)
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0)
        std[std < 1e-6] = 1.0  # 避免除零
        
        return {'mean': mean, 'std': std}
    
    def _normalize_data(self):
        """对数据进行归一化"""
        mean = self.feature_stats['mean']
        std = self.feature_stats['std']
        
        for item in self.data:
            item['features'] = (item['features'] - mean) / std
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            'features': torch.from_numpy(item['features']),
            'labels': torch.from_numpy(item['labels']),
            'positions': torch.from_numpy(item['positions']),
            'metadata': self.metadata[idx]
        }
    
    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        total_bins = sum(item['features'].shape[0] for item in self.data)
        total_pos = sum(meta['pos_count'] for meta in self.metadata)
        total_neg = sum(meta['neg_count'] for meta in self.metadata)
        
        return {
            'num_samples': len(self.data),
            'total_bins': total_bins,
            'total_positive': total_pos,
            'total_negative': total_neg,
            'pos_ratio': total_pos / total_bins if total_bins > 0 else 0,
            'feature_stats': self.feature_stats
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数，处理变长序列
    由于每条染色体长度不同，这里不进行padding，而是逐个处理
    """
    # 对于batch_size=1的情况，直接返回
    if len(batch) == 1:
        return batch[0]
    
    # 对于多个样本，返回列表（训练时逐个处理）
    return {
        'features': [item['features'] for item in batch],
        'labels': [item['labels'] for item in batch],
        'positions': [item['positions'] for item in batch],
        'metadata': [item['metadata'] for item in batch]
    }


def find_csv_files(data_dir: str, pattern: str = "*_multi_k_summary.csv") -> List[str]:
    """
    递归查找所有匹配的CSV文件
    
    Args:
        data_dir: 数据根目录
        pattern: 文件名模式
    
    Returns:
        CSV文件路径列表
    """
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('_multi_k_summary.csv'):
                csv_files.append(os.path.join(root, file))
    
    csv_files.sort()
    return csv_files


def split_dataset(
    csv_files: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    划分数据集
    
    Args:
        csv_files: 所有CSV文件列表
        train_ratio, val_ratio, test_ratio: 划分比例
        seed: 随机种子
    
    Returns:
        (train_files, val_files, test_files)
    """
    random.seed(seed)
    files = csv_files.copy()
    random.shuffle(files)
    
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    
    return train_files, val_files, test_files


def create_dataloaders(
    data_dir: str,
    batch_size: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    创建训练、验证、测试数据加载器
    
    Returns:
        (train_loader, val_loader, test_loader, feature_stats)
    """
    # 查找所有CSV文件
    csv_files = find_csv_files(data_dir)
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    if len(csv_files) == 0:
        raise ValueError(f"在 {data_dir} 中未找到CSV文件")
    
    # 划分数据集
    train_files, val_files, test_files = split_dataset(
        csv_files, train_ratio, val_ratio, test_ratio, seed
    )
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    print(f"测试集: {len(test_files)} 个文件")
    
    # 创建训练集（计算归一化统计）
    train_dataset = ChromosomeDataset(train_files, normalize=True)
    feature_stats = train_dataset.feature_stats
    
    # 创建验证集和测试集（使用训练集的统计信息）
    val_dataset = ChromosomeDataset(val_files, normalize=True, feature_stats=feature_stats)
    test_dataset = ChromosomeDataset(test_files, normalize=True, feature_stats=feature_stats)
    
    # 打印统计信息
    print("\n训练集统计:")
    train_stats = train_dataset.get_stats()
    print(f"  样本数: {train_stats['num_samples']}")
    print(f"  总bin数: {train_stats['total_bins']}")
    print(f"  正样本: {train_stats['total_positive']} ({train_stats['pos_ratio']*100:.2f}%)")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, feature_stats


if __name__ == "__main__":
    # 测试数据加载
    data_dir = "/home/centromere_area_prediction_v1/embedding/multi_k_summary/20251129_093947"
    
    csv_files = find_csv_files(data_dir)
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    if csv_files:
        # 测试加载单个文件
        dataset = ChromosomeDataset(csv_files[:5])
        print(f"\n数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本特征形状: {sample['features'].shape}")
            print(f"样本标签形状: {sample['labels'].shape}")
            print(f"样本位置形状: {sample['positions'].shape}")
            print(f"正样本数: {sample['labels'].sum().item()}")
            
            stats = dataset.get_stats()
            print(f"\n数据集统计: {stats}")






#!/usr/bin/env python3
"""
配置文件 - 着丝粒预测Transformer模型
"""

import os
from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class ModelConfig:
    """模型配置"""
    # Transformer编码器配置
    d_model: int = 128          # 特征维度
    nhead: int = 8              # 注意力头数
    num_layers: int = 4         # Transformer层数
    dim_feedforward: int = 512  # 前馈层隐藏维度
    dropout: float = 0.2        # Dropout率
    max_seq_len: int = 50000    # 最大序列长度（位置编码缓存）
    
    # 输入特征配置
    input_features: int = 8     # 输入特征数（4个k值 × 2个指标）
    
    # 多尺度卷积配置
    conv_kernels: List[int] = field(default_factory=lambda: [3, 11, 25])
    conv_channels: int = 64


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 1         # 批量大小（每个样本是一条染色体）
    learning_rate: float = 5e-4 # 学习率（提高以加速收敛）
    num_epochs: int = 100       # 训练轮数
    pos_weight: float = 50.0    # 正样本权重（大幅提高以处理1.5%的极度不平衡）
    weight_decay: float = 1e-5  # 权重衰减
    
    # 早停配置
    patience: int = 20          # 早停耐心值（增加以给模型更多学习时间）
    min_delta: float = 1e-5     # 最小改善阈值（降低以更敏感）
    
    # 数据划分
    train_ratio: float = 0.8    # 训练集比例
    val_ratio: float = 0.1      # 验证集比例
    test_ratio: float = 0.1     # 测试集比例
    
    # 保存配置
    save_every: int = 10        # 每N个epoch保存一次
    log_every: int = 1          # 每N个epoch记录一次


@dataclass
class InferenceConfig:
    """推理配置"""
    threshold: float = 0.3      # 分类阈值（降低以提高召回率）
    start_threshold: float = 0.1  # 区间起点阈值
    end_threshold: float = 0.1    # 区间终点阈值
    top_n: int = 5              # 返回top-N预测区间
    nms_iou: float = 0.3        # NMS IoU阈值
    min_region_bins: int = 3    # 最小区间长度（bin数）


@dataclass
class Config:
    """总配置"""
    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # 路径配置
    data_dir: str = "/home/centromere_area_prediction_v1/embedding/multi_k_summary"
    output_dir: str = "/home/centromere_area_prediction_v1/training/outputs"
    checkpoint_dir: str = "/home/centromere_area_prediction_v1/training/checkpoints"
    log_dir: str = "/home/centromere_area_prediction_v1/training/logs"
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 随机种子
    seed: int = 42
    
    def __post_init__(self):
        """初始化后创建必要的目录"""
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)


def count_parameters(model) -> int:
    """统计模型可训练参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_config() -> Config:
    """获取默认配置"""
    return Config()


if __name__ == "__main__":
    config = get_config()
    print(f"Model config: {config.model}")
    print(f"Training config: {config.training}")
    print(f"Inference config: {config.inference}")
    print(f"Device: {config.device}")


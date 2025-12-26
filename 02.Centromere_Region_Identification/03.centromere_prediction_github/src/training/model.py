#!/usr/bin/env python3
"""
Transformer模型 - 着丝粒区域预测
基于纯Transformer Encoder架构，输入多k-mer特征，预测每个位置是否为着丝粒
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 50000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        self.max_len = max_len
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            print(f"Warning: 序列长度 {seq_len} 超过位置编码缓存 {self.max_len}，将截断")
            seq_len = self.max_len
            x = x[:, :seq_len, :]
        
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiScaleConv1D(nn.Module):
    """多尺度1D卷积模块"""
    
    def __init__(self, in_channels: int, out_channels: int = 64, kernels: list = [3, 11, 25]):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernels:
            padding = k // 2  # same padding
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.out_channels = out_channels * len(kernels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels * num_kernels, seq_len)
        """
        outputs = []
        target_len = x.size(2)
        
        for conv in self.convs:
            out = conv(x)
            # 确保输出长度一致
            if out.size(2) != target_len:
                out = out[:, :, :target_len]
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)


class CentromereTransformer(nn.Module):
    """
    着丝粒预测Transformer模型
    
    输入：(batch, seq_len, 8) - 8个特征（4个k值 × 2个指标）
    输出：(batch, seq_len, 1) - 每个位置的着丝粒概率
    """
    
    def __init__(
        self,
        input_features: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        max_seq_len: int = 50000,
        conv_kernels: list = [3, 11, 25],
        conv_channels: int = 64
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_features = input_features
        
        # 输入投影层：将8维特征投影到d_model维
        self.input_projection = nn.Linear(input_features, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 使用 (batch, seq, feature) 格式
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 多尺度卷积模块
        self.multi_scale_conv = MultiScaleConv1D(d_model, conv_channels, conv_kernels)
        multi_scale_out = conv_channels * len(conv_kernels)  # 192
        
        # 区间预测头（用于预测着丝粒区间）
        self.range_head = nn.Sequential(
            nn.Linear(multi_scale_out, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # [start_score, end_score, confidence]
        )
        
        # 逐位置分类头（预测每个位置是否为着丝粒）
        self.position_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        x: torch.Tensor, 
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (batch, seq_len, input_features) 输入特征
            src_key_padding_mask: (batch, seq_len) padding掩码，True表示padding位置
        
        Returns:
            position_probs: (batch, seq_len, 1) 每个位置的着丝粒概率
            range_scores: (batch, seq_len, 3) 区间预测分数 [start, end, confidence]
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        encoder_output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # encoder_output: (batch, seq_len, d_model)
        
        # 逐位置分类
        position_probs = self.position_head(encoder_output)  # (batch, seq_len, 1)
        
        # 多尺度卷积用于区间预测
        # 转换为 (batch, d_model, seq_len) 供卷积使用
        conv_input = encoder_output.transpose(1, 2)
        multi_scale_features = self.multi_scale_conv(conv_input)  # (batch, 192, seq_len)
        multi_scale_features = multi_scale_features.transpose(1, 2)  # (batch, seq_len, 192)
        
        # 区间预测
        range_scores = self.range_head(multi_scale_features)  # (batch, seq_len, 3)
        
        return position_probs, range_scores
    
    def predict_positions(
        self, 
        x: torch.Tensor, 
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        预测每个位置是否为着丝粒
        
        Args:
            x: (batch, seq_len, input_features)
            threshold: 分类阈值
        
        Returns:
            predictions: (batch, seq_len) 二值预测
        """
        self.eval()
        with torch.no_grad():
            position_probs, _ = self.forward(x)
            predictions = (position_probs.squeeze(-1) > threshold).long()
        return predictions


def create_model(config) -> CentromereTransformer:
    """根据配置创建模型"""
    model = CentromereTransformer(
        input_features=config.model.input_features,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        max_seq_len=config.model.max_seq_len,
        conv_kernels=config.model.conv_kernels,
        conv_channels=config.model.conv_channels
    )
    return model


if __name__ == "__main__":
    # 测试模型
    from config import get_config, count_parameters
    
    config = get_config()
    model = create_model(config)
    
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 1000
    x = torch.randn(batch_size, seq_len, config.model.input_features)
    
    position_probs, range_scores = model(x)
    print(f"输入形状: {x.shape}")
    print(f"位置概率形状: {position_probs.shape}")
    print(f"区间分数形状: {range_scores.shape}")






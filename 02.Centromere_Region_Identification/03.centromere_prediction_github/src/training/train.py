#!/usr/bin/env python3
"""
训练脚本 - 着丝粒预测Transformer模型
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

from config import get_config, count_parameters, Config
from model import create_model, CentromereTransformer
from dataset import create_dataloaders, find_csv_files


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def compute_metrics(preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5, find_best_threshold: bool = False) -> dict:
    """计算评估指标"""
    
    # 如果需要，搜索最佳阈值
    best_threshold = threshold
    if find_best_threshold and labels.sum() > 0:
        best_f1 = 0
        for t in np.arange(0.05, 0.95, 0.05):
            binary = (preds > t).astype(int)
            _, _, f1_t, _ = precision_recall_fscore_support(
                labels, binary, average='binary', zero_division=0
            )
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = t
    
    # 二值化预测
    binary_preds = (preds > best_threshold).astype(int)
    
    # 基础指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_preds, average='binary', zero_division=0
    )
    
    # AUC（需要概率值）
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.0  # 当只有一个类别时
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()
    
    # 准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # 预测概率统计
    pred_mean = float(preds.mean())
    pred_max = float(preds.max())
    pred_pos_mean = float(preds[labels > 0.5].mean()) if (labels > 0.5).any() else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'accuracy': accuracy,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'threshold': best_threshold,
        'pred_mean': pred_mean,
        'pred_max': pred_max,
        'pred_pos_mean': pred_pos_mean
    }


def train_epoch(
    model: CentromereTransformer,
    train_loader,
    optimizer,
    criterion,
    device: str,
    epoch: int
) -> dict:
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(train_loader):
        # 处理单个样本（batch_size=1时）
        if isinstance(batch['features'], torch.Tensor):
            features = batch['features'].unsqueeze(0).to(device)
            labels = batch['labels'].unsqueeze(0).to(device)
        else:
            # 多个样本时逐个处理
            for i in range(len(batch['features'])):
                features = batch['features'][i].unsqueeze(0).to(device)
                labels = batch['labels'][i].unsqueeze(0).to(device)
                
                optimizer.zero_grad()
                position_probs, _ = model(features)
                position_probs = position_probs.squeeze(-1)
                
                loss = criterion(position_probs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                all_preds.extend(position_probs.detach().cpu().numpy().flatten())
                all_labels.extend(labels.detach().cpu().numpy().flatten())
            continue
        
        optimizer.zero_grad()
        
        # 前向传播
        position_probs, _ = model(features)
        position_probs = position_probs.squeeze(-1)  # (batch, seq_len)
        
        # 计算损失
        loss = criterion(position_probs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(position_probs.detach().cpu().numpy().flatten())
        all_labels.extend(labels.detach().cpu().numpy().flatten())
    
    # 计算指标
    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    metrics['loss'] = avg_loss
    
    return metrics


def validate(
    model: CentromereTransformer,
    val_loader,
    criterion,
    device: str
) -> dict:
    """验证"""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch['features'], torch.Tensor):
                features = batch['features'].unsqueeze(0).to(device)
                labels = batch['labels'].unsqueeze(0).to(device)
            else:
                for i in range(len(batch['features'])):
                    features = batch['features'][i].unsqueeze(0).to(device)
                    labels = batch['labels'][i].unsqueeze(0).to(device)
                    
                    position_probs, _ = model(features)
                    position_probs = position_probs.squeeze(-1)
                    
                    loss = criterion(position_probs, labels)
                    total_loss += loss.item()
                    all_preds.extend(position_probs.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
                continue
            
            position_probs, _ = model(features)
            position_probs = position_probs.squeeze(-1)
            
            loss = criterion(position_probs, labels)
            total_loss += loss.item()
            all_preds.extend(position_probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(val_loader)
    # 验证时搜索最佳阈值
    metrics = compute_metrics(np.array(all_preds), np.array(all_labels), find_best_threshold=True)
    metrics['loss'] = avg_loss
    
    return metrics


def save_checkpoint(
    model: CentromereTransformer,
    optimizer,
    epoch: int,
    metrics: dict,
    feature_stats: dict,
    config: Config,
    filepath: str
):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'feature_stats': feature_stats,
        'config': {
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'inference': config.inference.__dict__
        }
    }
    torch.save(checkpoint, filepath)
    print(f"检查点已保存: {filepath}")


def train(config: Config, data_dir: str = None):
    """主训练函数"""
    print("=" * 60)
    print("着丝粒预测Transformer模型训练")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 设置设备
    device = config.device
    print(f"使用设备: {device}")
    
    # 数据目录
    if data_dir is None:
        # 查找最新的数据目录
        base_dir = config.data_dir
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if subdirs:
            subdirs.sort(reverse=True)
            data_dir = os.path.join(base_dir, subdirs[0])
        else:
            data_dir = base_dir
    
    print(f"数据目录: {data_dir}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, feature_stats = create_dataloaders(
        data_dir=data_dir,
        batch_size=config.training.batch_size,
        train_ratio=config.training.train_ratio,
        val_ratio=config.training.val_ratio,
        test_ratio=config.training.test_ratio,
        seed=config.seed
    )
    
    # 创建模型
    model = create_model(config)
    model = model.to(device)
    print(f"\n模型参数量: {count_parameters(model):,}")
    
    # 损失函数（带正样本权重处理类别不平衡）
    # 由于模型输出已经经过Sigmoid，需要自定义加权BCELoss
    pos_weight = config.training.pos_weight
    print(f"正样本权重: {pos_weight}")
    
    def weighted_bce_loss(pred, target):
        """加权BCE损失，对正样本给予更高权重"""
        # 避免log(0)
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        # 加权损失：正样本权重为pos_weight，负样本权重为1
        weights = torch.where(target > 0.5, pos_weight, 1.0)
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        weighted_bce = weights * bce
        return weighted_bce.mean()
    
    criterion = weighted_bce_loss
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # 早停
    early_stopping = EarlyStopping(
        patience=config.training.patience,
        min_delta=config.training.min_delta
    )
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.log_dir, timestamp)
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    best_f1 = 0.0
    best_epoch = 0
    
    print("\n开始训练...")
    print("-" * 60)
    
    for epoch in range(1, config.training.num_epochs + 1):
        start_time = time.time()
        
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_metrics['f1'])
        
        epoch_time = time.time() - start_time
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('F1/train', train_metrics['f1'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        writer.add_scalar('AUC/val', val_metrics['auc'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 打印进度
        if epoch % config.training.log_every == 0:
            print(f"Epoch {epoch:3d}/{config.training.num_epochs} | "
                  f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                  f"F1: {train_metrics['f1']:.4f}/{val_metrics['f1']:.4f} | "
                  f"AUC: {val_metrics['auc']:.4f} | "
                  f"Thr: {val_metrics.get('threshold', 0.5):.2f} | "
                  f"PredMax: {val_metrics.get('pred_max', 0):.3f} | "
                  f"TP/FN: {val_metrics['tp']}/{val_metrics['fn']} | "
                  f"{epoch_time:.1f}s")
        
        # 保存最佳模型（使用AUC作为指标，因为F1在极度不平衡时波动太大）
        # 同时要求pred_max > 0.5，确保模型学会预测正类
        current_score = val_metrics['auc'] if val_metrics.get('pred_max', 0) > 0.5 else 0
        if current_score > best_f1:
            best_f1 = current_score
            best_epoch = epoch
            best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, val_metrics, feature_stats, config, best_model_path)
        
        # 定期保存
        if epoch % config.training.save_every == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, epoch, val_metrics, feature_stats, config, checkpoint_path)
        
        # 早停检查（使用AUC）
        if early_stopping(val_metrics['auc']):
            print(f"\n早停触发于 epoch {epoch}")
            break
    
    print("-" * 60)
    print(f"训练完成！最佳AUC: {best_f1:.4f} (Epoch {best_epoch})")
    
    # 测试集评估
    print("\n在测试集上评估...")
    test_metrics = validate(model, test_loader, criterion, device)
    print(f"测试集结果:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  TP: {test_metrics['tp']}, TN: {test_metrics['tn']}, FP: {test_metrics['fp']}, FN: {test_metrics['fn']}")
    
    # 保存最终模型
    final_model_path = os.path.join(config.checkpoint_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, epoch, test_metrics, feature_stats, config, final_model_path)
    
    # 保存训练结果
    results = {
        'best_epoch': best_epoch,
        'best_f1': best_f1,
        'test_metrics': test_metrics,
        'feature_stats': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in feature_stats.items()},
        'config': {
            'model': config.model.__dict__,
            'training': config.training.__dict__
        }
    }
    
    results_path = os.path.join(config.output_dir, f'training_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n训练结果已保存: {results_path}")
    
    writer.close()
    
    return model, test_metrics


def main():
    parser = argparse.ArgumentParser(description='训练着丝粒预测Transformer模型')
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--pos_weight', type=float, default=None, help='正样本权重')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config()
    
    # 覆盖配置
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.pos_weight:
        config.training.pos_weight = args.pos_weight
    if args.device:
        config.device = args.device
    
    # 开始训练
    train(config, args.data_dir)


if __name__ == "__main__":
    main()


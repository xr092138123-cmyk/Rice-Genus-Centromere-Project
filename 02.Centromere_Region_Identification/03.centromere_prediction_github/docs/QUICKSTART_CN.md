# 快速开始指南

本指南将帮助您快速上手使用着丝粒预测模型。

## 环境配置

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/centromere_prediction.git
cd centromere_prediction
```

### 2. 安装依赖

```bash
# 使用pip安装
pip install -r requirements.txt

# 或使用conda
conda create -n centromere python=3.8
conda activate centromere
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 数据准备

### 数据格式要求

您的输入数据应该是CSV格式，包含以下必需列：

| 列名 | 说明 | 类型 |
|------|------|------|
| start | 区间起始位置 | int |
| end | 区间结束位置 | int |
| has_cen | 是否为着丝粒（0或1） | int |
| 64_highlighted_percent | k=64高亮百分比 | float |
| 64_coverage_depth_avg | k=64覆盖深度 | float |
| 128_highlighted_percent | k=128高亮百分比 | float |
| 128_coverage_depth_avg | k=128覆盖深度 | float |
| 256_highlighted_percent | k=256高亮百分比 | float |
| 256_coverage_depth_avg | k=256覆盖深度 | float |
| 512_highlighted_percent | k=512高亮百分比 | float |
| 512_coverage_depth_avg | k=512覆盖深度 | float |

### 数据示例

```csv
start,end,has_cen,64_highlighted_percent,64_coverage_depth_avg,128_highlighted_percent,128_coverage_depth_avg,256_highlighted_percent,256_coverage_depth_avg,512_highlighted_percent,512_coverage_depth_avg
0,10000,0,0.15,2.3,0.12,1.8,0.10,1.5,0.08,1.2
10000,20000,1,0.85,15.2,0.82,14.5,0.78,13.8,0.75,12.9
20000,30000,1,0.90,16.5,0.88,15.8,0.85,15.1,0.82,14.3
30000,40000,0,0.18,2.5,0.15,2.1,0.13,1.9,0.10,1.5
```

### 文件命名

文件名应以 `_multi_k_summary.csv` 结尾，例如：
- `chr1_multi_k_summary.csv`
- `chr2_multi_k_summary.csv`
- `sample_genome_multi_k_summary.csv`

## 训练模型

### 基本训练

```bash
cd src/training

# 使用默认参数训练
python train.py --data_dir /path/to/your/data
```

### 自定义训练参数

```bash
python train.py \
    --data_dir /path/to/your/data \
    --epochs 100 \
    --lr 5e-4 \
    --pos_weight 50.0 \
    --device cuda
```

### 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --data_dir | 数据目录路径 | 必需 |
| --epochs | 训练轮数 | 100 |
| --lr | 学习率 | 5e-4 |
| --pos_weight | 正样本权重 | 50.0 |
| --device | 计算设备（cuda/cpu） | 自动检测 |

### 使用示例脚本

```bash
cd examples
bash run_training.sh /path/to/your/data
```

### 监控训练进度

训练时，模型会自动保存TensorBoard日志。在新终端中运行：

```bash
tensorboard --logdir=src/training/logs
```

然后在浏览器访问 `http://localhost:6006` 查看训练曲线，包括：
- 训练/验证损失
- F1 Score
- AUC
- 学习率变化

### 训练输出

训练完成后，会生成以下文件：

```
src/training/
├── checkpoints/
│   ├── best_model.pt          # 验证集上最佳模型
│   ├── final_model.pt         # 最终模型
│   └── checkpoint_epoch_*.pt  # 定期保存的检查点
├── outputs/
│   └── training_results_*.json  # 训练结果统计
└── logs/
    └── [timestamp]/           # TensorBoard日志
```

## 推理预测

### 加载模型并预测

```bash
cd src/training

# 预测单个文件
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input /path/to/test_file.csv \
    --output ./predictions

# 预测整个目录
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input /path/to/test_directory \
    --output ./predictions \
    --threshold 0.3
```

### 推理参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --checkpoint | 模型检查点路径 | 必需 |
| --input | 输入文件或目录 | 必需 |
| --output | 输出目录 | ./predictions |
| --threshold | 分类阈值 | 0.3 |
| --device | 计算设备 | 自动检测 |

### 推理输出

```
predictions/
├── predictions.json           # 详细预测结果
└── predictions_summary.csv    # 汇总表格
```

**predictions.json** 包含：
- 每个位置的预测概率
- 预测的着丝粒区域（起止位置、置信度）
- 如果有标签，还包括评估指标

**predictions_summary.csv** 包含：
- 文件名
- 序列长度
- 预测区域数量
- 最可能区域的位置和概率
- 评估指标（如果有标签）

## 评估结果

### 使用评估脚本

```bash
cd src/evaluation

# 生成汇总报告
python generate_summary_report.py /path/to/results_dir

# 计算平均指标
python calculate_average_metrics.py /path/to/results_dir

# 比较预测结果
python compare_predictions.py /path/to/results_dir

# Top-5预测评估
python evaluate_top5_prediction.py /path/to/inference_output
```

### 评估指标解读

- **Precision（精确率）**: 预测为着丝粒的区域中，真正是着丝粒的比例
- **Recall（召回率）**: 真实着丝粒区域中，被成功预测出来的比例
- **F1 Score**: 精确率和召回率的调和平均数
- **IoU**: 预测区域与真实区域的交并比
- **AUC**: ROC曲线下面积，衡量模型的整体分类能力

## 完整工作流程示例

```bash
# 1. 准备环境
git clone https://github.com/yourusername/centromere_prediction.git
cd centromere_prediction
pip install -r requirements.txt

# 2. 准备数据
# 将您的数据放在 /path/to/data 目录下
# 确保文件名以 _multi_k_summary.csv 结尾

# 3. 训练模型
cd src/training
python train.py --data_dir /path/to/data --device cuda

# 4. 监控训练（在新终端）
tensorboard --logdir=logs

# 5. 等待训练完成
# 训练会自动保存最佳模型到 checkpoints/best_model.pt

# 6. 进行预测
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input /path/to/test_data \
    --output ./predictions

# 7. 查看结果
cat predictions/predictions_summary.csv

# 8. 评估结果（如果有标签）
cd ../evaluation
python generate_summary_report.py ../training/predictions
```

## 常见问题排查

### 问题1: CUDA out of memory

**解决方案**:
1. 在 `src/training/config.py` 中减小 `max_seq_len`
2. 关闭其他占用GPU的程序
3. 使用CPU训练：`--device cpu`

### 问题2: 训练损失不下降

**可能原因**:
1. 学习率过大或过小
2. 数据质量问题
3. 类别严重不平衡

**解决方案**:
1. 尝试调整学习率：`--lr 1e-4` 或 `--lr 1e-3`
2. 检查数据分布和归一化
3. 调整正样本权重：`--pos_weight 100.0`

### 问题3: 模型预测全为0或全为1

**解决方案**:
1. 调整分类阈值：`--threshold 0.5` 或 `--threshold 0.2`
2. 在训练时调整 `pos_weight`
3. 检查数据是否平衡

### 问题4: FileNotFoundError

**解决方案**:
1. 确认数据路径正确
2. 确认文件名以 `_multi_k_summary.csv` 结尾
3. 使用绝对路径而非相对路径

### 问题5: 训练速度慢

**优化建议**:
1. 使用GPU：`--device cuda`
2. 确保CUDA版本的PyTorch已正确安装
3. 减少 `num_epochs` 或使用早停
4. 使用更小的模型（在config.py中调整）

## 进阶使用

### 调整模型配置

编辑 `src/training/config.py`：

```python
@dataclass
class ModelConfig:
    d_model: int = 256          # 增加模型容量
    nhead: int = 8
    num_layers: int = 6         # 增加层数
    dim_feedforward: int = 1024  # 增加前馈层
    dropout: float = 0.3        # 增加dropout防止过拟合
```

### 使用Python API

```python
from src.training.config import Config
from src.training.model import create_model
from src.training.train import train
from src.training.inference import load_model, predict_single_chromosome

# 训练
config = Config()
config.training.num_epochs = 50
model, metrics = train(config, data_dir="/path/to/data")

# 推理
model, feature_stats, config = load_model("checkpoints/best_model.pt")
result = predict_single_chromosome(
    model, 
    "/path/to/test.csv", 
    feature_stats, 
    config
)
print(result['predicted_regions'])
```

## 下一步

- 查看 [README.md](../README.md) 了解更多技术细节
- 查看 [模型架构文档](MODEL_ARCHITECTURE.md) 了解模型设计
- 访问 [Issues](https://github.com/yourusername/centromere_prediction/issues) 报告问题或提出建议

## 获取帮助

如果遇到问题：
1. 查看本文档的常见问题部分
2. 查看 GitHub Issues
3. 提交新的 Issue 描述您的问题

祝您使用愉快！



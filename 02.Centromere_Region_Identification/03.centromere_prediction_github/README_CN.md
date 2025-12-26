# 着丝粒区域预测

[English](README.md) | 简体中文

基于深度学习的染色体着丝粒区域预测工具，使用Transformer架构和多尺度k-mer特征。

**✨ 包含预训练模型，下载即用！**

## 项目简介

本项目提供了一个端到端的着丝粒预测解决方案，包括：
- 🎯 **预训练模型** - 直接使用，无需训练
- 🧬 基于Transformer的深度学习模型
- 📊 多尺度k-mer特征融合（64, 128, 256, 512）
- 🚀 完整的训练、推理和评估工具
- 📚 详细的中文文档

## 🚀 立即开始（3分钟上手）

### 1. 下载项目（已包含预训练模型）

```bash
git clone https://github.com/yourusername/centromere_prediction.git
cd centromere_prediction

# 查看预训练模型（已包含在checkpoints/目录中）
ls -lh checkpoints/
# best_model.pt (36MB) - 推荐使用
# final_model.pt (36MB) - 备用模型
```

### 2. 安装依赖

```bash
pip install -r requirements.txt

# 安装k-mer分析工具（从FASTA推理时需要）
conda install -c bioconda jellyfish
```

### 3. 开始预测

#### 方法一：从FASTA文件直接推理（最简单，推荐）

如果您有FASTA基因组文件和预训练模型，可以一键运行：

```bash
# 一键完成：FASTA → k-mer分析 → 特征提取 → 模型推理 → BED结果
chmod +x scripts/predict_from_fasta.sh
./scripts/predict_from_fasta.sh genome.fasta checkpoints/best_model.pt

# 查看结果
cat predictions_output/predictions/predictions_summary.csv
```

**详细教程**: 📖 [从FASTA到预测结果完整指南](docs/FROM_FASTA_TO_PREDICTION.md)

### 方法二：从特征CSV推理

如果您已有特征CSV文件：

```bash
cd src/training
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input /path/to/features.csv \
    --output ./predictions
```

### 方法三：训练自己的模型

```bash
cd src/training
python train.py --data_dir /path/to/data --device cuda
```

## 主要特性

- 🎁 **开箱即用**: 包含预训练模型，下载即可开始预测
- ✨ **先进架构**: Transformer + 多尺度卷积
- 🎯 **高精度**: F1 Score 0.82-0.93, IoU 0.70-0.88
- ⚡ **高效**: GPU加速，~10ms/1000bins
- 📦 **易用**: 一键脚本，完整的工具链和文档
- 🔧 **灵活**: 高度可配置的模型和训练参数

## 文档

- 📖 [快速开始指南](docs/QUICKSTART_CN.md) - 详细的使用教程
- 🏗️ [模型架构](docs/MODEL_ARCHITECTURE.md) - 技术细节和设计
- 📝 [数据格式](docs/DATA_FORMAT.md) - 输入数据规范
- 🤝 [贡献指南](CONTRIBUTING.md) - 如何参与开发

## 项目结构

```
├── checkpoints/           # 预训练模型 ⭐
│   ├── best_model.pt     # 最佳模型（推荐）
│   ├── final_model.pt    # 最终模型
│   └── README.md         # 模型说明
├── src/
│   ├── training/         # 训练和推理
│   ├── preprocessing/    # 数据预处理
│   ├── postprocessing/   # 结果后处理
│   └── evaluation/       # 评估工具
├── scripts/              # 实用脚本
│   └── predict_from_fasta.sh  # 一键推理
├── docs/                 # 详细文档
├── examples/             # 使用示例
└── README.md            # 本文件
```

## 数据格式

输入为CSV文件，包含以下列：
- `start`, `end`: 区间位置
- `has_cen`: 着丝粒标签（0/1）
- 多个k-mer特征列（8维）

详见 [数据格式文档](docs/DATA_FORMAT.md)

## 性能指标

| 指标 | 典型值 |
|------|--------|
| Precision | 0.85-0.95 |
| Recall | 0.80-0.92 |
| F1 Score | 0.82-0.93 |
| IoU | 0.70-0.88 |
| AUC | 0.90-0.98 |

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (可选，用于GPU)

详见 [requirements.txt](requirements.txt)

## 常见问题

**Q: 如何准备数据？**  
A: 参考 [数据格式文档](docs/DATA_FORMAT.md)，准备包含多k-mer特征的CSV文件。

**Q: 训练需要多久？**  
A: 取决于数据量，通常几小时到一天。使用GPU可大幅加速。

**Q: 如何调优模型？**  
A: 查看 [模型架构文档](docs/MODEL_ARCHITECTURE.md) 的超参数调优部分。

更多问题请查看 [快速开始指南](docs/QUICKSTART_CN.md)

## 示例

### 训练

```python
from src.training import train, Config

config = Config()
config.training.num_epochs = 50
model, metrics = train(config, data_dir="/path/to/data")
```

### 推理

```python
from src.training import load_model, predict_single_chromosome

model, stats, config = load_model("checkpoints/best_model.pt")
result = predict_single_chromosome(model, "test.csv", stats, config)
print(result['predicted_regions'])
```

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{centromere_prediction,
  title = {Centromere Area Prediction with Transformer},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/centromere_prediction}
}
```

## 贡献

欢迎贡献！请查看 [贡献指南](CONTRIBUTING.md)

## 许可证

本项目采用 [MIT License](LICENSE)

## 更新日志

查看 [CHANGELOG.md](CHANGELOG.md) 了解版本历史

## 致谢

感谢所有贡献者的支持！

## 联系方式

- GitHub Issues: [问题反馈](https://github.com/yourusername/centromere_prediction/issues)
- Email: your.email@example.com

---

⭐ 如果觉得有用，请给项目一个星标！



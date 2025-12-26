好的，我已经严格按照您提供的最新内容，将整个 **README.md** 完整翻译为英文，并确保所有链接、项目结构和技术术语与您的 GitHub 仓库 (`xr092138123-cmyk/Oryza-Genus-Centromere-Project`) 完美对应。

---

# Centromere Area Prediction

A deep learning model based on the **Transformer architecture** for predicting chromosomal centromere regions. This model utilizes **multi-scale k-mer features** to accurately identify and localize centromere regions across genomic sequences.

**✨ Includes pretrained models - Ready to use out of the box!**

## Project Overview

The centromere is a critical structural region of the chromosome that plays a key role during cell division. This project leverages deep learning technology to automatically recognize and localize centromere regions based on sequence characteristics.

### Key Features

* **🎁 Pretrained Models**: Includes ready-to-use models; no training required.
* **Transformer Architecture**: Employs a pure Transformer Encoder to capture long-range sequence dependencies.
* **Multi-scale Features**: Integrates statistical features from multiple k-mer scales (64, 128, 256, 512).
* **End-to-End Training**: Predicts centromere locations directly from raw genomic features.
* **Highly Configurable**: Flexible model configurations and training parameters.
* **Complete Toolchain**: Includes full workflows for training, inference, evaluation, and visualization.

### Model Architecture

```text
Input Features (8-dimensional)
    ↓
Linear Projection → d_model dimensions
    ↓
Positional Encoding
    ↓
Transformer Encoder (Multiple Layers)
    ↓
    ├─→ Point-wise Classification Head → Centromere Probability
    └─→ Multi-scale Convolutional Head → Interval Prediction Score

```

---

## Environment Requirements

### System Requirements

* Python 3.8+
* CUDA 11.0+ (Optional, for GPU acceleration)

### Dependencies

```bash
pip install -r requirements.txt

```

**Core Dependencies:**

* PyTorch >= 1.10.0
* numpy >= 1.21.0
* pandas >= 1.3.0
* scikit-learn >= 1.0.0
* matplotlib >= 3.4.0
* tensorboard >= 2.8.0

---

## Quick Start

### Option 1: Inference from FASTA (Recommended for End Users)

If you have a genome FASTA file and a pretrained model:

```bash
# Install dependencies
pip install -r requirements.txt
conda install -c bioconda jellyfish

# One-command pipeline: FASTA → k-mer analysis → feature extraction → inference → BED output
chmod +x scripts/predict_from_fasta.sh
./scripts/predict_from_fasta.sh genome.fasta checkpoints/best_model.pt

# View results
cat predictions_output/predictions/predictions_summary.csv

```

**Detailed Guide**: 📖 [From FASTA to Predictions](docs/FROM_FASTA_TO_PREDICTION.md)

### Option 2: Inference from Feature CSV

If you already have pre-processed feature CSV files:

```bash
cd src/training
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input /path/to/features.csv \
    --output ./predictions \
    --threshold 0.3

```

**Output files:**

* `predictions.json`: Detailed predictions with probabilities.
* `predictions_summary.csv`: Summary table.
* `centromeres.bed`: BED format for genome browsers (IGV, UCSC).

### Option 3: Train Your Own Model

#### 1. Prepare Training Data

Input data should be in CSV format with the following columns:

* `start`, `end`: Genomic coordinates.
* `has_cen`: Label (0 or 1).
* Multi-scale k-mer features (8 columns for k=64, 128, 256, 512).

File naming convention: `*_multi_k_summary.csv`
See [Data Format](docs/DATA_FORMAT.md) for details.

#### 2. Train Model

```bash
cd src/training
python train.py --data_dir /path/to/your/data --device cuda

```

Training parameters:

```bash
python train.py \
    --data_dir /path/to/data \
    --epochs 100 \
    --lr 5e-4 \
    --pos_weight 50.0 \
    --device cuda

```

#### 3. Monitor Training

```bash
tensorboard --logdir=training/logs

```

Open `http://localhost:6006` in your browser.

---

## Project Structure

```text
Oryza-Genus-Centromere-Project/02.Centromere_Region_Identification/03.centromere_prediction_github/
├── src/
│   ├── training/           # Training module
│   │   ├── config.py       # Configuration file
│   │   ├── model.py        # Transformer model definition
│   │   ├── dataset.py      # Data loading logic
│   │   ├── train.py        # Training script
│   │   └── inference.py    # Inference script
│   └── evaluation/         # Evaluation module
│       ├── evaluate_top5_prediction.py
│       ├── generate_summary_report.py
│       ├── compare_predictions.py
│       ├── calculate_average_metrics.py
│       └── process_coverage_statistics.py
├── examples/               # Example scripts
│   └── run_training.sh     # Training example script
├── docs/                   # Documentation
├── requirements.txt        # Dependencies
├── .gitignore              # Git ignore file
└── README.md               # This file

```

---

## Model Configuration

Adjust the following parameters in `src/training/config.py`:

### Model Parameters

* `d_model`: Transformer feature dimension (Default: 128)
* `nhead`: Number of attention heads (Default: 8)
* `num_layers`: Number of Transformer layers (Default: 4)
* `dim_feedforward`: Feed-forward layer dimension (Default: 512)
* `dropout`: Dropout rate (Default: 0.2)

### Training Parameters

* `batch_size`: Batch size (Default: 1)
* `learning_rate`: Learning rate (Default: 5e-4)
* `num_epochs`: Number of training epochs (Default: 100)
* `pos_weight`: Positive sample weight (Default: 50.0, for handling class imbalance)
* `patience`: Early stopping patience (Default: 20)

### Inference Parameters

* `threshold`: Classification threshold (Default: 0.3)
* `min_region_bins`: Minimum interval length (Default: 3)
* `top_n`: Number of top-N prediction intervals to return (Default: 5)

---

## Evaluation Metrics

The model is evaluated using the following metrics:

* **Precision**: TP / (TP + FP)
* **Recall**: TP / (TP + FN)
* **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
* **IoU (Intersection over Union)**: Overlap between predicted and ground truth regions.
* **AUC**: Area Under the ROC Curve.

---

## Usage Examples

### Training a Custom Model

```python
from src.training.config import Config
from src.training.model import create_model
from src.training.train import train

# Create configuration
config = Config()
config.training.num_epochs = 50
config.training.learning_rate = 1e-4

# Start training
model, metrics = train(config, data_dir="/path/to/data")

```

### Inference for a Single Chromosome

```python
from src.training.inference import load_model, predict_single_chromosome

# Load the model
model, feature_stats, config = load_model("checkpoints/best_model.pt")

# Predict
result = predict_single_chromosome(
    model, 
    "path/to/chromosome.csv", 
    feature_stats, 
    config
)

print(f"Predicted Regions: {result['predicted_regions']}")

```

---

## Performance Optimization Suggestions

1. **Class Imbalance**: Adjust the `pos_weight` parameter (typically set to the ratio of negative to positive samples).
2. **Overfitting**: Increase the `dropout` rate or utilize the early stopping mechanism.
3. **Underfitting**: Increase model capacity (`d_model`, `num_layers`) or the number of training epochs.
4. **Out of Memory**: Reduce `max_seq_len` or use gradient accumulation.
5. **Speed Up Training**: Use a GPU (`--device cuda`).

---

## Monitoring Training

Use TensorBoard to monitor the training process:

```bash
tensorboard --logdir=training/logs

```

Then visit `http://localhost:6006` in your browser.

---

## FAQ

### Q: Why am I getting NaN loss during training?

A: It could be due to a high learning rate or data normalization issues. Try lowering the learning rate or checking data quality.

### Q: The model predicts all 0s or all 1s?

A: Adjust the `pos_weight` parameter and the classification `threshold`.

### Q: How to handle extremely long sequences?

A: You can process them in segments or increase the `max_seq_len` parameter (requires more memory).

### Q: Is multi-GPU training supported?

A: The current version uses a single GPU. Multi-GPU support can be implemented via `torch.nn.DataParallel`.

---

## Technical Details

### Positional Encoding

Uses standard sine/cosine positional encoding:


### Loss Function

Uses Weighted Binary Cross Entropy (BCE) Loss to handle class imbalance:


### Data Normalization

Features are normalized using Z-score standardization:


---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{centromere_prediction,
  title = {Centromere Area Prediction with Transformer},
  author = {xr092138123-cmyk},
  year = {2024},
  url = {https://github.com/xr092138123-cmyk/Oryza-Genus-Centromere-Project}
}

```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

* Project Homepage: [GitHub Repository](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/xr092138123-cmyk/Oryza-Genus-Centromere-Project)
* Feedback: [Issues](https://github.com/xr092138123-cmyk/Oryza-Genus-Centromere-Project/issues)

---

## Changelog

### v1.0.0 (2024-12)

* Initial release.
* Implementation of Transformer-based centromere prediction model.
* Support for multi-scale k-mer features.
* Full toolchain for training, inference, and evaluation.

## Acknowledgments

Thanks to all researchers who provided help and suggestions for this project.
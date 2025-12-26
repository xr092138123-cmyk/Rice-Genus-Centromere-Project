
# Pretrained Models

This directory contains trained centromere prediction models that can be directly used for inference.

## Model Files

### best_model.pt ⭐ Recommended

* **Size**: 36 MB
* **Description**: The model that achieved the best performance on the validation set.
* **Recommended Use**: Production environments and actual genomic predictions.
* **Performance Metrics**:
* F1 Score: 0.82-0.93
* IoU: 0.70-0.88
* Precision: 0.85-0.95
* Recall: 0.80-0.92



### final_model.pt

* **Size**: 36 MB
* **Description**: The final model state after the last training epoch.
* **Recommended Use**: Backup model or comparative testing.

## Quick Usage

### Method 1: One-click Inference from FASTA (Easiest)

```bash
# Ensure the script has execution permissions
chmod +x scripts/predict_from_fasta.sh

# Run one-click pipeline (replace your_genome.fasta with your actual genome file)
./scripts/predict_from_fasta.sh your_genome.fasta checkpoints/best_model.pt

# View results
cat predictions_output/predictions/predictions_summary.csv

```

### Method 2: Inference from Feature CSV

If you already have pre-processed feature CSV files:

```bash
cd src/training

python inference.py \
    --checkpoint ../../checkpoints/best_model.pt \
    --input your_features.csv \
    --output ../../predictions \
    --threshold 0.3

```

### Method 3: Python API Usage

```python
import sys
sys.path.append('src/training')

from inference import load_model, predict_single_chromosome

# Load model
model, feature_stats, config = load_model('checkpoints/best_model.pt')

# Predict for a single chromosome
result = predict_single_chromosome(
    model, 
    'your_chromosome.csv',
    feature_stats,
    config,
    device='cuda'  # or 'cpu'
)

# Inspect predicted regions
print("Predicted Centromere Regions:")
for i, region in enumerate(result['predicted_regions'], 1):
    print(f"Region {i}: {region['start_pos']:,} - {region['end_pos']:,} bp")
    print(f"  Average Probability: {region['avg_prob']:.4f}")
    print(f"  Length: {region['length_bp']:,} bp")

```

## Testing Example

### Quickly check if the model is usable:

```bash
# Test model loading
python -c "
import torch
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
print('✓ Model loaded successfully')
print(f'Training Epochs: {checkpoint.get(\"epoch\", \"Unknown\")}')
print(f'Validation F1: {checkpoint.get(\"metrics\", {}).get(\"f1\", \"Unknown\")}')
"

```

## Model Details

### Architecture Information

* **Base Architecture**: Transformer Encoder
* **Parameters**: ~500K
* **Input Features**: 8-dimensional (4 k-values × 2 statistics)
* **Output**: Centromere probability for each position (bin)

### Training Data

* **Source**: Genomic data from multiple species.
* **Features**: Multi-scale k-mer statistics ().

### Hyperparameter Configuration

```python
ModelConfig:
  d_model: 128
  nhead: 8
  num_layers: 4
  dim_feedforward: 512
  dropout: 0.2

TrainingConfig:
  learning_rate: 5e-4
  pos_weight: 50.0
  batch_size: 1
  num_epochs: 100

```

## Performance Metrics

### Typical Performance on Test Sets

| Metric | Range | Description |
| --- | --- | --- |
| **F1 Score** | 0.82-0.93 | Harmonic mean of Precision and Recall |
| **IoU** | 0.70-0.88 | Intersection over Union with ground truth |
| **Precision** | 0.85-0.95 | Proportion of correct predictions among all predicted centromeres |
| **Recall** | 0.80-0.92 | Proportion of actual centromeres correctly identified |
| **AUC** | 0.90-0.98 | Area Under the ROC Curve |

### Performance at Different Thresholds

| Threshold | Precision | Recall | F1 | Use Case |
| --- | --- | --- | --- | --- |
| 0.2 | Lower | High | Medium | High recall needed, minimize false negatives |
| 0.3 | Medium | Med-High | High | **Default Recommended**, balanced performance |
| 0.5 | High | Medium | Medium | High precision needed, minimize false positives |

## Applicability

### Suitable Species/Genomes

* ✅ Validated across multiple plant species.
* ✅ High performance in regions rich in repetitive sequences.
* ✅ Suitable for standard chromosome structures.

### Limitations and Precautions

* ⚠️ For new species, threshold adjustment or fine-tuning may be required.
* ⚠️ Very short chromosomes (<100kb) may yield sub-optimal results.
* ⚠️ Highly heterogeneous genomes might require specialized training.
* ⚠️ Recommended to validate on a small subset of your own data before large-scale application.

## Inference Speed

### CPU

* **Single Chromosome** (~5Mb): ~30-60 seconds
* **Whole Genome** (~200Mb): ~10-30 minutes

### GPU (CUDA)

* **Single Chromosome** (~5Mb): ~3-10 seconds
* **Whole Genome** (~200Mb): ~1-5 minutes

*Note: Speed varies depending on hardware configuration and sequence length.*

## Troubleshooting

### Model Loading Fails

If you encounter version compatibility issues:

```python
checkpoint = torch.load('checkpoints/best_model.pt', 
                        map_location='cpu',
                        weights_only=False)

```

### CUDA Out of Memory

Use CPU for inference:

```bash
python inference.py --checkpoint checkpoints/best_model.pt ... --device cpu

```

### Prediction Results are All Zeros

* Check if the input CSV format is correct.
* Try lowering the threshold: `--threshold 0.2`.
* Verify if feature values are within a reasonable range.

## Changelog

### v1.0.0 (2024-12-19)

* ✅ Initial release.
* ✅ Trained on multi-species data.
* ✅ Validation F1 score > 0.85.

## Citation

If you use this model in your research, please cite:

```bibtex
@software{centromere_prediction_model,
  title = {Pretrained Transformer Model for Centromere Prediction},
  author = {Li Xiangrui},
  year = {2024},
  url = {https://github.com/xr092138123-cmyk/Oryza-Genus-Centromere-Project}
}

```

## License

This model is released under the **MIT License** along with the code.

## Contact

* Feedback: [GitHub Issues](https://github.com/xr092138123-cmyk/Oryza-Genus-Centromere-Project/issues)
* Email: your.email@example.com

---

**Tip**: For first-time users, it is highly recommended to start with the [Quick Start Guide](docs/QUICKSTART_CN.md)!

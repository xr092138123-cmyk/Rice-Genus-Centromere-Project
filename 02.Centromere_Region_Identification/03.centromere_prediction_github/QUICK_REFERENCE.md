这是为您翻译的英文版 **QUICK_REFERENCE.md**。我已根据您的要求，将所有内容翻译为专业英文，并更新了相关的 GitHub 仓库路径（`Oryza-Genus-Centromere-Project`），同时保持了原有的 Markdown 格式和代码块结构。

---

# Quick Reference

## Three Ways to Use

### 🚀 Method 1: One-Click Inference from FASTA (Easiest)

**Scenario**: You have a genomic FASTA file and want to get centromere predictions directly.

```bash
# Run the full pipeline with one command
./scripts/predict_from_fasta.sh genome.fasta checkpoints/best_model.pt

# View results
cat predictions_output/predictions/predictions_summary.csv

# Visualize in IGV
# Load: predictions_output/predictions/centromeres.bed

```

**Requirements**:

* ✅ Genomic FASTA file
* ✅ Pre-trained model file
* ✅ Jellyfish tool: `conda install -c bioconda jellyfish`

**Detailed Tutorial**: [docs/FROM_FASTA_TO_PREDICTION.md](docs/FROM_FASTA_TO_PREDICTION.md)

---

### 📊 Method 2: Inference from Feature CSV

**Scenario**: You already have pre-processed feature CSV files.

```bash
cd src/training
python inference.py \
    --checkpoint ../../checkpoints/best_model.pt \
    --input your_features.csv \
    --output ./predictions

```

**Requirements**:

* ✅ Feature CSV file (containing 8 k-mer feature columns)
* ✅ Pre-trained model file

**CSV Format**: [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)

---

### 🎓 Method 3: Training Your Own Model

**Scenario**: You have labeled training data and want to train a new model.

```bash
cd src/training
python train.py --data_dir /path/to/data --device cuda

# Monitor training
tensorboard --logdir=logs

```

**Requirements**:

* ✅ Labeled training data (multiple CSV files)
* ✅ GPU (optional, but strongly recommended)

**Detailed Tutorial**: [docs/QUICKSTART_CN.md](docs/QUICKSTART_CN.md)

---

## Command Cheat Sheet

### Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# k-mer analysis tool (required for FASTA inference)
conda install -c bioconda jellyfish

# Install as a Python package
pip install -e .

```

### Generate Feature CSV from FASTA

```bash
# Step 1: k-mer counting
for k in 64 128 256 512; do
    jellyfish count -m $k -s 1G -t 8 -C -o ${k}mer.jf genome.fasta
    jellyfish dump ${k}mer.jf > ${k}mer_counts.txt
done

# Step 2: Feature generation
python src/preprocessing/generate_features.py \
    --genome genome.fasta \
    --kmer-dir . \
    --output features.csv

```

### Model Inference

```bash
# Basic inference
python src/training/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input features.csv \
    --output predictions

# Adjust threshold (more sensitive)
python src/training/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input features.csv \
    --output predictions \
    --threshold 0.2

# Use CPU
python src/training/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input features.csv \
    --output predictions \
    --device cpu

```

### Convert to BED Format

```bash
# Basic conversion
python src/postprocessing/predictions_to_bed.py \
    predictions/predictions.json \
    centromeres.bed

# Retain only high-confidence regions
python src/postprocessing/predictions_to_bed.py \
    predictions/predictions.json \
    centromeres.bed \
    --min-prob 0.7

# Retain only top 3 regions
python src/postprocessing/predictions_to_bed.py \
    predictions/predictions.json \
    centromeres.bed \
    --top-n 3

# Generate detailed BED
python src/postprocessing/predictions_to_bed.py \
    predictions/predictions.json \
    centromeres_detail.bed \
    --detailed

```

### Evaluate Results

```bash
# Generate summary report
python src/evaluation/generate_summary_report.py predictions/

# Calculate average metrics
python src/evaluation/calculate_average_metrics.py predictions/

# Top-5 evaluation
python src/evaluation/evaluate_top5_prediction.py predictions/

```

### Model Training

```bash
# Basic training
python src/training/train.py --data_dir /path/to/data

# Custom parameters
python src/training/train.py \
    --data_dir /path/to/data \
    --epochs 100 \
    --lr 5e-4 \
    --pos_weight 50.0 \
    --device cuda

# Monitor training
tensorboard --logdir=src/training/logs

```

## Output File Description

### Inference Output

| File | Description |
| --- | --- |
| `predictions.json` | Detailed prediction results, containing probabilities for each position |
| `predictions_summary.csv` | Tabular summary containing predicted region information |
| `centromeres.bed` | BED format, viewable in tools like IGV |
| `centromeres_detailed.bed` | Detailed BED, containing probabilities and other info |

### Training Output

| File/Directory | Description |
| --- | --- |
| `checkpoints/best_model.pt` | Best model saved based on validation set |
| `checkpoints/final_model.pt` | Final trained model |
| `outputs/training_results_*.json` | Training result statistics |
| `logs/` | TensorBoard logs |

## Parameter Reference Table

### inference.py Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| --checkpoint | Required | Path to the model file |
| --input | Required | Input CSV file or directory |
| --output | ./predictions | Output directory |
| --threshold | 0.3 | Classification threshold (0.1-0.5) |
| --device | Auto | cuda or cpu |

### train.py Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| --data_dir | Required | Training data directory |
| --epochs | 100 | Number of training epochs |
| --lr | 5e-4 | Learning rate |
| --pos_weight | 50.0 | Weight for positive samples |
| --device | Auto | cuda or cpu |

### generate_features.py Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| --genome | Required | FASTA file |
| --kmer-dir | Required | k-mer count directory |
| --output | Required | Output CSV file |
| --bin-size | 10000 | Bin size (bp) |
| --chromosome | None | Process only the specified chromosome |

### predict_from_fasta.sh Parameters

```bash
./scripts/predict_from_fasta.sh <genome.fasta> <model.pt> [output_dir] [bin_size] [threads] [threshold]

```

| Parameter | Default | Description |
| --- | --- | --- |
| genome.fasta | Required | Genome file |
| model.pt | Required | Model file |
| output_dir | predictions_output | Output directory |
| bin_size | 10000 | Bin size |
| threads | 8 | Number of threads |
| threshold | 0.3 | Prediction threshold |

## FAQ

### Q: I only have a FASTA file, what should I do?

A: Use Method 1 - `./scripts/predict_from_fasta.sh genome.fasta model.pt`

### Q: Inference is too slow?

A: 1) Use GPU: `--device cuda`; 2) Increase the number of threads.

### Q: Prediction results are inaccurate?

A: 1) Adjust the threshold `--threshold`; 2) If it's a new species, retraining might be necessary.

### Q: How to batch process multiple genomes?

```bash
for fasta in *.fasta; do
    ./scripts/predict_from_fasta.sh $fasta model.pt ${fasta%.fasta}_results
done

```

### Q: How to view in IGV?

A: File → Load from File → Select `centromeres.bed`

### Q: Out of memory?

A: 1) Process by chromosome; 2) Use the `--chromosome` parameter.

### Q: No Jellyfish?

A: `conda install -c bioconda jellyfish` or use KMC.

## Resource Links

* 📖 [Full Documentation](README.md)
* 🚀 [Quick Start Guide](docs/QUICKSTART_CN.md)
* 🧬 [From FASTA to Prediction](docs/FROM_FASTA_TO_PREDICTION.md)
* 📊 [Data Format](docs/DATA_FORMAT.md)
* 🏗️ [Model Architecture](docs/MODEL_ARCHITECTURE.md)
* 🤝 [Contributing Guide](CONTRIBUTING.md)
* 📝 [Changelog](CHANGELOG.md)

## Contact & Support

* GitHub Issues: [Submit an Issue](https://github.com/xr092138123-cmyk/Oryza-Genus-Centromere-Project/issues)
* Email: your.email@example.com

---

**Tip**: If you are a first-time user, it is highly recommended to start with the [Quick Start Guide](docs/QUICKSTART_CN.md)!
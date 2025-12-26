# From FASTA to Prediction - Complete Workflow Guide

This document describes the step-by-step process of obtaining centromere prediction results starting from raw FASTA genomic files.

## Workflow Overview

```text
Genomic FASTA File
    ↓
Step 1: k-mer Analysis
    ↓
Step 2: Generate Feature CSV
    ↓
Step 3: Model Inference
    ↓
Step 4: View Results

```

---

## Preparation

### Required Files

* **Genomic FASTA file**: (e.g., `genome.fasta`)
* **Trained model**: (`best_model.pt`)

### Required Tools

```bash
# Install this project
pip install -r requirements.txt

# Install k-mer analysis tools (choose one of the two)
# Option 1: Jellyfish (Recommended)
conda install -c bioconda jellyfish

# Option 2: KMC
conda install -c bioconda kmc

```

---

## Complete Workflow

### Step 1: k-mer Frequency Statistics

Use Jellyfish to calculate k-mer frequencies for different k-values:

```bash
# Set variables for your genome file
GENOME="genome.fasta"
OUTPUT_DIR="kmer_analysis"
mkdir -p $OUTPUT_DIR

# Calculate k-mer frequencies for 4 k-values
for k in 64 128 256 512; do
    echo "Processing k=${k}..."
    
    # Count k-mers
    jellyfish count \
        -m $k \
        -s 1G \
        -t 8 \
        -C \
        -o ${OUTPUT_DIR}/${k}mer.jf \
        $GENOME
    
    # Export to text format
    jellyfish dump \
        ${OUTPUT_DIR}/${k}mer.jf \
        > ${OUTPUT_DIR}/${k}mer_counts.txt
    
    echo "k=${k} completed"
done

```

### Step 2: Generate Feature CSV File

Create a preprocessing script (or use the tools provided by the project):

```python
# generate_features.py
import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import defaultdict
import argparse

def load_kmer_counts(kmer_file):
    """Load k-mer count file"""
    kmer_counts = {}
    with open(kmer_file, 'r') as f:
        while True:
            kmer = f.readline().strip()
            if not kmer:
                break
            count = int(f.readline().strip())
            kmer_counts[kmer] = count
    return kmer_counts

def calculate_bin_statistics(genome_file, kmer_counts_dict, bin_size=10000):
    """
    Calculate statistical features for each bin
    
    Args:
        genome_file: Path to the FASTA file
        kmer_counts_dict: Dictionary of {k_value: kmer_counts}
        bin_size: Bin size (default 10kb)
    
    Returns:
        DataFrame with features
    """
    results = []
    
    # Read genome sequences
    for record in SeqIO.parse(genome_file, "fasta"):
        seq = str(record.seq).upper()
        seq_len = len(seq)
        
        # Split by bin
        for start in range(0, seq_len, bin_size):
            end = min(start + bin_size, seq_len)
            bin_seq = seq[start:end]
            
            if len(bin_seq) < bin_size // 2:  # Skip bins that are too short
                continue
            
            bin_features = {
                'chromosome': record.id,
                'start': start,
                'end': end,
                'has_cen': 0  # Set to 0 for inference; can be modified if real labels are available
            }
            
            # Calculate features for each k-value
            for k, kmer_counts in kmer_counts_dict.items():
                if len(bin_seq) < k:
                    bin_features[f'{k}_highlighted_percent'] = 0.0
                    bin_features[f'{k}_coverage_depth_avg'] = 0.0
                    continue
                
                # Extract all k-mers within the bin
                bin_kmers = []
                for i in range(len(bin_seq) - k + 1):
                    kmer = bin_seq[i:i+k]
                    if 'N' not in kmer:  # Skip k-mers containing 'N'
                        bin_kmers.append(kmer)
                
                if not bin_kmers:
                    bin_features[f'{k}_highlighted_percent'] = 0.0
                    bin_features[f'{k}_coverage_depth_avg'] = 0.0
                    continue
                
                # Calculate statistics
                counts = [kmer_counts.get(kmer, 0) for kmer in bin_kmers]
                
                # highlighted_percent: Proportion of high-frequency k-mers
                # Define "high-frequency" as greater than 2x the median
                median_count = np.median(counts) if counts else 0
                threshold = median_count * 2
                highlighted = sum(1 for c in counts if c > threshold)
                highlighted_percent = highlighted / len(counts) if counts else 0
                
                # coverage_depth_avg: Average coverage depth
                avg_depth = np.mean(counts) if counts else 0
                
                bin_features[f'{k}_highlighted_percent'] = highlighted_percent
                bin_features[f'{k}_coverage_depth_avg'] = avg_depth
            
            results.append(bin_features)
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Generate feature CSV from FASTA and k-mer counts')
    parser.add_argument('--genome', required=True, help='Input FASTA file')
    parser.add_argument('--kmer-dir', required=True, help='Directory containing k-mer count files')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--bin-size', type=int, default=10000, help='Bin size (default: 10000)')
    
    args = parser.parse_args()
    
    print("Loading k-mer counts...")
    kmer_counts_dict = {}
    for k in [64, 128, 256, 512]:
        kmer_file = f"{args.kmer_dir}/{k}mer_counts.txt"
        print(f"  Loading k={k}...")
        kmer_counts_dict[k] = load_kmer_counts(kmer_file)
    
    print("Calculating bin statistics...")
    df = calculate_bin_statistics(args.genome, kmer_counts_dict, args.bin_size)
    
    print(f"Saving to {args.output}...")
    df.to_csv(args.output, index=False)
    
    print(f"Done! Generated {len(df)} bins")
    print(f"\nOutput columns:")
    print(df.columns.tolist())
    print(f"\nFirst few rows:")
    print(df.head())

if __name__ == '__main__':
    main()

```

Run feature extraction:

```bash
python generate_features.py \
    --genome genome.fasta \
    --kmer-dir kmer_analysis \
    --output genome_multi_k_summary.csv \
    --bin-size 10000

```

### Step 3: Model Inference

Use the trained model to perform predictions:

```bash
cd src/training

python inference.py \
    --checkpoint ../../checkpoints/best_model.pt \
    --input ../../genome_multi_k_summary.csv \
    --output ../../predictions \
    --threshold 0.3 \
    --device cuda

```

**Parameter Descriptions**:

* **--checkpoint**: Path to the model file
* **--input**: CSV file generated in Step 2
* **--output**: Directory to save the results
* **--threshold**: Classification threshold (0.1–0.5; lower values yield higher recall)
* **--device**: Use `cuda` or `cpu`

### Step 4: View Results

#### 4.1 View Detailed JSON Results

```bash
# View detailed prediction information
cat predictions/predictions.json

```

JSON Format Example:

```json
{
  "csv_file": "genome_multi_k_summary.csv",
  "seq_len": 1000,
  "predictions": [0.05, 0.08, 0.12, 0.89, 0.92, 0.88, 0.15, ...],
  "predicted_regions": [
    {
      "start_bin": 350,
      "end_bin": 450,
      "start_pos": 3500000,
      "end_pos": 4500000,
      "length_bins": 100,
      "length_bp": 1000000,
      "avg_prob": 0.87,
      "max_prob": 0.95
    }
  ]
}

```

#### 4.2 View Summary CSV Results

```bash
# View predicted regions in tabular format
cat predictions/predictions_summary.csv

```

CSV Format Example:

```csv
file,seq_len,num_regions,top_region_start,top_region_end,top_region_prob
genome_multi_k_summary.csv,1000,3,3500000,4500000,0.8700

```

#### 4.3 Extract Predicted Regions to BED Format

Create a conversion script:

```python
# predictions_to_bed.py
import json
import sys

def json_to_bed(json_file, bed_file, min_prob=0.5):
    """Convert prediction results to BED format"""
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    with open(bed_file, 'w') as f:
        # If it is a result for a single chromosome
        if isinstance(results, dict):
            results = [results]
        
        for result in results:
            for i, region in enumerate(result['predicted_regions']):
                if region['avg_prob'] >= min_prob:
                    # BED format: chr start end name score
                    chrom = result.get('chromosome', 'chr1')
                    start = region['start_pos']
                    end = region['end_pos']
                    name = f"centromere_{i+1}"
                    score = int(region['avg_prob'] * 1000)
                    
                    f.write(f"{chrom}\t{start}\t{end}\t{name}\t{score}\n")

if __name__ == '__main__':
    json_to_bed(
        'predictions/predictions.json',
        'predictions/centromeres.bed',
        min_prob=0.5
    )
    print("BED file created: predictions/centromeres.bed")

```

Run the script:

```bash
python predictions_to_bed.py

```

---

## Simplified Workflow (One-click Script)

Integrate all steps into a single script:

```bash
#!/bin/bash
# predict_from_fasta.sh - One-click script from FASTA to prediction results

set -e

# Parameters
GENOME=$1
MODEL=$2
OUTPUT_DIR=${3:-"predictions_output"}
BIN_SIZE=${4:-10000}
THREADS=${5:-8}

if [ -z "$GENOME" ] || [ -z "$MODEL" ]; then
    echo "Usage: $0 <genome.fasta> <model.pt> [output_dir] [bin_size] [threads]"
    echo "Example: $0 genome.fasta best_model.pt results 10000 8"
    exit 1
fi

echo "========================================="
echo "Centromere Prediction Pipeline"
echo "========================================="
echo "Genome: $GENOME"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Bin size: $BIN_SIZE"
echo "Threads: $THREADS"
echo "========================================="

# Create output directory
mkdir -p $OUTPUT_DIR
KMER_DIR="$OUTPUT_DIR/kmer_analysis"
mkdir -p $KMER_DIR

# Step 1: k-mer statistics
echo ""
echo "[Step 1/4] Computing k-mer frequencies..."
for k in 64 128 256 512; do
    echo "  Processing k=$k..."
    jellyfish count -m $k -s 1G -t $THREADS -C \
        -o ${KMER_DIR}/${k}mer.jf $GENOME
    jellyfish dump ${KMER_DIR}/${k}mer.jf \
        > ${KMER_DIR}/${k}mer_counts.txt
    rm ${KMER_DIR}/${k}mer.jf  # Delete intermediate files to save space
done
echo "  k-mer analysis completed!"

# Step 2: Generate features
echo ""
echo "[Step 2/4] Generating feature CSV..."
python generate_features.py \
    --genome $GENOME \
    --kmer-dir $KMER_DIR \
    --output ${OUTPUT_DIR}/features.csv \
    --bin-size $BIN_SIZE
echo "  Feature CSV created!"

# Step 3: Model inference
echo ""
echo "[Step 3/4] Running model inference..."
python src/training/inference.py \
    --checkpoint $MODEL \
    --input ${OUTPUT_DIR}/features.csv \
    --output ${OUTPUT_DIR}/predictions \
    --threshold 0.3
echo "  Inference completed!"

# Step 4: Generate BED file
echo ""
echo "[Step 4/4] Generating BED file..."
python predictions_to_bed.py
echo "  BED file created!"

# Summary
echo ""
echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="
echo "Results:"
echo "  - Feature CSV: ${OUTPUT_DIR}/features.csv"
echo "  - Predictions JSON: ${OUTPUT_DIR}/predictions/predictions.json"
echo "  - Summary CSV: ${OUTPUT_DIR}/predictions/predictions_summary.csv"
echo "  - BED file: ${OUTPUT_DIR}/predictions/centromeres.bed"
echo "========================================="

```

Use the one-click script:

```bash
chmod +x predict_from_fasta.sh
./predict_from_fasta.sh genome.fasta checkpoints/best_model.pt

```

---

## Result Interpretation

### Predicted Region Information

| Field | Description |
| --- | --- |
| **start_pos / end_pos** | Position of the centromere on the genome (bp) |
| **length_bp** | Length of the centromere |
| **avg_prob** | Average predicted probability (0–1) |
| **max_prob** | Highest predicted probability (0–1) |

### Confidence Assessment

* **avg_prob > 0.8**: High confidence; very likely to be a centromere.
* **avg_prob 0.5–0.8**: Medium confidence; likely to be a centromere.
* **avg_prob < 0.5**: Low confidence; potentially a false positive.

### Visualization

View results using IGV or other genome browsers:

1. Load the reference genome into IGV.
2. Load the predicted BED file: `predictions/centromeres.bed`.
3. Inspect the predicted centromere regions.

---

## Full Example

Suppose you have an Arabidopsis genome:

```bash
# 1. Download or prepare the genome
# genome.fasta (already exists)

# 2. Download the pre-trained model
# best_model.pt (already exists)

# 3. Run the complete pipeline
./predict_from_fasta.sh \
    genome.fasta \
    checkpoints/best_model.pt \
    arabidopsis_results

# 4. View results
cat arabidopsis_results/predictions/predictions_summary.csv

# 5. Visualize in IGV
# File -> Load from File -> arabidopsis_results/predictions/centromeres.bed

```

---

## Performance and Resources

### Estimated Calculation Time

For a 200Mb genome:

* **k-mer statistics**: ~10–30 minutes
* **Feature generation**: ~5–10 minutes
* **Model inference**: ~1–5 minutes (GPU) or ~10–30 minutes (CPU)
* **Total**: ~20–60 minutes

### Memory Requirements

* **k-mer statistics**: ~2–8GB (depending on genome size)
* **Feature generation**: ~1–4GB
* **Model inference**: ~2–4GB (GPU) or ~1–2GB (CPU)

### Disk Space

* **k-mer intermediate files**: ~1–5GB (can be deleted after CSV generation)
* **Feature CSV**: ~10–100MB
* **Prediction results**: ~1–10MB

---

## FAQ

* **Q: What if I don't have Jellyfish?**
* **A**: You can use KMC or other k-mer counting tools, but corresponding script sections must be modified.


* **Q: What if the genome is very large (>1GB)?**
1. Increase Jellyfish memory limit (via the `-s` parameter).
2. Process chromosome by chromosome.
3. Use more threads for acceleration.


* **Q: How to batch process multiple genomes?**
* **A**:


```bash
for genome in *.fasta; do
    name=$(basename $genome .fasta)
    ./predict_from_fasta.sh $genome checkpoints/best_model.pt ${name}_results
done

```


* **Q: What if prediction results are unsatisfactory?**
1. Adjust threshold parameters (`--threshold`).
2. Check if the numerical ranges of the feature CSV are reasonable.
3. If working with a new species, retraining the model might be necessary.


* **Q: Can the k-mer statistics step be skipped?**
* **A**: If you already have k-mer statistics results, you can start directly from Step 2, provided that the format compatibility is ensured.



---

## Advanced Usage

### Custom Bin Size

```bash
# Use a 5kb bin
python generate_features.py \
    --genome genome.fasta \
    --kmer-dir kmer_analysis \
    --output features_5kb.csv \
    --bin-size 5000

```

### Adjusting Prediction Threshold

```bash
# More sensitive (higher recall, potentially more false positives)
python inference.py --threshold 0.2 ...

# More conservative (higher precision, may miss some regions)
python inference.py --threshold 0.5 ...

```

### Processing by Chromosome

```bash
# If the genome is very large, process by chromosome
for chr in chr*.fasta; do
    ./predict_from_fasta.sh $chr model.pt ${chr}_results
done

# Merge results
cat */predictions/centromeres.bed > all_centromeres.bed

```

---

## Getting Help

* 📖 View the [Quick Start Guide](QUICKSTART_CN.md)
* 🔧 View the [Data Format Documentation](DATA_FORMAT.md)
* 💬 Submit a [GitHub Issue]( https://github.com/xr092138123-cmyk/Oryza-Genus-Centromere-Project/issues)

---

**Last Updated**: 2024-12-19

---

## 🔗 Project Link

**GitHub**: [ https://github.com/xr092138123-cmyk/Oryza-Genus-Centromere-Project/tree/main/02.Centromere_Region_Identification/03.centromere_prediction_github]

---

你觉得这段翻译怎么样？如果需要，我还可以帮你优化 GitHub 的 README 首页。
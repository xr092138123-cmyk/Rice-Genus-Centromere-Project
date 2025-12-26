# Data Format Specification

This document provides a detailed description of the input data format required by the model.

## Overview

The model receives input files in CSV format, where each file represents a chromosome or a genomic fragment. File names must end with `_multi_k_summary.csv`.

## CSV File Format

### Required Columns

| Column Name | Data Type | Description | Example Value |
| --- | --- | --- | --- |
| `start` | int | Interval start position (base pairs) | 0 |
| `end` | int | Interval end position (base pairs) | 10000 |
| `has_cen` | int | Whether it is a centromere (0 or 1) | 1 |
| `64_highlighted_percent` | float | Highlight percentage at k=64 | 0.85 |
| `64_coverage_depth_avg` | float | Average coverage depth at k=64 | 15.2 |
| `128_highlighted_percent` | float | Highlight percentage at k=128 | 0.82 |
| `128_coverage_depth_avg` | float | Average coverage depth at k=128 | 14.5 |
| `256_highlighted_percent` | float | Highlight percentage at k=256 | 0.78 |
| `256_coverage_depth_avg` | float | Average coverage depth at k=256 | 13.8 |
| `512_highlighted_percent` | float | Highlight percentage at k=512 | 0.75 |
| `512_coverage_depth_avg` | float | Average coverage depth at k=512 | 12.9 |

### Column Descriptions

#### Position Columns

* **start/end**: Define the intervals on the genome.
* Usually equal-length bins (e.g., 10kb).
* Must be continuous and non-overlapping.
* Unit: base pairs (bp).



#### Label Column

* **has_cen**: Centromere label.
* 0: Non-centromere region.
* 1: Centromere region.
* Required for training, optional for inference.



#### Feature Columns

For each k-value (64, 128, 256, 512), there are two features:

1. **highlighted_percent**: Highlight percentage.
* Range: 0.0 to 1.0.
* Represents the proportion of k-mers within the interval marked as "highlighted".
* Centromere regions typically have higher values.


2. **coverage_depth_avg**: Average coverage depth.
* Range: Typically 0 to 100+.
* Represents the average sequencing depth or k-mer count for the interval.
* Centromere regions typically have higher values.



### Example CSV File

```csv
start,end,has_cen,64_highlighted_percent,64_coverage_depth_avg,128_highlighted_percent,128_coverage_depth_avg,256_highlighted_percent,256_coverage_depth_avg,512_highlighted_percent,512_coverage_depth_avg
0,10000,0,0.15,2.3,0.12,1.8,0.10,1.5,0.08,1.2
10000,20000,0,0.18,2.5,0.15,2.1,0.13,1.9,0.10,1.5
20000,30000,1,0.85,15.2,0.82,14.5,0.78,13.8,0.75,12.9
30000,40000,1,0.90,16.5,0.88,15.8,0.85,15.1,0.82,14.3
40000,50000,1,0.87,15.8,0.84,15.2,0.80,14.5,0.77,13.7
50000,60000,0,0.16,2.4,0.13,2.0,0.11,1.7,0.09,1.3

```

## File Naming Convention

### Naming Format

```text
{sample_id}_{chromosome_id}_multi_k_summary.csv

```

### Example File Names

* `sample1_chr1_multi_k_summary.csv`
* `HG002_chr2_multi_k_summary.csv`
* `human_genome_hap1_chr3_multi_k_summary.csv`
* `arabidopsis_chr5_multi_k_summary.csv`

### Naming Requirements

1. Must end with `_multi_k_summary.csv`.
2. It is recommended to include Sample ID and Chromosome ID for easy identification.
3. Avoid using spaces and special characters.
4. Use lowercase letters and underscores.

## Data Organization

### Directory Structure Example

```text
data/
├── training_data/
│   ├── sample1_chr1_multi_k_summary.csv
│   ├── sample1_chr2_multi_k_summary.csv
│   ├── sample2_chr1_multi_k_summary.csv
│   └── sample2_chr2_multi_k_summary.csv
├── test_data/
│   ├── sample3_chr1_multi_k_summary.csv
│   └── sample3_chr2_multi_k_summary.csv
└── validation_data/
    ├── sample4_chr1_multi_k_summary.csv
    └── sample4_chr2_multi_k_summary.csv

```

### Data Partitioning Suggestions

* **Partition by Sample**: Assign data from different individuals to different sets.
* **Partition by Chromosome**: Treat each chromosome as an independent sample.
* **Avoid Data Leakage**: Different chromosomes from the same sample should not appear in both the training and test sets simultaneously.

## Data Quality Requirements

### Basic Requirements

1. **Completeness**: All required columns are present.
2. **Consistency**: Bin sizes within the same file should be identical.
3. **Continuity**: `start` and `end` should be continuous and non-overlapping.
4. **Validity**: Numerical values are within reasonable ranges.

### Data Checklist

* [ ] CSV format is correct and readable by pandas.
* [ ] All required columns are present.
* [ ] `start < end` holds true for all rows.
* [ ] `has_cen` contains only 0 and 1.
* [ ] Feature values are within reasonable ranges (no NaN or Inf).
* [ ] File name ends with `_multi_k_summary.csv`.
* [ ] Each file contains at least 10 rows of data.

### Data Validation Script

```python
import pandas as pd
import numpy as np

def validate_csv(filepath):
    """Validate CSV file format"""
    required_cols = [
        'start', 'end', 'has_cen',
        '64_highlighted_percent', '64_coverage_depth_avg',
        '128_highlighted_percent', '128_coverage_depth_avg',
        '256_highlighted_percent', '256_coverage_depth_avg',
        '512_highlighted_percent', '512_coverage_depth_avg'
    ]
    
    try:
        df = pd.read_csv(filepath)
        
        # Check columns
        for col in required_cols:
            if col not in df.columns:
                return False, f"Missing column: {col}"
        
        # Check start < end
        if not (df['start'] < df['end']).all():
            return False, "start must be < end"
        
        # Check has_cen
        if not df['has_cen'].isin([0, 1]).all():
            return False, "has_cen must be 0 or 1"
        
        # Check NaN
        if df[required_cols].isnull().any().any():
            return False, "Contains NaN values"
        
        return True, "Valid"
    
    except Exception as e:
        return False, str(e)

# Example usage
valid, msg = validate_csv("your_file.csv")
print(f"Validation: {valid}, {msg}")

```

## Data Preprocessing

### Normalization

The model internally performs Z-score normalization automatically:

```python
X_normalized = (X - mean) / std

```

Where `mean` and `std` are calculated from the training set.

### Notes

1. **No manual normalization required**: The model handles this automatically.
2. **Maintain original scale**: Provide raw feature values.
3. **Consistency**: Ensure all files use the same calculation method.

## Data Acquisition

### Generating from BAM Files

If you have raw BAM alignment files, you can use the following tools to generate the required CSV:

```bash
# Pseudocode example
# Actual tools need to be adjusted based on your data processing workflow

# 1. Count k-mer coverage
for k in 64 128 256 512; do
    jellyfish count -m $k -s 100M -t 8 genome.fasta
    jellyfish dump mer_counts.jf > ${k}mer_counts.txt
done

# 2. Calculate statistics for each bin
python calculate_bin_statistics.py \
    --genome genome.fasta \
    --kmers 64 128 256 512 \
    --bin-size 10000 \
    --output chr1_multi_k_summary.csv

```

### Converting from Existing Data

If you have data in other formats, you need to convert it to this format:

```python
import pandas as pd

# Example: Merging from multiple files
df_64 = pd.read_csv('64mer_stats.csv')
df_128 = pd.read_csv('128mer_stats.csv')
df_256 = pd.read_csv('256mer_stats.csv')
df_512 = pd.read_csv('512mer_stats.csv')

# Merge
result = pd.DataFrame({
    'start': df_64['start'],
    'end': df_64['end'],
    'has_cen': df_64['is_centromere'],
    '64_highlighted_percent': df_64['highlight_pct'],
    '64_coverage_depth_avg': df_64['coverage'],
    '128_highlighted_percent': df_128['highlight_pct'],
    '128_coverage_depth_avg': df_128['coverage'],
    '256_highlighted_percent': df_256['highlight_pct'],
    '256_coverage_depth_avg': df_256['coverage'],
    '512_highlighted_percent': df_512['highlight_pct'],
    '512_coverage_depth_avg': df_512['coverage'],
})

result.to_csv('chr1_multi_k_summary.csv', index=False)

```

## FAQ

### Q: Must the bin size be 10kb?

A: No. It can be any size, but it is recommended:

* Not too small (<1kb): Features become unstable.
* Not too large (>100kb): Insufficient resolution.
* 10kb is the recommended value, balancing performance and resolution.

### Q: Can bin sizes differ between chromosomes?

A: Yes, but it is recommended to keep them consistent so the model can learn a unified pattern.

### Q: What if feature values have different scales?

A: The model will automatically normalize them, so there is no need to worry. However, ensure that the same feature is calculated using the same method across all files.

### Q: What if data for some k-values is missing?

A: All 8 feature columns are required. If certain k-values are missing, they need to be supplemented or replaced with other k-values.

### Q: Is the `has_cen` column required during inference?

A: No, but if provided, the model will calculate evaluation metrics. You can fill it with -1 to indicate it is unknown.

### Q: Can other k-values be used?

A: The current model was trained using 64/128/256/512. To use other k-values, you would need to:

1. Modify the data format.
2. Modify the model input dimensions.
3. Retrain the model.

## Example Datasets

Example data is provided in the project repository (if available):

```text
examples/data/
├── sample_chr1_multi_k_summary.csv
├── sample_chr2_multi_k_summary.csv
└── README.md

```

## Contact and Support

If you have questions regarding the data format:

* Check the project documentation.
* Submit a GitHub Issue.
* Email the maintainers.

---

**Last Updated**: 2024-12-19

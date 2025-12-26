# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-19

### Added

* Initial version release
* Transformer-based centromere prediction model
* Support for multi-scale k-mer features (64, 128, 256, 512)
* Complete training script (`train.py`)
* Inference script (`inference.py`)
* Data loading module (`dataset.py`)
* Model configuration system (`config.py`)
* Evaluation toolkit:
* Top-5 prediction evaluation (`evaluate_top5_prediction.py`)
* Summary report generation (`generate_summary_report.py`)
* Prediction comparison (`compare_predictions.py`)
* Average metrics calculation (`calculate_average_metrics.py`)
* Coverage statistics processing (`process_coverage_statistics.py`)


* TensorBoard training monitoring
* Detailed documentation:
* README.md (Project description)
* QUICKSTART_CN.md (Quick start guide)
* MODEL_ARCHITECTURE.md (Model architecture document)
* CONTRIBUTING.md (Contribution guide)


* Example scripts and configuration files
* MIT open source license

### Features

* **Transformer Encoder Architecture**: 4-layer Transformer with 8 attention heads
* **Multi-scale Feature Fusion**: Integrating statistical information from different k-values
* **Weighted Loss Function**: Handling class imbalance issues
* **Automatic Threshold Selection**: Searching for the optimal classification threshold on the validation set
* **Early Stopping Mechanism**: Preventing overfitting
* **Learning Rate Scheduling**: Adaptive learning rate adjustment
* **GPU Acceleration**: Supporting CUDA accelerated training and inference
* **Batch Prediction**: Supporting directory-level batch inference
* **Detailed Output**: Prediction results in JSON and CSV formats
* **Visualization Support**: Generating prediction probability curves and comparison plots

### Performance

* Typical F1 Score: 0.82-0.93
* Typical IoU: 0.70-0.88
* Training Speed: ~100-1000 epochs/hour (depending on data volume and hardware)
* Inference Speed: ~10ms/1000bins (GPU)

### Tech Stack

* Python 3.8+
* PyTorch 1.10+
* NumPy, Pandas, Scikit-learn
* Matplotlib, Seaborn
* TensorBoard

---

## [Unreleased]

### Planned

* [ ] Model visualization tool (`visualize.py`)
* [ ] More pre-trained models
* [ ] Multi-GPU training support
* [ ] Model quantization and acceleration
* [ ] Interactive Web interface
* [ ] Docker containerization deployment
* [ ] Dataset support for more species
* [ ] Model Ensemble functionality
* [ ] Automatic hyperparameter search
* [ ] Incremental learning support

### Improvements

* [ ] Optimize memory usage
* [ ] Accelerate long sequence inference
* [ ] Improve data augmentation strategies
* [ ] Better imbalance handling methods
* [ ] Multi-task learning framework
* [ ] Transfer learning support

### Known Issues

* Extremely long sequences (>50000 bins) may cause out-of-memory issues
* Highly imbalanced data (<0.5% positive samples) may require special tuning
* Training instability may occur in certain scenarios

---

## Versioning

### Semantic Versioning Specification

* **Major**: Incompatible API changes
* **Minor**: Backward-compatible functional additions
* **Patch**: Backward-compatible bug fixes

### Labels

* `Added`: New features
* `Changed`: Changes in existing functionality
* `Deprecated`: Soon-to-be deprecated features
* `Removed`: Deleted features
* `Fixed`: Bug fixes
* `Security`: Security improvements

---

## Contributors

Thanks to all developers who contributed to the project!

---

## Links

* [Project Homepage](https://www.google.com/search?q=https://github.com/xr092138123-cmyk/Oryza-Genus-Centromere-Project)
* [Issue Tracker](https://www.google.com/search?q=https://github.com/xr092138123-cmyk/Oryza-Genus-Centromere-Project/issues)
* [Releases](https://www.google.com/search?q=https://github.com/xr092138123-cmyk/Oryza-Genus-Centromere-Project/releases)
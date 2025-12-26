这是为您翻译的英文版 **CONTRIBUTING.md**。我严格遵循了您的要求，仅进行文本翻译，完整保留了所有 Markdown 格式、代码块、目录结构和逻辑顺序。

---

# Contributing Guide

First of all, thank you for considering contributing to this project!

## How to Contribute

### Reporting Bugs

If you find a bug, please create an Issue and include the following information:

* Detailed description of the bug
* Steps to reproduce
* Expected behavior vs. actual behavior
* System environment (OS, Python version, PyTorch version, etc.)
* If possible, provide a minimal reproducible example

### Proposing New Features

If you have ideas for new features, please:

1. Create an Issue first to discuss the feature
2. Explain the purpose and expected effects of the feature
3. Wait for feedback from maintainers before starting the implementation

### Submitting Code

1. **Fork the Repository**
```bash
# Click the Fork button on GitHub
# Then clone your fork
git clone https://github.com/xr092138123-cmyk/Oryza-Genus-Centromere-Project/tree/main/02.Centromere_Region_Identification/03.centromere_prediction_github
cd centromere_prediction

```


2. **Create a Branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix

```


3. **Make Changes**
* Follow the existing code style
* Add necessary comments and docstrings
* If possible, add tests


4. **Test Your Changes**
```bash
# Run basic tests
python -m pytest tests/

```


5. **Commit Changes**
```bash
git add .
git commit -m "feat: add new feature X"
# or
git commit -m "fix: resolve issue with Y"

```


Commit message format:
* `feat:` New features
* `fix:` Bug fixes
* `docs:` Documentation updates
* `style:` Code formatting (non-functional changes)
* `refactor:` Code refactoring
* `test:` Adding tests
* `chore:` Build/tooling updates


6. **Push to GitHub**
```bash
git push origin feature/your-feature-name

```


7. **Create a Pull Request**
* Visit your GitHub repository
* Click "New Pull Request"
* Fill in the PR description and explain your changes
* Wait for code review



## Code Style

### Python Code Standards

* Follow the PEP 8 specification
* Use 4 spaces for indentation
* Maximum 100 characters per line
* Use meaningful variable and function names
* Add docstrings to all public functions

Example:

```python
def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calculate evaluation metrics
    
    Args:
        predictions: Array of predicted values, shape (n_samples,)
        labels: Array of ground truth labels, shape (n_samples,)
    
    Returns:
        A dictionary containing various metrics
    """
    # Implementation code...
    return metrics

```

### Documentation Standards

* Add documentation for new features
* Update relevant README or documentation files
* Use clear English and Chinese descriptions
* Provide usage examples

## Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/xr092138123-cmyk/Oryza-Genus-Centromere-Project/tree/main/02.Centromere_Region_Identification/03.centromere_prediction_github
cd centromere_prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -e .  # Install in editable mode

# Install development tools (optional)
pip install pytest black flake8 mypy

```

## Testing

Before submitting a PR, please ensure:

```bash
# Run code format check
black src/ --check
flake8 src/

# Run type check
mypy src/

# Run tests
pytest tests/

```

## Code Review Process

1. After submitting a PR, maintainers will conduct a code review.
2. Changes may be requested.
3. Once all discussions are resolved, the PR will be merged.
4. Your contribution will be recorded in the project history.

## Code of Conduct

### Our Commitment

To foster an open and friendly environment, we as contributors and maintainers commit to:

* Respecting everyone
* Accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

### Unacceptable Behavior

* Use of sexualized language or imagery
* Personal or political attacks
* Public or private harassment
* Publishing others' private information without permission
* Other unprofessional or unwelcome behavior

## Questions and Discussions

* For bug reports and feature requests, please use GitHub Issues.
* For general discussions, you can use GitHub Discussions.
* For urgent matters, you can email the maintainers.

## License

By contributing code, you agree that your contributions will be released under the MIT License.

## Contact Information

* GitHub Issues: [Project Issues Page]
* Email: your.email@example.com

## Acknowledgments

Thank you to all contributors for your support and help!

Your name will be added to the contributors list.
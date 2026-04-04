# Contributing to deep-ml

Thank you for your interest in contributing to deep-ml! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Poetry for dependency management
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-sagar100rathod/deep-ml.git
   cd deep-ml
   ```

2. **Install Poetry** (if not already installed)
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**
   ```bash
   poetry install
   ```

4. **Install pre-commit hooks**
   ```bash
   poetry run pre-commit install
   ```

5. **Verify installation**
   ```bash
   poetry run pytest tests/
   ```

## 🔨 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style
- Add docstrings to functions and classes
- Update documentation if needed

### 3. Format Your Code

```bash
# Auto-format code
make format

# Or manually
poetry run black deepml tests
poetry run isort deepml tests
```

### 4. Run Tests

```bash
# Run all tests
make test

# Run specific test file
poetry run pytest tests/test_tasks.py

# Run with coverage
poetry run pytest --cov=deepml --cov-report=html
```

### 5. Check Code Quality

```bash
# Run all checks
make check

# Or individually
poetry run black --check deepml tests
poetry run isort --check-only deepml tests
poetry run pre-commit run --all-files
```

### 6. Commit Your Changes

Use semantic commit messages:

```bash
git commit -m "feat: add support for custom loss functions"
git commit -m "fix: resolve memory leak in data loader"
git commit -m "docs: update installation instructions"
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a PR on GitHub with:
- Clear title following semantic conventions
- Description of changes
- Reference to related issues (if any)
- Screenshots (if applicable)

## 📝 Code Style Guidelines

### Python Style

- Follow PEP 8
- Use Black formatter (line length: 88)
- Use isort for import sorting
- Add type hints where possible

### Docstring Format

Use Google-style docstrings:

```python
def train_model(
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int = 10
) -> Dict[str, float]:
    """Train a PyTorch model on the given data.

    Args:
        model: PyTorch model to train.
        loader: DataLoader containing training data.
        epochs: Number of training epochs. Defaults to 10.

    Returns:
        Dictionary containing training metrics (loss, accuracy, etc.).

    Raises:
        ValueError: If loader is empty.

    Example:
        >>> model = MyModel()
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> metrics = train_model(model, loader, epochs=5)
    """
    pass
```

### File Organization

```python
# Standard library imports
import os
from typing import Dict, List

# Third-party imports
import numpy as np
import torch

# Local imports
from deepml.base import BaseTrainer
from deepml.utils import setup_logger
```

## 🧪 Testing Guidelines

### Writing Tests

1. **Test file naming**: `test_<module_name>.py`
2. **Test function naming**: `test_<functionality>`
3. **Use fixtures for reusable setup**
4. **Test both success and failure cases**

Example:

```python
import pytest
import torch
from deepml.tasks import ImageClassification

@pytest.fixture
def sample_model():
    """Create a simple model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2)
    )

def test_image_classification_initialization(sample_model, tmp_path):
    """Test that ImageClassification initializes correctly."""
    task = ImageClassification(
        model=sample_model,
        model_dir=str(tmp_path),
        classes=['cat', 'dog']
    )
    assert task.model is not None
    assert len(task._classes) == 2

def test_image_classification_invalid_model(tmp_path):
    """Test that initialization fails with invalid model."""
    with pytest.raises(AssertionError):
        ImageClassification(
            model="not_a_model",
            model_dir=str(tmp_path)
        )
```

### Running Specific Tests

```bash
# Run specific test file
poetry run pytest tests/test_tasks.py

# Run specific test function
poetry run pytest tests/test_tasks.py::test_image_classification_initialization

# Run with verbose output
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=deepml --cov-report=term-missing
```

## 📚 Documentation

### Updating Documentation

1. **API Documentation**: Auto-generated from docstrings
   ```bash
   make docs-api
   ```

2. **Build HTML documentation**
   ```bash
   make docs-build
   ```

3. **View documentation locally**
   ```bash
   open docs/build/html/index.html
   ```

### Documentation Structure

- **docs/source/**: Sphinx documentation source
- **API docs**: Auto-generated from code docstrings
- **Examples**: Add Jupyter notebooks to `notebooks/`

## 🐛 Reporting Issues

### Bug Reports

Include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Code snippet (if applicable)
- Error messages/stack traces

### Feature Requests

Include:
- Clear description of the feature
- Use case/motivation
- Example usage (pseudo-code)
- Potential implementation ideas

## 🔍 Code Review Process

1. **Automated checks must pass**
   - CI tests
   - Code formatting
   - Code quality checks

2. **Manual review by maintainer**
   - Code quality
   - Test coverage
   - Documentation updates

3. **Address feedback**
   - Make requested changes
   - Push updates to your branch

4. **Merge**
   - Maintainer merges PR
   - Delete feature branch

## 📦 Release Process

### For Maintainers

1. **Update version** in `pyproject.toml`
   ```bash
   poetry version patch  # or minor, major
   ```

2. **Update CHANGELOG.md**
   ```markdown
   ## [2.0.2] - 2026-04-04
   ### Added
   - New feature X

   ### Fixed
   - Bug Y

   ### Changed
   - Updated Z
   ```

3. **Commit and tag**
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: bump version to 2.0.2"
   git tag -a v2.0.2 -m "Release version 2.0.2"
   git push origin main --tags
   ```

4. **GitHub Actions automatically**:
   - Creates release
   - Builds package
   - Publishes to PyPI

## 🙋 Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open an issue
- **Feature requests**: Open an issue with "enhancement" label
- **Security issues**: Email sagar100rathod@gmail.com

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🎉 Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes
- Special mentions in README (for significant contributions)

Thank you for contributing to deep-ml! 🚀

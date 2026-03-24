Contributing
============

We welcome contributions to deep-ml! This guide will help you get started.

Getting Started
---------------

Fork and Clone
~~~~~~~~~~~~~~

.. code-block:: bash

   # Fork the repository on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/deep-ml.git
   cd deep-ml

Setup Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install in development mode
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

Code Style
----------

We follow PEP 8 and use these tools:

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Format Code
~~~~~~~~~~~

.. code-block:: bash

   # Format with black
   black deepml/

   # Sort imports
   isort deepml/

   # Check linting
   flake8 deepml/

   # Type check
   mypy deepml/

Docstrings
~~~~~~~~~~

Use **Google-style docstrings**:

.. code-block:: python

   def my_function(arg1, arg2):
       """One-line summary.

       Extended description if needed.

       Args:
           arg1: Description of arg1.
           arg2: Description of arg2.

       Returns:
           Description of return value.

       Raises:
           ValueError: When invalid argument.

       Example:
           >>> my_function(1, 2)
           3
       """
       return arg1 + arg2

Testing
-------

Run Tests
~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest

   # Run specific test file
   pytest tests/test_fabric_trainer.py

   # Run with coverage
   pytest --cov=deepml --cov-report=html

   # View coverage report
   open htmlcov/index.html

Write Tests
~~~~~~~~~~~

Use pytest for all tests:

.. code-block:: python

   import pytest
   import torch
   from deepml.tasks import ImageClassification

   @pytest.fixture
   def simple_model():
       return torch.nn.Linear(10, 2)

   def test_image_classification_init(simple_model):
       """Test ImageClassification initialization."""
       task = ImageClassification(
           model=simple_model,
           model_dir='./temp'
       )

       assert task._model is simple_model
       assert task._device is not None

   def test_prediction_shape(simple_model):
       """Test prediction output shape."""
       task = ImageClassification(
           model=simple_model,
           model_dir='./temp'
       )

       batch = torch.randn(4, 10)
       output = task.predict_batch(batch)

       assert output.shape == (4, 2)

Test Coverage
~~~~~~~~~~~~~

Aim for >80% test coverage for new code.

Pull Request Process
--------------------

1. Create Feature Branch
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git checkout -b feature/my-new-feature

2. Make Changes
~~~~~~~~~~~~~~~

- Write code
- Add tests
- Update documentation
- Follow code style guidelines

3. Commit Changes
~~~~~~~~~~~~~~~~~

Use conventional commits:

.. code-block:: bash

   # Feature
   git commit -m "feat: add support for custom callbacks"

   # Bug fix
   git commit -m "fix: resolve gradient accumulation issue"

   # Documentation
   git commit -m "docs: update training guide"

   # Tests
   git commit -m "test: add tests for AcceleratorTrainer"

4. Push and Create PR
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git push origin feature/my-new-feature

Then create a pull request on GitHub.

PR Checklist
~~~~~~~~~~~~

- [ ] Code follows style guidelines (black, isort, flake8)
- [ ] All tests pass
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] Added docstrings to new functions/classes
- [ ] Updated CHANGELOG.md
- [ ] PR description explains changes clearly

Areas to Contribute
-------------------

Bug Fixes
~~~~~~~~~

- Check open issues labeled "bug"
- Fix and submit PR with test

New Features
~~~~~~~~~~~~

- Propose feature in an issue first
- Implement with tests and documentation
- Submit PR

Documentation
~~~~~~~~~~~~~

- Fix typos
- Add examples
- Improve clarity
- Add tutorials

Tests
~~~~~

- Increase test coverage
- Add edge case tests
- Test on different platforms

Examples
~~~~~~~~

- Add Jupyter notebooks
- Create example scripts
- Share use cases

Code Review
-----------

What We Look For
~~~~~~~~~~~~~~~~

1. **Correctness**: Does it work?
2. **Tests**: Are there tests?
3. **Style**: Follows guidelines?
4. **Documentation**: Clear docstrings?
5. **Backward compatibility**: Breaking changes?

Review Process
~~~~~~~~~~~~~~

1. Automated checks run (tests, linting)
2. Maintainers review code
3. Discussion and requested changes
4. Approval and merge

Release Process
---------------

Versioning
~~~~~~~~~~

We follow semantic versioning (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Release Checklist
~~~~~~~~~~~~~~~~~

1. Update version in ``pyproject.toml``
2. Update ``CHANGELOG.md``
3. Create git tag
4. Build and publish to PyPI
5. Create GitHub release

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

Communication
~~~~~~~~~~~~~

- Use GitHub issues for bugs and features
- Use pull requests for code contributions
- Be clear and concise
- Provide context and examples

Recognition
-----------

Contributors are:

- Listed in ``CONTRIBUTORS.md``
- Mentioned in release notes
- Credited in documentation

Thank you for contributing to deep-ml! 🎉

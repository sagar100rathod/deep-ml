# GitHub Workflows Documentation

This directory contains automated CI/CD workflows for the deep-ml project.

## 📋 Workflows Overview

### 1. **CI (Continuous Integration)** - `python-ci.yml`
[![CI](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/python-ci.yml/badge.svg)](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/python-ci.yml)

**Triggers:** Push/PR to `main` or `develop` branches

**Jobs:**
- **Lint**: Code formatting checks with Black, isort, and pre-commit
- **Test**: Multi-OS (Ubuntu, macOS, Windows) and multi-Python version (3.11, 3.12) testing
- **Build**: Package building with Poetry
- **Verify**: Package installation verification

**Features:**
- ✅ Dependency caching for faster builds
- ✅ Code coverage reporting (Codecov)
- ✅ Parallel testing across multiple environments
- ✅ Build artifact retention

---

### 2. **Publish to PyPI** - `python-publish.yml`
[![Publish](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/python-publish.yml/badge.svg)](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/python-publish.yml)

**Triggers:**
- Release published
- Manual dispatch (with PyPI/TestPyPI selection)

**Jobs:**
- Build and test package
- Publish to PyPI or TestPyPI
- Upload distribution artifacts

**Required Secrets:**
- `PYPI_API_TOKEN` - PyPI API token
- `TEST_PYPI_API_TOKEN` - Test PyPI API token

**Usage:**
```bash
# Create a new release via GitHub UI or:
gh release create v2.0.2 --generate-notes

# Or manually trigger for testing:
gh workflow run python-publish.yml -f repository=testpypi
```

---

### 3. **Documentation** - `docs.yml`
[![Docs](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/docs.yml/badge.svg)](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/docs.yml)

**Triggers:**
- Push to `main` (docs/** or deepml/** changes)
- Pull requests
- Manual dispatch

**Jobs:**
- **build-docs**: Generate API docs and build HTML
- **deploy-docs**: Deploy to GitHub Pages (on push to main)

**Deployment:**
- Docs available at: `https://your-sagar100rathod.github.io/deep-ml/`

---

### 4. **Release Management** - `release.yml`
[![Release](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/release.yml/badge.svg)](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/release.yml)

**Triggers:**
- Push tags matching `v*.*.*` pattern
- Manual dispatch with version input

**Jobs:**
- Version bumping in pyproject.toml
- Run tests
- Build package
- Generate changelog
- Create GitHub release with artifacts

**Usage:**
```bash
# Create and push a new tag
git tag -a v2.0.2 -m "Release version 2.0.2"
git push origin v2.0.2

# Or use manual dispatch
gh workflow run release.yml -f version=2.0.2
```

---

### 5. **Code Quality & Security** - `code-quality.yml`
[![Code Quality](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/code-quality.yml/badge.svg)](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/code-quality.yml)

**Triggers:**
- Push/PR to `main` or `develop`
- Weekly schedule (Mondays)
- Manual dispatch

**Jobs:**
- **code-quality**: Pylint, Flake8, MyPy, Bandit, Safety checks
- **codeql-analysis**: GitHub CodeQL security scanning

**Features:**
- Static code analysis
- Security vulnerability scanning
- Type checking
- Dependency vulnerability checks

---

### 6. **PR Automation** - `pr-automation.yml`
[![PR Automation](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/pr-automation.yml/badge.svg)](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/pr-automation.yml)

**Triggers:** PR opened/edited, Issues opened/edited

**Jobs:**
- **auto-label-pr**: Auto-label based on changed files and PR size
- **validate-pr**: Validate PR title follows semantic conventions
- **greet-contributor**: Welcome first-time contributors

**PR Title Format:**
```
<type>: <description>

Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
Example: feat: Add new segmentation task support
```

---

### 7. **Dependabot** - `dependabot.yml`

**Schedule:** Weekly on Mondays

**Updates:**
- GitHub Actions dependencies
- Python dependencies (via Poetry)

**Features:**
- Grouped patch/minor updates
- Automated PR creation
- Automatic labels

---

## 🚀 Quick Start

### Setup Required Secrets

Go to **Settings → Secrets and Variables → Actions** and add:

1. `PYPI_API_TOKEN` - For publishing to PyPI
   - Get from: https://pypi.org/manage/account/token/

2. `TEST_PYPI_API_TOKEN` - For testing publishing
   - Get from: https://test.pypi.org/manage/account/token/

### Enable GitHub Pages

1. Go to **Settings → Pages**
2. Source: **GitHub Actions**
3. Docs will be available at: `https://your-sagar100rathod.github.io/deep-ml/`

### Enable CodeQL

1. Go to **Settings → Code security and analysis**
2. Enable **CodeQL analysis**

---

## 🔄 CI/CD Pipeline Flow

```
┌─────────────────┐
│   Code Push/PR  │
└────────┬────────┘
         │
         ├─────────────────────────────────────────┐
         │                                         │
         ▼                                         ▼
┌────────────────┐                    ┌──────────────────┐
│  Code Quality  │                    │    CI Build      │
│  & Security    │                    │  (Multi-OS/Py)   │
└────────────────┘                    └────────┬─────────┘
                                               │
                                               ▼
                                      ┌────────────────┐
                                      │  Tests Pass?   │
                                      └────────┬───────┘
                                               │ Yes
                                               ▼
                                      ┌────────────────┐
                                      │  Build Docs    │
                                      └────────┬───────┘
                                               │
                     ┌─────────────────────────┴──────────┐
                     │                                    │
                     ▼                                    ▼
            ┌────────────────┐                  ┌─────────────────┐
            │ Deploy to GH   │                  │   Tag Release   │
            │     Pages      │                  └────────┬────────┘
            └────────────────┘                           │
                                                         ▼
                                              ┌──────────────────┐
                                              │ Create Release & │
                                              │  Publish to PyPI │
                                              └──────────────────┘
```

---

## 🎯 Best Practices

### For Contributors

1. **Fork and clone** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make changes** and ensure tests pass locally: `make test`
4. **Format code**: `make format`
5. **Commit with semantic messages**: `git commit -m "feat: add new feature"`
6. **Push and create PR**

### For Maintainers

#### Creating a Release

1. **Update version** in `pyproject.toml`
2. **Commit changes**: `git commit -m "chore: bump version to X.Y.Z"`
3. **Create and push tag**:
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin vX.Y.Z
   ```
4. **Workflow automatically**:
   - Creates GitHub release
   - Builds package
   - Publishes to PyPI

#### Manual PyPI Testing

```bash
# Test on TestPyPI first
gh workflow run python-publish.yml -f repository=testpypi

# Verify installation
pip install -i https://test.pypi.org/simple/ deepml

# If all good, publish to PyPI
gh workflow run python-publish.yml -f repository=pypi
```

---

## 🛠️ Maintenance

### Updating Workflows

1. Edit workflow files in `.github/workflows/`
2. Test using `act` (local GitHub Actions runner):
   ```bash
   brew install act
   act -l  # List jobs
   act -j test  # Run specific job
   ```

### Monitoring

- **Actions Tab**: Monitor workflow runs
- **Security Tab**: Review security alerts
- **Insights → Dependency graph**: View dependencies

---

## 📊 Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/python-ci.yml/badge.svg)](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/python-ci.yml)
[![Publish](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/python-publish.yml/badge.svg)](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/python-publish.yml)
[![Docs](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/docs.yml/badge.svg)](https://github.com/your-sagar100rathod/deep-ml/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/your-sagar100rathod/deep-ml/branch/main/graph/badge.svg)](https://codecov.io/gh/your-sagar100rathod/deep-ml)
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## 📄 License

MIT License - see [LICENSE](../LICENSE) for details.

# 🚀 Deep-ML CI/CD Pipeline - Complete Setup

## 📊 Overview

This document provides a complete overview of the CI/CD infrastructure for the deep-ml project.

---

## 🗂️ File Structure

```
.github/
├── workflows/
│   ├── python-ci.yml              # Main CI workflow
│   ├── python-publish.yml         # PyPI publishing
│   ├── docs.yml                   # Documentation deployment
│   ├── release.yml                # Release automation
│   ├── code-quality.yml           # Code quality & security
│   ├── pr-automation.yml          # PR/Issue automation
│   └── README.md                  # Workflows documentation
├── ISSUE_TEMPLATE/
│   ├── bug_report.md              # Bug report template
│   └── feature_request.md         # Feature request template
├── pull_request_template.md       # PR template
├── dependabot.yml                 # Dependency updates
├── labeler.yml                    # Auto-labeling config
└── FUNDING.yml                    # Existing funding info
```

---

## 🔄 CI/CD Pipeline Visualization

```
                                    ┌──────────────────┐
                                    │   Developer      │
                                    │   Push/PR Code   │
                                    └────────┬─────────┘
                                             │
                ┌────────────────────────────┼────────────────────────────┐
                │                            │                            │
                ▼                            ▼                            ▼
    ┌─────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
    │  PR Automation      │    │   Code Quality &     │    │    CI Pipeline       │
    │  - Auto-label       │    │   Security Scan      │    │  - Lint (Black,      │
    │  - Size labeling    │    │  - Pylint, Flake8    │    │    isort)            │
    │  - Title validation │    │  - MyPy, Bandit      │    │  - Test (multi-OS,   │
    │  - First-time greet │    │  - Safety, CodeQL    │    │    multi-Python)     │
    └─────────────────────┘    └──────────────────────┘    │  - Build package     │
                                                           │  - Verify install    │
                                                           └──────────┬───────────┘
                                                                      │
                                                           ┌──────────▼───────────┐
                                                           │   All Checks Pass?   │
                                                           └──────────┬───────────┘
                                                                      │ YES
                                      ┌───────────────────────────────┼───────────────────────────────┐
                                      │                               │                               │
                                      ▼                               ▼                               ▼
                          ┌───────────────────┐         ┌──────────────────────┐       ┌──────────────────────┐
                          │  Documentation    │         │   Merge to Main      │       │   Create Tag/        │
                          │  - Generate API   │         │   - Auto-deploy docs │       │   Release            │
                          │  - Build HTML     │         │   - Update coverage  │       └──────────┬───────────┘
                          │  - Deploy to GH   │         └──────────────────────┘                  │
                          │    Pages          │                                                   │
                          └───────────────────┘                                                   ▼
                                                                                    ┌──────────────────────┐
                                                                                    │  Release Workflow    │
                                                                                    │  - Bump version      │
                                                                                    │  - Generate changelog│
                                                                                    │  - Create GH release │
                                                                                    │  - Build artifacts   │
                                                                                    └──────────┬───────────┘
                                                                                               │
                                                                                               ▼
                                                                                    ┌──────────────────────┐
                                                                                    │  Publish to PyPI     │
                                                                                    │  - Test on TestPyPI  │
                                                                                    │  - Publish to PyPI   │
                                                                                    │  - Upload artifacts  │
                                                                                    └──────────────────────┘

                          ┌───────────────────────────────────────────────────────────────┐
                          │                    Background Tasks                           │
                          ├───────────────────────────────────────────────────────────────┤
                          │  • Dependabot: Weekly dependency updates (Mondays)            │
                          │  • Code Quality: Weekly security scans (Mondays)              │
                          │  • CodeQL: Continuous security analysis                       │
                          └───────────────────────────────────────────────────────────────┘
```

---

## 🎯 Workflow Details

### 1. **CI Pipeline** (`python-ci.yml`)

**Purpose:** Continuous Integration for code validation

**Triggers:**
- Push to `main` or `develop`
- Pull requests to `main` or `develop`
- Manual dispatch

**Jobs:**
```yaml
Jobs:
  1. lint (Ubuntu, Python 3.12)
     - Black formatting check
     - isort import sorting check
     - Pre-commit hooks

  2. test (Matrix: Ubuntu/macOS/Windows × Python 3.11/3.12)
     - Install dependencies
     - Run pytest
     - Generate coverage (Ubuntu + Python 3.12 only)
     - Upload to Codecov

  3. build (requires: lint, test)
     - Build wheel and sdist
     - Upload artifacts

  4. verify-package (requires: build)
     - Install built package
     - Verify imports work
```

**Features:**
- ✅ Dependency caching (Poetry)
- ✅ Multi-OS testing
- ✅ Multi-Python version testing
- ✅ Code coverage tracking
- ✅ Artifact retention (7 days)

---

### 2. **Publish Pipeline** (`python-publish.yml`)

**Purpose:** Automated package publishing to PyPI

**Triggers:**
- GitHub release published
- Manual dispatch (with TestPyPI/PyPI selection)

**Jobs:**
```yaml
Jobs:
  publish:
    - Run full test suite
    - Build package
    - Validate with twine
    - Publish to TestPyPI or PyPI
    - Upload artifacts (30 days retention)
```

**Required Secrets:**
- `PYPI_API_TOKEN`
- `TEST_PYPI_API_TOKEN`

**Usage:**
```bash
# Publish to TestPyPI (manual)
gh workflow run python-publish.yml -f repository=testpypi

# Publish to PyPI (automatic on release)
gh release create v2.0.2 --generate-notes
```

---

### 3. **Documentation** (`docs.yml`)

**Purpose:** Build and deploy Sphinx documentation

**Triggers:**
- Push to `main` (docs/** or deepml/** changes)
- Pull requests
- Manual dispatch

**Jobs:**
```yaml
Jobs:
  1. build-docs:
     - Generate API docs (sphinx-apidoc)
     - Build HTML (Sphinx)
     - Upload artifacts

  2. deploy-docs (only on push to main):
     - Download artifacts
     - Deploy to GitHub Pages
```

**Output:** https://sagar100rathod.github.io/deep-ml/

---

### 4. **Release Management** (`release.yml`)

**Purpose:** Automated release creation and tagging

**Triggers:**
- Tag push matching `v*.*.*`
- Manual dispatch with version input

**Jobs:**
```yaml
Jobs:
  create-release:
    - Get version from tag/input
    - Update pyproject.toml
    - Run tests
    - Build package
    - Generate changelog (from git commits)
    - Create GitHub release
    - Attach build artifacts
```

**Usage:**
```bash
# Via git tag
git tag -a v2.0.2 -m "Release 2.0.2"
git push origin v2.0.2

# Via manual dispatch
gh workflow run release.yml -f version=2.0.2
```

---

### 5. **Code Quality & Security** (`code-quality.yml`)

**Purpose:** Static analysis and security scanning

**Triggers:**
- Push/PR to `main` or `develop`
- Weekly schedule (Mondays 00:00 UTC)
- Manual dispatch

**Jobs:**
```yaml
Jobs:
  1. code-quality:
     - Pylint (code quality)
     - Flake8 (style guide)
     - MyPy (type checking)
     - Bandit (security issues)
     - Safety (dependency vulnerabilities)

  2. codeql-analysis:
     - GitHub CodeQL scanning
     - Security vulnerability detection
```

---

### 6. **PR Automation** (`pr-automation.yml`)

**Purpose:** Automated PR and issue management

**Triggers:**
- PR opened/edited
- Issue opened/edited

**Jobs:**
```yaml
Jobs:
  1. auto-label-pr:
     - Label by changed files (via labeler.yml)
     - Label by PR size (XS/S/M/L/XL)

  2. validate-pr:
     - Validate semantic PR title format
     - Enforce commit conventions

  3. greet-contributor:
     - Welcome first-time contributors
     - Provide helpful guidelines
```

**PR Title Format:**
```
<type>: <description>

Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
Example: feat: Add support for multi-label classification
```

---

### 7. **Dependabot** (`dependabot.yml`)

**Purpose:** Automated dependency updates

**Schedule:** Weekly on Mondays

**Configuration:**
```yaml
Updates:
  - GitHub Actions (weekly)
  - Python dependencies via pip (weekly)

Settings:
  - Max 5 PRs for Actions
  - Max 10 PRs for Python deps
  - Group patch/minor updates
  - Auto-label PRs
  - Ignore major version updates
```

---

## 🔐 Required Setup

### 1. GitHub Secrets

Navigate to: **Settings → Secrets and Variables → Actions**

Add the following secrets:

| Secret Name | Description | How to Get |
|------------|-------------|------------|
| `PYPI_API_TOKEN` | PyPI publishing token | https://pypi.org/manage/account/token/ |
| `TEST_PYPI_API_TOKEN` | Test PyPI token | https://test.pypi.org/manage/account/token/ |

### 2. GitHub Pages

1. Go to **Settings → Pages**
2. Source: Select **GitHub Actions**
3. Wait for first docs deployment
4. Access docs at: `https://sagar100rathod.github.io/deep-ml/`

### 3. CodeQL (Optional but Recommended)

1. Go to **Settings → Code security and analysis**
2. Enable **CodeQL analysis**
3. GitHub will automatically scan for vulnerabilities

### 4. Branch Protection

**Status:** Branch protection is already enabled for `main` branch requiring pull requests before merge.

**Recommended status checks to require:**
- `lint` - Code formatting validation
- `test (ubuntu-latest, 3.11)` - Ubuntu Python 3.11 tests
- `test (ubuntu-latest, 3.12)` - Ubuntu Python 3.12 tests
- `test (macos-latest, 3.11)` - macOS Python 3.11 tests
- `test (macos-latest, 3.12)` - macOS Python 3.12 tests
- `test (windows-latest, 3.11)` - Windows Python 3.11 tests
- `test (windows-latest, 3.12)` - Windows Python 3.12 tests
- `build` - Package build verification
- `verify-package` - Installation verification

**To add these:**
1. Go to **Settings → Branches → Edit protection rule for `main`**
2. Enable "Require status checks to pass before merging"
3. Search for and select the checks listed above
4. Save changes

---

## 📈 Monitoring & Metrics

### Workflow Status Badges

Add to README.md:

```markdown
[![CI](https://github.com/sagar100rathod/deep-ml/actions/workflows/python-ci.yml/badge.svg)](https://github.com/sagar100rathod/deep-ml/actions/workflows/python-ci.yml)
[![Publish](https://github.com/sagar100rathod/deep-ml/actions/workflows/python-publish.yml/badge.svg)](https://github.com/sagar100rathod/deep-ml/actions/workflows/python-publish.yml)
[![Docs](https://github.com/sagar100rathod/deep-ml/actions/workflows/docs.yml/badge.svg)](https://github.com/sagar100rathod/deep-ml/actions/workflows/docs.yml)
[![Code Quality](https://github.com/sagar100rathod/deep-ml/actions/workflows/code-quality.yml/badge.svg)](https://github.com/sagar100rathod/deep-ml/actions/workflows/code-quality.yml)
[![codecov](https://codecov.io/gh/sagar100rathod/deep-ml/branch/main/graph/badge.svg)](https://codecov.io/gh/sagar100rathod/deep-ml)
```

### Key Metrics to Track

1. **Test Coverage:** Via Codecov
2. **Build Success Rate:** GitHub Actions dashboard
3. **Dependency Updates:** Dependabot PRs
4. **Security Alerts:** Security tab
5. **Code Quality Trends:** CodeQL results

---

## 🚦 Release Process

### Standard Release

```bash
# 1. Ensure you're on main and up to date
git checkout main
git pull origin main

# 2. Update version
poetry version patch  # or minor/major

# 3. Update CHANGELOG.md manually

# 4. Commit changes
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to X.Y.Z"
git push origin main

# 5. Create and push tag
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z

# 6. Workflow automatically:
#    - Creates GitHub release
#    - Publishes to PyPI
#    - Attaches build artifacts
```

### Hotfix Release

```bash
# 1. Create hotfix branch from main
git checkout -b hotfix/X.Y.Z main

# 2. Make fixes and test
# ... make changes ...
git commit -m "fix: critical bug description"

# 3. Update version (patch)
poetry version patch

# 4. Merge to main
git checkout main
git merge hotfix/X.Y.Z

# 5. Tag and push
git tag -a vX.Y.Z -m "Hotfix: description"
git push origin main --tags

# 6. Delete hotfix branch
git branch -d hotfix/X.Y.Z
```

---

## 🛠️ Maintenance

### Weekly Tasks

- [ ] Review Dependabot PRs
- [ ] Check code quality reports
- [ ] Review security alerts
- [ ] Update documentation if needed

### Monthly Tasks

- [ ] Review and update workflows
- [ ] Check action versions for updates
- [ ] Review test coverage trends
- [ ] Clean up old artifacts

### Quarterly Tasks

- [ ] Audit all GitHub Actions used
- [ ] Review and update dependencies
- [ ] Update CI/CD documentation
- [ ] Review branch protection rules

---

## 🤝 Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.

---

## 📞 Support

- **Issues:** https://github.com/sagar100rathod/deep-ml/issues
- **Discussions:** https://github.com/sagar100rathod/deep-ml/discussions
- **Email:** sagar100rathod@gmail.com

---

## 📝 License

MIT License - See [LICENSE](../LICENSE)

---

## ✅ Setup Checklist

Use this checklist to verify your CI/CD setup:

- [ ] All workflow files are in `.github/workflows/`
- [ ] GitHub secrets configured (PYPI_API_TOKEN, TEST_PYPI_API_TOKEN)
- [ ] GitHub Pages enabled with "GitHub Actions" source
- [x] Branch protection rules set for `main` ✅ (Already configured)
- [ ] Required status checks added to branch protection
- [ ] CodeQL analysis enabled
- [ ] Dependabot configured and enabled
- [ ] Issue templates available
- [ ] PR template configured
- [ ] CONTRIBUTING.md reviewed
- [ ] Badges added to README.md
- [ ] First workflow run successful
- [ ] Documentation builds and deploys correctly
- [ ] Test PyPI publish works
- [ ] Release workflow tested

---

**Last Updated:** April 4, 2026
**Version:** 1.0.0

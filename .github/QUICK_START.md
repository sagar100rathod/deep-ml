# 🚀 CI/CD Quick Start Guide

## ⚡ 5-Minute Setup

### Step 1: Configure Secrets (2 minutes)

1. Go to your repo: `https://github.com/sagar100rathod/deep-ml`
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add these two secrets:

   **Secret 1: PYPI_API_TOKEN**
   - Name: `PYPI_API_TOKEN`
   - Value: Get from https://pypi.org/manage/account/token/
   - Click **Add secret**

   **Secret 2: TEST_PYPI_API_TOKEN**
   - Name: `TEST_PYPI_API_TOKEN`
   - Value: Get from https://test.pypi.org/manage/account/token/
   - Click **Add secret**

### Step 2: Enable GitHub Pages (1 minute)

1. Go to **Settings** → **Pages**
2. Under **Source**, select **GitHub Actions**
3. Click **Save**

### Step 3: Push the Workflows (2 minutes)

**Note:** Since `main` branch has protection enabled, you'll need to create a PR:

```bash
cd /Users/sagar.rathod/Documents/Github/deep-ml

# Create a feature branch
git checkout -b ci/add-workflows

# Add all CI/CD files
git add .github/ CONTRIBUTING.md pyproject.toml poetry.lock

# Commit
git commit -m "ci: add comprehensive CI/CD workflows

- Enhanced CI with multi-OS and multi-Python testing
- Added automated PyPI publishing
- Added documentation deployment
- Added release automation
- Added code quality and security scanning
- Added PR automation and templates
- Added Dependabot configuration
- Fixed pyproject.toml format (PEP 621)"

# Push to feature branch
git push origin ci/add-workflows

# Create pull request
gh pr create --title "Add comprehensive CI/CD workflows" --body "This PR adds complete CI/CD infrastructure including testing, publishing, documentation, and automation workflows."

# Or create PR via GitHub UI
# Go to: https://github.com/sagar100rathod/deep-ml/pulls
```

After creating the PR:
1. Wait for CI checks to pass (first run takes ~10 minutes)
2. Review the PR
3. Merge when ready

### Step 4: Verify Workflows (1 minute)

1. Go to **Actions** tab in GitHub
2. You should see workflows running
3. Check that `CI` workflow passes ✅

---

## 🎯 Common Tasks

### Create a Pull Request

```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes
# ... edit files ...

# Commit with semantic message
git commit -m "feat: add new feature description"

# Push
git push origin feature/my-new-feature

# Create PR on GitHub
# Workflows will automatically:
# - Run tests on 6 environments
# - Check code formatting
# - Auto-label the PR
# - Check PR title format
```

### Make a Release

```bash
# Update version
poetry version patch  # or minor/major

# Commit
git add pyproject.toml
git commit -m "chore: bump version to X.Y.Z"
git push origin main

# Create and push tag
git tag -a vX.Y.Z -m "Release X.Y.Z"
git push origin vX.Y.Z

# Workflow automatically:
# ✅ Creates GitHub release
# ✅ Publishes to PyPI
# ✅ Attaches build artifacts
```

### Test on TestPyPI First

```bash
# Use GitHub CLI to trigger manual publish
gh workflow run python-publish.yml -f repository=testpypi

# Wait for completion, then verify
pip install -i https://test.pypi.org/simple/ deepml==X.Y.Z
```

### Update Documentation

```bash
# Edit docstrings in code or docs/source/

# Commit and push
git add .
git commit -m "docs: update documentation"
git push origin main

# Workflow automatically:
# ✅ Generates API docs
# ✅ Builds HTML
# ✅ Deploys to GitHub Pages
```

---

## 📋 Workflow Cheat Sheet

| Workflow | Trigger | What It Does |
|----------|---------|--------------|
| **CI** | Push/PR to main | Tests on 6 envs, lints, builds package |
| **Code Quality** | Push/PR, Weekly | Security scans, type checking, quality checks |
| **Docs** | Push to main | Builds and deploys documentation |
| **Publish** | Release created | Publishes package to PyPI |
| **Release** | Tag push `v*` | Creates release with changelog |
| **PR Automation** | PR opened | Auto-labels, validates title, greets contributors |

---

## 🔍 Troubleshooting

### CI Fails on Formatting

```bash
# Fix locally
make format

# Commit
git add .
git commit -m "style: fix formatting"
git push
```

### Tests Fail Locally

```bash
# Run tests
poetry run pytest -v

# Run specific test
poetry run pytest tests/test_tasks.py -v

# Check what changed
git diff main...HEAD
```

### Workflow Not Triggering

1. Check `.github/workflows/` files are committed
2. Check workflow triggers in YAML
3. Go to **Actions** tab → Click workflow → **Enable workflow**

### PyPI Publish Fails

1. Verify secrets are set: **Settings** → **Secrets**
2. Check token hasn't expired
3. Ensure version number is unique (not already on PyPI)

---

## 🎨 Customization

### Change Python Versions

Edit `.github/workflows/python-ci.yml`:

```yaml
matrix:
  python-version: ["3.11", "3.12", "3.13"]  # Add/remove versions
```

### Change Test OS

Edit `.github/workflows/python-ci.yml`:

```yaml
matrix:
  os: [ubuntu-latest, macos-latest, windows-latest]  # Add/remove OS
```

### Add More Code Quality Tools

Edit `.github/workflows/code-quality.yml`:

```yaml
- name: Run your-tool
  run: poetry run your-tool deepml
```

---

## 📊 Monitoring

### View Workflow Status

```bash
# List recent workflow runs
gh run list

# View specific run
gh run view <run-id>

# Watch a running workflow
gh run watch
```

### Add Badges to README

```markdown
[![CI](https://github.com/sagar100rathod/deep-ml/actions/workflows/python-ci.yml/badge.svg)](https://github.com/sagar100rathod/deep-ml/actions/workflows/python-ci.yml)
[![codecov](https://codecov.io/gh/sagar100rathod/deep-ml/branch/main/graph/badge.svg)](https://codecov.io/gh/sagar100rathod/deep-ml)
```

---

## ✅ Success Checklist

After setup, verify:

- [ ] Pushed all workflow files to GitHub
- [ ] Both PyPI secrets configured
- [ ] GitHub Pages enabled
- [ ] CI workflow runs and passes
- [ ] Documentation builds successfully
- [ ] Can create a test PR
- [ ] PR gets auto-labeled
- [ ] Tests run on PR
- [ ] Badges added to README

---

## 🆘 Get Help

- **Full Documentation**: See `.github/CI_CD_SETUP.md`
- **Workflow Details**: See `.github/workflows/README.md`
- **Contributing**: See `CONTRIBUTING.md`
- **Issues**: https://github.com/sagar100rathod/deep-ml/issues

---

**Quick Start Complete!** 🎉

Your CI/CD pipeline is now ready. Every push will be tested, every release will be automated, and your documentation will always be up-to-date.

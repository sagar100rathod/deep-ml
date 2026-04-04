# 🔒 Working with Branch Protection - PR Workflow Guide

## Overview

The `main` branch is protected and requires pull requests before merging. This guide shows you how to work effectively with this setup.

---

## 🚀 Standard Development Workflow

### 1. Create a Feature Branch

```bash
# Make sure you're on main and up to date
git checkout main
git pull origin main

# Create a new feature branch
git checkout -b feature/your-feature-name
# or for fixes:
git checkout -b fix/bug-description
```

### 2. Make Your Changes

```bash
# Make changes to your code
# ... edit files ...

# Test locally
make test
make format

# Commit with semantic message
git add .
git commit -m "feat: add new feature description"
```

### 3. Push Your Branch

```bash
git push origin feature/your-feature-name
```

### 4. Create Pull Request

**Via GitHub UI:**
1. Go to: https://github.com/YOUR_USERNAME/deep-ml
2. Click "Compare & pull request"
3. Fill in the PR template
4. Click "Create pull request"

**Via GitHub CLI:**
```bash
gh pr create --fill
```

### 5. Wait for CI Checks

The following workflows will run automatically:
- ✅ **Lint** - Code formatting checks
- ✅ **Test** - Tests on 6 environments (3 OS × 2 Python)
- ✅ **Build** - Package building
- ✅ **Verify** - Installation verification
- ✅ **Code Quality** - Security and quality scans
- ✅ **PR Automation** - Auto-labeling and validation

### 6. Address Review Feedback

```bash
# Make requested changes
# ... edit files ...

# Commit and push (updates the PR automatically)
git add .
git commit -m "fix: address review feedback"
git push origin feature/your-feature-name
```

### 7. Merge (After Approval)

Once all checks pass and you have approval:
- Click **"Squash and merge"** on GitHub
- Delete the feature branch

---

## 🤖 Dependabot PR Workflow

### Automatic Handling

With the `dependabot-auto-merge.yml` workflow:
- ✅ **Patch updates** - Auto-approved and auto-merged
- ✅ **Minor updates** - Auto-approved and auto-merged
- ⚠️ **Major updates** - Flagged for manual review

### Manual Review Process

For major updates:
1. Dependabot creates PR
2. Workflow adds comment: "⚠️ This is a major version update"
3. Review the changes
4. Check the changelog of the updated package
5. Run tests locally if needed
6. Approve and merge manually

---

## 🔥 Hotfix Workflow

For urgent fixes to production:

```bash
# 1. Create hotfix branch from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-bug-fix

# 2. Make the fix
# ... edit files ...
git add .
git commit -m "fix: resolve critical bug"

# 3. Push and create PR
git push origin hotfix/critical-bug-fix
gh pr create --fill

# 4. Request expedited review
# Tag with "priority" label
gh pr edit --add-label "priority"

# 5. After merge, create release if needed
git checkout main
git pull origin main
poetry version patch
git add pyproject.toml
git commit -m "chore: bump version for hotfix"
git push origin main
git tag -a v$(poetry version -s) -m "Hotfix release"
git push origin --tags
```

---

## 🎯 Making Direct Pushes (Special Cases)

**Note:** With branch protection, you cannot push directly to `main`. All changes must go through PRs.

### For Documentation-Only Changes

```bash
# Still use a PR, but it's quick:
git checkout -b docs/update-readme
# ... edit README.md ...
git commit -m "docs: update installation instructions"
git push origin docs/update-readme
gh pr create --fill
# Merge once CI passes
```

### For Version Bumps (Releases)

Use a temporary branch:
```bash
# 1. Create release branch
git checkout -b release/v2.0.2

# 2. Bump version
poetry version patch
git add pyproject.toml
git commit -m "chore: bump version to 2.0.2"

# 3. Create PR
git push origin release/v2.0.2
gh pr create --title "Release v2.0.2" --body "Version bump for release"

# 4. After merge, tag the release
git checkout main
git pull origin main
git tag -a v2.0.2 -m "Release v2.0.2"
git push origin v2.0.2
```

---

## 🔧 Configuring Required Status Checks

### Recommended Checks to Require

Go to: **Settings → Branches → Edit protection rule for `main`**

Enable "Require status checks to pass before merging" and select:

**Critical checks (must pass):**
- ✅ `lint` - Code formatting
- ✅ `test (ubuntu-latest, 3.12)` - Ubuntu tests
- ✅ `build` - Package build
- ✅ `verify-package` - Installation check

**Optional checks (good to have):**
- `test (ubuntu-latest, 3.11)`
- `test (macos-latest, 3.11)`
- `test (macos-latest, 3.12)`
- `test (windows-latest, 3.11)`
- `test (windows-latest, 3.12)`
- `code-quality` - Code quality analysis

### Configuration

```yaml
Require status checks to pass before merging: ✅
  - Require branches to be up to date before merging: ✅

Status checks that are required:
  - lint
  - test (ubuntu-latest, 3.12)
  - build
  - verify-package
```

---

## 📊 PR Checklist

Before creating a PR, ensure:

- [ ] Branch is up to date with `main`
- [ ] Code is formatted (`make format`)
- [ ] Tests pass locally (`make test`)
- [ ] Commits follow semantic conventions
- [ ] PR title is descriptive
- [ ] PR description filled out
- [ ] Related issues are linked

---

## 🚨 Troubleshooting

### "Required status checks must pass"

**Problem:** Can't merge because checks are failing.

**Solution:**
```bash
# Fix the issues locally
make format  # Fix formatting
make test    # Fix failing tests

# Commit and push
git add .
git commit -m "fix: resolve CI failures"
git push origin your-branch-name
```

### "Branch is out of date"

**Problem:** Your branch doesn't have the latest changes from `main`.

**Solution:**
```bash
# Update your branch
git checkout your-branch-name
git fetch origin
git rebase origin/main

# Resolve any conflicts
# ... fix conflicts ...
git add .
git rebase --continue

# Force push (updates the PR)
git push --force-with-lease origin your-branch-name
```

### "No reviewers available"

**Problem:** You're a solo maintainer and need to approve your own PR.

**Solution:**
1. Go to **Settings → Branches → Edit protection rule**
2. Uncheck "Require approvals" (or set to 0)
3. Keep status checks enabled
4. Merge after CI passes

Or configure GitHub Actions to auto-approve:
```yaml
# In a workflow
- uses: hmarr/auto-approve-action@v3
  with:
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

---

## 🎓 Best Practices

### 1. Small, Focused PRs
- Keep PRs small and focused on one feature/fix
- Easier to review
- Faster to merge

### 2. Clear Commit Messages
```bash
# Good
git commit -m "feat: add support for custom loss functions"
git commit -m "fix: resolve memory leak in data loader"

# Bad
git commit -m "update"
git commit -m "fix stuff"
```

### 3. Keep Branch Updated
```bash
# Regularly sync with main
git fetch origin
git rebase origin/main
```

### 4. Delete Merged Branches
```bash
# After PR is merged
git checkout main
git pull origin main
git branch -d feature/your-feature
git remote prune origin
```

---

## 🤝 Team Workflow (Future)

When working with a team:

### Code Review Process
1. **Author** creates PR
2. **CI** runs automated checks
3. **Reviewer** reviews code
4. **Author** addresses feedback
5. **Reviewer** approves
6. **Author** or **Reviewer** merges

### Review Guidelines
- Review within 24 hours
- Focus on logic, not style (CI handles that)
- Ask questions, don't demand changes
- Approve when satisfied

---

## 📚 Related Documentation

- **Quick Start:** `.github/QUICK_START.md`
- **Contributing:** `CONTRIBUTING.md`
- **CI/CD Setup:** `.github/CI_CD_SETUP.md`
- **Release Process:** See "Release Management" in CI_CD_SETUP.md

---

## ✅ Summary

With branch protection enabled:
- ✅ All changes go through PRs
- ✅ CI checks run automatically
- ✅ Code quality is enforced
- ✅ History stays clean
- ✅ Collaboration is structured

**Your workflow is now professional and production-ready!** 🚀

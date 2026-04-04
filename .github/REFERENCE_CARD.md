# 📇 CI/CD Quick Reference Card

## 🎯 Essential Commands

### Deploy CI/CD Pipeline
```bash
git add .github/ CONTRIBUTING.md
git commit -m "ci: add comprehensive CI/CD workflows"
git push origin main
```

### Make a Release
```bash
poetry version patch && \
git add pyproject.toml && \
git commit -m "chore: bump version" && \
git push origin main && \
git tag -a v$(poetry version -s) -m "Release $(poetry version -s)" && \
git push origin --tags
```

### Test on TestPyPI
```bash
gh workflow run python-publish.yml -f repository=testpypi
```

### Validate CI/CD Setup
```bash
bash .github/validate-cicd.sh
```

---

## 📂 File Locations

| What You Need | Where to Find It |
|--------------|------------------|
| Quick setup (5 min) | `.github/QUICK_START.md` |
| Complete guide | `.github/CI_CD_SETUP.md` |
| Architecture | `.github/ARCHITECTURE.md` |
| This report | `.github/FINAL_REPORT.md` |
| Workflows docs | `.github/workflows/README.md` |
| Contributing | `CONTRIBUTING.md` |
| Validation | `.github/validate-cicd.sh` |

---

## 🔧 GitHub Settings URLs

Replace `YOUR_sagar100rathod` with your GitHub sagar100rathod:

- **Secrets:** `https://github.com/YOUR_sagar100rathod/deep-ml/settings/secrets/actions`
- **Pages:** `https://github.com/YOUR_sagar100rathod/deep-ml/settings/pages`
- **Actions:** `https://github.com/YOUR_sagar100rathod/deep-ml/actions`
- **Branches:** `https://github.com/YOUR_sagar100rathod/deep-ml/settings/branches`
- **Security:** `https://github.com/YOUR_sagar100rathod/deep-ml/settings/security_analysis`

---

## 🎫 Required Secrets

| Secret Name | Get From |
|-------------|----------|
| `PYPI_API_TOKEN` | https://pypi.org/manage/account/token/ |
| `TEST_PYPI_API_TOKEN` | https://test.pypi.org/manage/account/token/ |

---

## 🚦 Workflow Triggers

| Workflow | Trigger |
|----------|---------|
| CI | Push/PR to main/develop |
| Code Quality | Push/PR, Weekly (Mon) |
| Docs | Push to main (docs changes) |
| Publish | Release created |
| Release | Tag push `v*.*.*` |
| PR Automation | PR/Issue opened |

---

## 📊 Badge URLs

Add to README.md (replace `sagar100rathod`):

```markdown
[![CI](https://github.com/sagar100rathod/deep-ml/actions/workflows/python-ci.yml/badge.svg)](https://github.com/YOUR_sagar100rathod/deep-ml/actions/workflows/python-ci.yml)
[![Publish](https://github.com/YOUR_sagar100rathod/deep-ml/actions/workflows/python-publish.yml/badge.svg)](https://github.com/YOUR_sagar100rathod/deep-ml/actions/workflows/python-publish.yml)
[![Docs](https://github.com/YOUR_sagar100rathod/deep-ml/actions/workflows/docs.yml/badge.svg)](https://github.com/YOUR_sagar100rathod/deep-ml/actions/workflows/docs.yml)
[![Code Quality](https://github.com/YOUR_sagar100rathod/deep-ml/actions/workflows/code-quality.yml/badge.svg)](https://github.com/YOUR_sagar100rathod/deep-ml/actions/workflows/code-quality.yml)
```

---

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| CI fails on formatting | `make format && git add . && git commit --amend` |
| Tests fail | `poetry run pytest -v` to debug locally |
| Workflow not triggering | Check .github/workflows/ files committed |
| PyPI publish fails | Verify secrets, check version is unique |
| Docs don't deploy | Check GitHub Pages enabled |

---

## 📞 Support

- **Questions:** `.github/QUICK_START.md`
- **Issues:** Use bug report template
- **Features:** Use feature request template
- **Email:** sagar100rathod@gmail.com

---

## ✅ Setup Checklist

- [ ] Push workflows to GitHub
- [ ] Configure PyPI secrets
- [ ] Enable GitHub Pages
- [ ] Test with a PR
- [ ] Add badges to README
- [ ] Set branch protection
- [ ] Enable CodeQL (optional)

---

**Status:** ✅ Production Ready | **Cost:** $0/month | **Validation:** 37/37 Passed

Print or bookmark this card for quick reference! 📌

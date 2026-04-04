#!/bin/bash

# CI/CD Validation Script
# This script validates that all CI/CD components are properly configured


echo "🔍 Deep-ML CI/CD Validation Script"
echo "===================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Helper functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# Check 1: Workflow files exist
echo "📁 Checking workflow files..."
WORKFLOWS=(
    ".github/workflows/python-ci.yml"
    ".github/workflows/python-publish.yml"
    ".github/workflows/docs.yml"
    ".github/workflows/release.yml"
    ".github/workflows/code-quality.yml"
    ".github/workflows/pr-automation.yml"
)

for workflow in "${WORKFLOWS[@]}"; do
    if [ -f "$workflow" ]; then
        check_pass "$workflow exists"
    else
        check_fail "$workflow missing"
    fi
done
echo ""

# Check 2: Configuration files exist
echo "⚙️  Checking configuration files..."
CONFIGS=(
    ".github/dependabot.yml"
    ".github/labeler.yml"
    ".github/pull_request_template.md"
    ".github/ISSUE_TEMPLATE/bug_report.md"
    ".github/ISSUE_TEMPLATE/feature_request.md"
)

for config in "${CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        check_pass "$config exists"
    else
        check_fail "$config missing"
    fi
done
echo ""

# Check 3: Documentation files exist
echo "📚 Checking documentation files..."
DOCS=(
    ".github/CI_CD_SETUP.md"
    ".github/QUICK_START.md"
    ".github/ARCHITECTURE.md"
    ".github/workflows/README.md"
    "CONTRIBUTING.md"
)

for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        check_pass "$doc exists"
    else
        check_fail "$doc missing"
    fi
done
echo ""

# Check 4: YAML syntax validation
echo "✅ Validating YAML syntax..."
if command -v yamllint &> /dev/null; then
    for workflow in "${WORKFLOWS[@]}"; do
        if yamllint -d relaxed "$workflow" &> /dev/null; then
            check_pass "$workflow has valid YAML syntax"
        else
            check_fail "$workflow has YAML syntax errors"
        fi
    done
else
    check_warn "yamllint not installed, skipping YAML validation"
    check_warn "Install with: pip install yamllint"
fi
echo ""

# Check 5: Python project files
echo "🐍 Checking Python project files..."
if [ -f "pyproject.toml" ]; then
    check_pass "pyproject.toml exists"

    # Check version
    if grep -q "version = " pyproject.toml; then
        VERSION=$(grep "version = " pyproject.toml | head -1 | cut -d'"' -f2)
        check_pass "Version found: $VERSION"
    else
        check_fail "Version not found in pyproject.toml"
    fi
else
    check_fail "pyproject.toml missing"
fi

if [ -f "poetry.lock" ]; then
    check_pass "poetry.lock exists"
else
    check_warn "poetry.lock missing (run: poetry install)"
fi
echo ""

# Check 6: Test directory
echo "🧪 Checking test setup..."
if [ -d "tests" ]; then
    check_pass "tests/ directory exists"

    TEST_COUNT=$(find tests -name "test_*.py" | wc -l | tr -d ' ')
    if [ "$TEST_COUNT" -gt 0 ]; then
        check_pass "Found $TEST_COUNT test files"
    else
        check_warn "No test files found (test_*.py)"
    fi
else
    check_fail "tests/ directory missing"
fi
echo ""

# Check 7: Pre-commit configuration
echo "🔧 Checking pre-commit setup..."
if [ -f ".pre-commit-config.yaml" ]; then
    check_pass ".pre-commit-config.yaml exists"

    if command -v pre-commit &> /dev/null; then
        check_pass "pre-commit installed"
    else
        check_warn "pre-commit not installed (run: pip install pre-commit)"
    fi
else
    check_fail ".pre-commit-config.yaml missing"
fi
echo ""

# Check 8: Git setup
echo "🔀 Checking Git configuration..."
if [ -d ".git" ]; then
    check_pass "Git repository initialized"

    # Check remote
    if git remote -v | grep -q "github.com"; then
        REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
        check_pass "GitHub remote configured: $REMOTE"
    else
        check_warn "No GitHub remote found"
    fi

    # Check current branch
    BRANCH=$(git branch --show-current)
    check_pass "Current branch: $BRANCH"
else
    check_fail "Not a Git repository"
fi
echo ""

# Check 9: Documentation directory
echo "📖 Checking documentation setup..."
if [ -d "docs" ]; then
    check_pass "docs/ directory exists"

    if [ -f "docs/Makefile" ]; then
        check_pass "docs/Makefile exists"
    else
        check_warn "docs/Makefile missing"
    fi

    if [ -d "docs/source" ]; then
        check_pass "docs/source/ directory exists"
    else
        check_warn "docs/source/ directory missing"
    fi
else
    check_warn "docs/ directory missing"
fi
echo ""

# Check 10: Makefile targets
echo "🛠️  Checking Makefile..."
if [ -f "Makefile" ]; then
    check_pass "Makefile exists"

    TARGETS=("install" "test" "lint" "format" "docs")
    for target in "${TARGETS[@]}"; do
        if grep -q "^${target}:" Makefile; then
            check_pass "Makefile target '$target' found"
        else
            check_warn "Makefile target '$target' missing"
        fi
    done
else
    check_fail "Makefile missing"
fi
echo ""

# Check 11: Source code
echo "📦 Checking source code..."
if [ -d "deepml" ]; then
    check_pass "deepml/ package directory exists"

    if [ -f "deepml/__init__.py" ]; then
        check_pass "deepml/__init__.py exists"

        if grep -q "__version__" deepml/__init__.py; then
            check_pass "__version__ defined in __init__.py"
        else
            check_warn "__version__ not defined in __init__.py"
        fi
    else
        check_fail "deepml/__init__.py missing"
    fi
else
    check_fail "deepml/ package directory missing"
fi
echo ""

# Summary
echo "======================================"
echo "📊 Validation Summary"
echo "======================================"
echo -e "${GREEN}Passed:${NC}   $PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC}   $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Commit and push changes: git add . && git commit -m 'ci: add CI/CD workflows' && git push"
    echo "2. Configure GitHub secrets (PYPI_API_TOKEN, TEST_PYPI_API_TOKEN)"
    echo "3. Enable GitHub Pages"
    echo "4. Review workflows at: https://github.com/sagar100rathod/deep-ml/actions"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please fix the issues above.${NC}"
    echo ""
    exit 1
fi

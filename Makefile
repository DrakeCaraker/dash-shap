.PHONY: setup test test-fast test-slow lint fmt fmt-check typecheck coverage ci rebase clean audit

setup:
	pip install -e ".[dev]" 2>/dev/null || pip install -e . && pip install -r requirements-dev.txt
	git config core.hooksPath .githooks
	@echo "Verifying tools..."
	@python3 -m ruff --version >/dev/null 2>&1 && echo "  ruff: $$(python3 -m ruff --version)" || echo "  ruff: NOT FOUND (install with: pip install ruff)"
	@python3 -m mypy --version >/dev/null 2>&1 && echo "  mypy: $$(python3 -m mypy --version)" || echo "  mypy: NOT FOUND (install with: pip install mypy)"
	@python3 -c "import pytest; print(f'  pytest: {pytest.__version__}')" 2>/dev/null || echo "  pytest: NOT FOUND (install with: pip install pytest)"
	@python3 -c "import xgboost; print(f'  xgboost: {xgboost.__version__}')" 2>/dev/null || echo "  xgboost: NOT FOUND (install with: pip install xgboost)"
	@echo ""
	@echo "Setup complete. Git hooks activated. Run 'make test-fast' to verify."

test:
	python3 -m pytest -v

test-fast:
	python3 -m pytest -v -m "not slow"

test-slow:
	python3 -m pytest -v -m "slow"

lint:
	python3 -m ruff check .

fmt:
	python3 -m ruff format .

fmt-check:
	python3 -m ruff format --check .

typecheck:
	python3 -m mypy dash_shap/ --ignore-missing-imports --no-error-summary

coverage:
	python3 -m pytest --cov=dash_shap --cov-report=term-missing --cov-fail-under=70

ci: lint fmt-check typecheck test coverage
	@echo "All CI checks passed."

rebase:
	git fetch origin main
	git rebase origin/main
	@echo "Rebase complete. Run 'git push --force-with-lease' to update the remote branch."

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

audit:
	@echo "=== Notebook ID Check ==="
	-python3 scripts/check_notebook_ids.py
	@echo ""
	@echo "=== Sensitive Data Scan ==="
	@git ls-files | xargs grep -nE '(sk-[a-zA-Z0-9]{20,}|password\s*=\s*["'"'"'][^"'"'"']+|api_key\s*=\s*["'"'"'][^"'"'"']+)' || echo "No sensitive patterns found."
	@echo ""
	@echo "For full AI-powered audit, run /audit in Claude Code"

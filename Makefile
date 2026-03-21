.PHONY: test test-fast test-slow lint fmt fmt-check typecheck coverage ci rebase clean

test:
	pytest -v

test-fast:
	pytest -v -m "not slow"

test-slow:
	pytest -v -m "slow"

lint:
	ruff check .

fmt:
	ruff format .

fmt-check:
	ruff format --check .

typecheck:
	mypy dash_shap/ --ignore-missing-imports --no-error-summary

coverage:
	pytest --cov=dash_shap --cov-report=term-missing --cov-fail-under=70

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

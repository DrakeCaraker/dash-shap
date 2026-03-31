"""CLI entry point for dash-shap experiment runner.

Delegates to run_experiments_parallel.py's __main__ block.
Installed as `dash-run-experiments` console_script via pyproject.toml.
"""

import sys
import os


def main():
    """Run the experiment CLI by executing run_experiments_parallel.py as __main__."""
    # Find the runner script relative to the repo root
    # When installed via pip, the script lives at the repo root (not in the package)
    # So we use runpy to execute it with the right __name__
    script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "run_experiments_parallel.py")
    if not os.path.exists(script):
        print(
            "Error: run_experiments_parallel.py not found.\n"
            "This command must be run from within the dash-shap repository.\n"
            "Use `python run_experiments_parallel.py` directly instead.",
            file=sys.stderr,
        )
        sys.exit(1)
    import runpy

    sys.argv[0] = script
    runpy.run_path(script, run_name="__main__")

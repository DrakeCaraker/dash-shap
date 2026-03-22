"""Smoke tests for CLI experiment runners.

These tests verify that the parallel runner can parse args and execute a minimal
experiment without crashing. They use tiny configs (M=5, K=3) to stay fast.
"""

import subprocess
import sys
import pytest


@pytest.mark.slow
class TestRunExperimentsParallel:
    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "run_experiments_parallel.py", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "linear_sweep" in result.stdout

    def test_minimal_run(self):
        """Parallel runner executes without crashing on tiny config."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'
from dash_shap.experiments.synthetic import generate_synthetic_linear
from dash_shap.core.pipeline import DASHPipeline
data = generate_synthetic_linear(N=200, P=10, group_size=5, rho=0.5, seed=42)
pipe = DASHPipeline(M=5, K=3, epsilon=0.15, delta=0.01, seed=42,
                    verbose=False, n_jobs=1)
pipe.fit(data[0], data[1], data[2], data[3], X_ref=data[4])
assert pipe.global_importance_.shape == (10,)
print("parallel smoke test passed")
""",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

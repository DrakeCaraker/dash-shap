"""Regression gate: v7 results must reproduce v6 within tolerance.

These tests skip gracefully until v7 JSON files exist in results/tables/.
Once the v7 notebook run completes, they become the automated guard against
accidental methodology changes breaking published results.
"""
import json
import pytest
from pathlib import Path

TOLERANCE = 0.01  # max allowed stability deviation from v6

V6_PATH = Path("results/tables/benchmark_v6_frozen.json")


def _load_result(name):
    p = Path(f"results/tables/{name}.json")
    return json.load(p.open()) if p.exists() else None


@pytest.fixture(scope="module")
def v6():
    assert V6_PATH.exists(), f"v6 frozen results not found at {V6_PATH}"
    return json.load(V6_PATH.open())


@pytest.mark.slow
def test_linear_sweep_dash_stability_regression(v6):
    v7 = _load_result("linear_sweep")
    if v7 is None:
        pytest.skip("linear_sweep.json not yet generated — run v7 notebook first")
    for rho in ["0.9", "0.95"]:
        v6_val = v6["linear_sweep"][rho]["DASH (MaxMin)"]["stability"]
        v7_val = v7[rho]["DASH (MaxMin)"]["stability"]
        assert abs(v7_val - v6_val) < TOLERANCE, (
            f"DASH stability at rho={rho}: v7={v7_val:.4f} vs v6={v6_val:.4f} "
            f"(tolerance={TOLERANCE})"
        )


@pytest.mark.slow
def test_breast_cancer_regression(v6):
    v7 = _load_result("real_breast_cancer")
    if v7 is None:
        pytest.skip("real_breast_cancer.json not yet generated")
    v6_val = v6["real_breast_cancer"]["DASH (MaxMin)"]["stability"]
    v7_val = v7["DASH (MaxMin)"]["stability"]
    assert abs(v7_val - v6_val) < TOLERANCE * 2, (
        f"Breast Cancer DASH: v7={v7_val:.4f} vs v6={v6_val:.4f}"
    )

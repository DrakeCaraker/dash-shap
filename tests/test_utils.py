"""Tests for dash_shap.utils (checkpoint and I/O)."""

import io
import json
import pickle
import warnings
import numpy as np
from dash_shap.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    has_checkpoint,
    clear_checkpoint,
    clear_checkpoints_by_prefix,
    _checkpoint_path,
    _sanitize_ckpt_name,
    _LegacyUnpickler,
    _config_fingerprint,
)
from dash_shap.utils.io import save_json
from dash_shap.utils.provenance import (
    append_provenance_md,
    capture_run_meta,
    validate_result,
)
from dash_shap.core.pipeline import DASHPipeline


class TestLegacyUnpickler:
    def test_legacy_module_remapped(self):
        """find_class remaps 'dash.core.pipeline' → 'dash_shap.core.pipeline'."""
        unpickler = _LegacyUnpickler(io.BytesIO(b""))
        cls = unpickler.find_class("dash.core.pipeline", "DASHPipeline")
        assert cls is DASHPipeline

    def test_current_module_passthrough(self):
        """find_class passes through 'dash_shap.*' modules unchanged."""
        unpickler = _LegacyUnpickler(io.BytesIO(b""))
        cls = unpickler.find_class("dash_shap.core.pipeline", "DASHPipeline")
        assert cls is DASHPipeline


class TestSanitizeName:
    def test_spaces_replaced(self):
        assert _sanitize_ckpt_name("hello world") == "hello_world"

    def test_parens_removed(self):
        assert _sanitize_ckpt_name("rho(0.9)") == "rho0.9"

    def test_commas_removed(self):
        assert _sanitize_ckpt_name("a,b,c") == "abc"

    def test_lowercased(self):
        assert _sanitize_ckpt_name("ABC") == "abc"


class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        data = {"arr": np.array([1.0, 2.0, 3.0]), "val": 42}
        save_checkpoint("test1", checkpoint_dir=str(tmp_path), **data)
        loaded = load_checkpoint("test1", checkpoint_dir=str(tmp_path))
        assert loaded is not None
        np.testing.assert_array_equal(loaded["arr"], data["arr"])
        assert loaded["val"] == 42

    def test_has_checkpoint(self, tmp_path):
        assert not has_checkpoint("missing", checkpoint_dir=str(tmp_path))
        save_checkpoint("exists", checkpoint_dir=str(tmp_path), x=1)
        assert has_checkpoint("exists", checkpoint_dir=str(tmp_path))

    def test_load_missing_returns_none(self, tmp_path):
        assert load_checkpoint("nonexistent", checkpoint_dir=str(tmp_path)) is None

    def test_clear_checkpoint(self, tmp_path):
        save_checkpoint("to_clear", checkpoint_dir=str(tmp_path), x=1)
        assert has_checkpoint("to_clear", checkpoint_dir=str(tmp_path))
        clear_checkpoint("to_clear", checkpoint_dir=str(tmp_path))
        assert not has_checkpoint("to_clear", checkpoint_dir=str(tmp_path))

    def test_clear_nonexistent_no_error(self, tmp_path):
        clear_checkpoint("does_not_exist", checkpoint_dir=str(tmp_path))

    def test_clear_by_prefix(self, tmp_path):
        save_checkpoint("exp_a", checkpoint_dir=str(tmp_path), x=1)
        save_checkpoint("exp_b", checkpoint_dir=str(tmp_path), x=2)
        save_checkpoint("other", checkpoint_dir=str(tmp_path), x=3)
        clear_checkpoints_by_prefix("exp_", checkpoint_dir=str(tmp_path))
        assert not has_checkpoint("exp_a", checkpoint_dir=str(tmp_path))
        assert not has_checkpoint("exp_b", checkpoint_dir=str(tmp_path))
        assert has_checkpoint("other", checkpoint_dir=str(tmp_path))

    def test_clear_by_prefix_empty_dir(self, tmp_path):
        """No error when checkpoint dir doesn't exist."""
        clear_checkpoints_by_prefix("whatever", checkpoint_dir=str(tmp_path / "nope"))

    def test_checkpoint_creates_directory(self, tmp_path):
        nested = str(tmp_path / "sub" / "dir")
        save_checkpoint("nested", checkpoint_dir=nested, x=1)
        assert has_checkpoint("nested", checkpoint_dir=nested)

    def test_checkpoint_path_format(self):
        path = _checkpoint_path("my_test")
        assert str(path).endswith("ckpt_my_test.pkl")


class TestCheckpointConfigValidation:
    def test_config_hash_matching_no_warning(self, tmp_path):
        """Matching config on save and load → no warning."""
        config = {"M": 200, "K": 30, "N_REPS": 50}
        save_checkpoint("match", checkpoint_dir=str(tmp_path), config=config, val=42)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning → test failure
            result = load_checkpoint("match", checkpoint_dir=str(tmp_path), config=config)
        assert result["val"] == 42

    def test_config_hash_mismatch_warns(self, tmp_path):
        """Different config on load → UserWarning, data still returned."""
        save_checkpoint("mismatch", checkpoint_dir=str(tmp_path), config={"N_REPS": 20}, val=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_checkpoint("mismatch", checkpoint_dir=str(tmp_path), config={"N_REPS": 50})
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "mismatch" in str(w[0].message)
        assert result["val"] == 1  # data still usable

    def test_legacy_checkpoint_no_warning(self, tmp_path):
        """Old checkpoint with no __meta__ loads silently regardless of config."""
        path = tmp_path / "ckpt_legacy.pkl"
        with open(path, "wb") as f:
            pickle.dump({"val": 99}, f)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_checkpoint("legacy", checkpoint_dir=str(tmp_path), config={"M": 200})
        assert len(w) == 0
        assert result["val"] == 99

    def test_no_config_no_warning(self, tmp_path):
        """config=None (default) never warns even if __meta__ present."""
        save_checkpoint("noconfig", checkpoint_dir=str(tmp_path), config={"M": 200}, val=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_checkpoint("noconfig", checkpoint_dir=str(tmp_path))
        assert len(w) == 0
        assert result["val"] == 5

    def test_meta_key_transparent_to_callers(self, tmp_path):
        """__meta__ in returned dict doesn't interfere with named-key access."""
        save_checkpoint("meta", checkpoint_dir=str(tmp_path), config={"M": 200}, rho_results={"rho": 0.9})
        result = load_checkpoint("meta", checkpoint_dir=str(tmp_path), config={"M": 200})
        assert result["rho_results"] == {"rho": 0.9}
        assert "__meta__" in result
        assert "config_hash" in result["__meta__"]


class TestConfigFingerprint:
    def test_deterministic(self):
        cfg = {"M": 200, "K": 30}
        assert _config_fingerprint(cfg) == _config_fingerprint(cfg)

    def test_order_independent(self):
        assert _config_fingerprint({"M": 200, "K": 30}) == _config_fingerprint({"K": 30, "M": 200})

    def test_none_returns_none(self):
        assert _config_fingerprint(None) is None

    def test_different_values_differ(self):
        assert _config_fingerprint({"N_REPS": 20}) != _config_fingerprint({"N_REPS": 50})


class TestSaveJson:
    def test_numpy_types_converted(self, tmp_path):
        data = {
            "float": np.float64(1.5),
            "int": np.int32(42),
            "array": np.array([1, 2, 3]),
            "nested": {"x": np.float32(0.1)},
        }
        path = tmp_path / "test.json"
        save_json(data, str(path))
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["float"] == 1.5
        assert loaded["int"] == 42
        assert loaded["array"] == [1, 2, 3]
        assert isinstance(loaded["nested"]["x"], float)

    def test_plain_python_types(self, tmp_path):
        data = {"a": 1, "b": "hello", "c": [1, 2]}
        path = tmp_path / "plain.json"
        save_json(data, str(path))
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_dict_keys_stringified(self, tmp_path):
        data = {0: "zero", 1: "one"}
        path = tmp_path / "intkeys.json"
        save_json(data, str(path))
        with open(path) as f:
            loaded = json.load(f)
        assert "0" in loaded
        assert "1" in loaded

    def test_meta_embedded_as_first_key(self, tmp_path):
        data = {"stability": 0.95, "rmse": 0.12}
        meta = {"experiment": "test", "n_reps": 50}
        path = tmp_path / "meta_test.json"
        save_json(data, str(path), meta=meta)
        with open(path) as f:
            loaded = json.load(f)
        keys = list(loaded.keys())
        assert keys[0] == "_meta"
        assert loaded["_meta"]["experiment"] == "test"
        assert loaded["stability"] == 0.95

    def test_overwrite_creates_backup(self, tmp_path):
        data = {"x": 1}
        path = tmp_path / "overwrite_test.json"
        save_json(data, str(path), overwrite_protection=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            save_json({"x": 2}, str(path), overwrite_protection=True)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        bak_files = list(tmp_path.glob("overwrite_test.*.bak.json"))
        assert len(bak_files) == 1

    def test_no_backup_when_protection_off(self, tmp_path):
        data = {"x": 1}
        path = tmp_path / "nobackup.json"
        save_json(data, str(path), overwrite_protection=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            save_json({"x": 2}, str(path), overwrite_protection=False)
        assert len(w) == 0
        bak_files = list(tmp_path.glob("*.bak.json"))
        assert len(bak_files) == 0


class TestProvenance:
    def _make_valid_entry(self, n_reps=10):
        return {
            "stability": 0.95,
            "accuracy": 0.88,
            "equity": 0.91,
            "n_successful": n_reps,
        }

    def test_validate_result_clean(self):
        data = {"0.5": {"DASH": self._make_valid_entry()}}
        issues = validate_result(data, "test_exp")
        assert issues == []

    def test_validate_result_nan_stability(self):
        data = {"0.5": {"DASH": {"stability": float("nan"), "n_successful": 10}}}
        issues = validate_result(data, "test_exp")
        assert any("NaN" in i and "stability" in i for i in issues)

    def test_validate_result_zero_successful(self):
        data = {"0.5": {"DASH": {"stability": 0.9, "n_successful": 0}}}
        issues = validate_result(data, "test_exp")
        assert any("n_successful" in i for i in issues)

    def test_validate_result_incomplete_runs(self):
        data = {
            "0.5": {
                "DASH": {
                    "stability": 0.9,
                    "n_successful": 10,
                    "acc_runs": [0.8, 0.9],
                    "n_reps": 10,
                }
            }
        }
        issues = validate_result(data, "test_exp")
        assert any("acc_runs" in i for i in issues)

    def test_validate_result_ci_sanity(self):
        data = {
            "0.5": {
                "DASH": {
                    "stability": 0.9,
                    "stability_lo": 0.95,  # lo > point — invalid
                    "stability_hi": 0.99,
                }
            }
        }
        issues = validate_result(data, "test_exp")
        assert any("CI sanity" in i for i in issues)

    def test_validate_result_degenerate_bootstrap(self):
        data = {
            "0.5": {
                "DASH": {
                    "stability": 0.9,
                    "stability_se": -0.01,
                }
            }
        }
        issues = validate_result(data, "test_exp")
        assert any("stability_se" in i for i in issues)

    def test_validate_skips_meta_keys(self):
        data = {
            "_meta": {"experiment": "foo"},
            "_significance": {"p": 0.01},
            "0.5": {"DASH": self._make_valid_entry()},
        }
        issues = validate_result(data, "test_exp")
        assert issues == []

    def test_validate_result_float_keys(self):
        """Regression: linear_sweep uses float rho keys (0.0, 0.5, etc.)."""
        data = {
            0.0: {"DASH": self._make_valid_entry(), "Single Best": self._make_valid_entry()},
            0.5: {"DASH": self._make_valid_entry()},
            0.9: {"DASH": self._make_valid_entry()},
            "_meta": {"experiment": "linear_sweep"},
        }
        issues = validate_result(data, "linear_sweep")
        assert issues == []

    def test_validate_result_float_keys_with_issues(self):
        """Float keys should still detect data quality problems."""
        data = {
            0.9: {"DASH": {"stability": float("nan"), "n_successful": 10}},
        }
        issues = validate_result(data, "test_exp")
        assert any("NaN" in i for i in issues)

    def test_capture_run_meta_fields(self, tmp_path):
        config = {"M": 200, "K": 30, "N_REPS": 50}
        meta = capture_run_meta("test_exp", 50, config, 123.4, str(tmp_path / "out.json"))
        required = {
            "experiment",
            "timestamp",
            "code_sha",
            "code_dirty",
            "config_sha",
            "n_reps",
            "paper_config",
            "elapsed_s",
            "output",
            "hardware",
        }
        assert required.issubset(meta.keys())
        assert meta["experiment"] == "test_exp"
        assert meta["n_reps"] == 50
        assert isinstance(meta["config_sha"], str) and len(meta["config_sha"]) == 64

    def test_append_provenance_md_creates_file(self, tmp_path):
        config = {"M": 200, "K": 30}
        meta = capture_run_meta("exp1", 20, config, 60.0, str(tmp_path / "out.json"))
        append_provenance_md(meta, str(tmp_path))
        prov = tmp_path / "PROVENANCE.md"
        assert prov.exists()
        content = prov.read_text()
        assert "exp1" in content
        assert "## exp1" in content

    def test_append_provenance_md_accumulates(self, tmp_path):
        config = {"M": 200}
        meta1 = capture_run_meta("exp_a", 10, config, 30.0, str(tmp_path / "a.json"))
        meta2 = capture_run_meta("exp_b", 10, config, 45.0, str(tmp_path / "b.json"))
        append_provenance_md(meta1, str(tmp_path))
        append_provenance_md(meta2, str(tmp_path))
        content = (tmp_path / "PROVENANCE.md").read_text()
        assert "## exp_a" in content
        assert "## exp_b" in content
        assert content.count("## exp_a") == 1
        assert content.count("## exp_b") == 1

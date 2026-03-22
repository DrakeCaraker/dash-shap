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
        save_checkpoint("mismatch", checkpoint_dir=str(tmp_path),
                        config={"N_REPS": 20}, val=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_checkpoint("mismatch", checkpoint_dir=str(tmp_path),
                                     config={"N_REPS": 50})
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
            result = load_checkpoint("legacy", checkpoint_dir=str(tmp_path),
                                     config={"M": 200})
        assert len(w) == 0
        assert result["val"] == 99

    def test_no_config_no_warning(self, tmp_path):
        """config=None (default) never warns even if __meta__ present."""
        save_checkpoint("noconfig", checkpoint_dir=str(tmp_path),
                        config={"M": 200}, val=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_checkpoint("noconfig", checkpoint_dir=str(tmp_path))
        assert len(w) == 0
        assert result["val"] == 5

    def test_meta_key_transparent_to_callers(self, tmp_path):
        """__meta__ in returned dict doesn't interfere with named-key access."""
        save_checkpoint("meta", checkpoint_dir=str(tmp_path),
                        config={"M": 200}, rho_results={"rho": 0.9})
        result = load_checkpoint("meta", checkpoint_dir=str(tmp_path), config={"M": 200})
        assert result["rho_results"] == {"rho": 0.9}
        assert "__meta__" in result
        assert "config_hash" in result["__meta__"]


class TestConfigFingerprint:
    def test_deterministic(self):
        cfg = {"M": 200, "K": 30}
        assert _config_fingerprint(cfg) == _config_fingerprint(cfg)

    def test_order_independent(self):
        assert _config_fingerprint({"M": 200, "K": 30}) == \
               _config_fingerprint({"K": 30, "M": 200})

    def test_none_returns_none(self):
        assert _config_fingerprint(None) is None

    def test_different_values_differ(self):
        assert _config_fingerprint({"N_REPS": 20}) != \
               _config_fingerprint({"N_REPS": 50})


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

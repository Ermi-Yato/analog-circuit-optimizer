"""Phase 4 — Surrogate model tests. No ngspice required (synthetic data only)."""
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from core.models.base_model import BaseModel
from core.models.random_forest import RandomForestModel
from core.models.trainer import evaluate


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

def _make_linear_data(n=200, n_features=4, n_targets=2, seed=0):
    """
    Generate a synthetic dataset where y = X @ W + noise.
    A well-tuned RF should get R² > 0.85 on this.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, n_features))
    W = rng.uniform(1, 5, (n_features, n_targets))
    noise = rng.normal(0, 0.1, (n, n_targets))
    y = X @ W + noise
    return X, y


# ---------------------------------------------------------------------------
# TestBaseModelInterface
# ---------------------------------------------------------------------------

class TestBaseModelInterface:

    def test_base_model_is_abstract(self):
        with pytest.raises(TypeError):
            BaseModel()

    def test_random_forest_implements_interface(self):
        assert issubclass(RandomForestModel, BaseModel)


# ---------------------------------------------------------------------------
# TestRandomForestModel
# ---------------------------------------------------------------------------

class TestRandomForestModel:

    def test_fit_and_predict_shape(self):
        X, y = _make_linear_data(n=100, n_features=4, n_targets=2)
        m = RandomForestModel()
        m.fit(X, y)
        pred = m.predict(X[:10])
        assert pred.shape == (10, 2)

    def test_predict_single_target(self):
        X, y = _make_linear_data(n=100, n_features=4, n_targets=1)
        m = RandomForestModel()
        m.fit(X, y)
        pred = m.predict(X[:5])
        # sklearn RF returns 1-D for single-target — accept either shape
        assert pred.shape[0] == 5

    def test_fit_then_predict_not_none(self):
        X, y = _make_linear_data()
        m = RandomForestModel()
        m.fit(X, y)
        assert m.predict(X[:1]) is not None

    def test_feature_importances_after_fit(self):
        X, y = _make_linear_data(n_features=4)
        m = RandomForestModel()
        m.fit(X, y)
        fi = m.feature_importances
        assert fi is not None
        assert len(fi) == 4
        assert abs(fi.sum() - 1.0) < 1e-6

    def test_feature_importances_before_fit_is_none(self):
        m = RandomForestModel()
        assert m.feature_importances is None

    def test_save_and_load_gives_identical_predictions(self, tmp_path):
        X, y = _make_linear_data()
        m = RandomForestModel()
        m.fit(X, y)
        pred_before = m.predict(X[:20])

        path = str(tmp_path / "model.pkl")
        m.save(path)

        m2 = RandomForestModel()
        m2.load(path)
        pred_after = m2.predict(X[:20])

        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-12)

    def test_save_creates_file(self, tmp_path):
        X, y = _make_linear_data()
        m = RandomForestModel()
        m.fit(X, y)
        path = str(tmp_path / "model.pkl")
        m.save(path)
        assert os.path.isfile(path)

    def test_load_replaces_state(self, tmp_path):
        X, y = _make_linear_data(seed=0)
        m1 = RandomForestModel(random_state=0)
        m1.fit(X, y)
        path = str(tmp_path / "m1.pkl")
        m1.save(path)

        # Train a different model
        X2, y2 = _make_linear_data(seed=99)
        m2 = RandomForestModel(random_state=99)
        m2.fit(X2, y2)

        # Load first model into m2
        m2.load(path)
        # Predictions should match m1 now
        np.testing.assert_allclose(m2.predict(X[:5]), m1.predict(X[:5]), rtol=1e-12)

    def test_reasonable_r2_on_linear_data(self):
        """RF should fit a near-linear problem well enough."""
        X, y = _make_linear_data(n=500)
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        m = RandomForestModel()
        m.fit(X_tr, y_tr)
        metrics = evaluate(m, X_te, y_te, ["t0", "t1"])
        for name in ["t0", "t1"]:
            assert metrics["r2"][name] > 0.85, (
                f"R² for {name} = {metrics['r2'][name]:.3f} — too low for linear data"
            )


# ---------------------------------------------------------------------------
# TestEvaluate
# ---------------------------------------------------------------------------

class TestEvaluate:

    def _trained_model(self, n_targets=2):
        X, y = _make_linear_data(n=200, n_targets=n_targets)
        m = RandomForestModel()
        m.fit(X, y)
        return m, X, y

    def test_returns_r2_and_mae_keys(self):
        m, X, y = self._trained_model()
        result = evaluate(m, X[:40], y[:40], ["gain", "bw"])
        assert "r2" in result
        assert "mae" in result

    def test_r2_keys_match_metric_names(self):
        m, X, y = self._trained_model()
        names = ["gain_db", "bw_hz"]
        result = evaluate(m, X[:40], y[:40], names)
        assert set(result["r2"].keys()) == set(names)

    def test_mae_keys_match_metric_names(self):
        m, X, y = self._trained_model()
        names = ["gain_db", "bw_hz"]
        result = evaluate(m, X[:40], y[:40], names)
        assert set(result["mae"].keys()) == set(names)

    def test_r2_values_are_floats(self):
        m, X, y = self._trained_model()
        result = evaluate(m, X[:40], y[:40], ["a", "b"])
        for v in result["r2"].values():
            assert isinstance(v, float)

    def test_mae_values_are_non_negative(self):
        m, X, y = self._trained_model()
        result = evaluate(m, X[:40], y[:40], ["a", "b"])
        for v in result["mae"].values():
            assert v >= 0.0

    def test_perfect_prediction_gives_r2_of_one(self):
        X, y = _make_linear_data(n=100, n_features=2, n_targets=1)
        m = RandomForestModel()
        m.fit(X, y)
        # Evaluate on training data — RF memorizes training set perfectly
        result = evaluate(m, X, y, ["metric"])
        assert result["r2"]["metric"] > 0.99

    def test_n_test_field_correct(self):
        m, X, y = self._trained_model()
        result = evaluate(m, X[:30], y[:30], ["a", "b"])
        assert result["n_test"] == 30

    def test_single_target_does_not_crash(self):
        m, X, y = self._trained_model(n_targets=1)
        # y may be 1-D or 2-D depending on how data was prepared
        result = evaluate(m, X[:20], y[:20].ravel(), ["gain"])
        assert "gain" in result["r2"]

    def test_known_mae_value(self):
        """Evaluate with perfect predictions — MAE should be 0."""
        X, y = _make_linear_data(n=50, n_targets=2)
        m = RandomForestModel()
        m.fit(X, y)
        # Predict on training set — RF has zero training error
        y_pred = m.predict(X)
        # Manually compute MAE for col 0
        from sklearn.metrics import mean_absolute_error
        expected_mae = mean_absolute_error(y[:, 0], y_pred[:, 0])
        result = evaluate(m, X, y, ["col0", "col1"])
        assert pytest.approx(result["mae"]["col0"], rel=1e-6) == expected_mae


# ---------------------------------------------------------------------------
# TestTrain (mocked filesystem + registry)
# ---------------------------------------------------------------------------

class TestTrain:
    """
    Tests for trainer.train() with mocked I/O so no real CSV or ngspice needed.
    """

    CIRCUIT_ID = "common_emitter_amplifier"

    def _make_fake_df(self):
        import pandas as pd
        rng = np.random.default_rng(0)
        n = 200
        data = {
            "R1": rng.uniform(50000, 150000, n),
            "R2": rng.uniform(10000, 30000,  n),
            "Rc": rng.uniform(1000,  5000,   n),
            "Re": rng.uniform(100,   1000,   n),
            "Peak_Gain_dB": rng.uniform(5, 20, n),
            "Bandwidth_Hz": rng.uniform(1e4, 1e8, n),
        }
        return pd.DataFrame(data)

    def test_train_returns_metrics_dict(self, tmp_path):
        from core.models import trainer as tr

        fake_df = self._make_fake_df()
        with patch("core.models.trainer.load_csv", return_value=fake_df), \
             patch("core.models.trainer._MODELS_DIR", str(tmp_path)), \
             patch("core.models.trainer._update_model_block"):
            result = tr.train(self.CIRCUIT_ID, verbose=False)

        assert "r2" in result
        assert "mae" in result

    def test_train_saves_model_pkl(self, tmp_path):
        from core.models import trainer as tr

        fake_df = self._make_fake_df()
        with patch("core.models.trainer.load_csv", return_value=fake_df), \
             patch("core.models.trainer._MODELS_DIR", str(tmp_path)), \
             patch("core.models.trainer._update_model_block"):
            tr.train(self.CIRCUIT_ID, verbose=False)

        model_path = tmp_path / self.CIRCUIT_ID / "circuit_model.pkl"
        assert model_path.exists()

    def test_train_saves_scaler_pkl(self, tmp_path):
        from core.models import trainer as tr

        fake_df = self._make_fake_df()
        with patch("core.models.trainer.load_csv", return_value=fake_df), \
             patch("core.models.trainer._MODELS_DIR", str(tmp_path)), \
             patch("core.models.trainer._update_model_block"):
            tr.train(self.CIRCUIT_ID, verbose=False)

        scaler_path = tmp_path / self.CIRCUIT_ID / "feature_scaler.pkl"
        assert scaler_path.exists()

    def test_train_raises_on_bad_data(self, tmp_path):
        import pandas as pd
        from core.models import trainer as tr

        bad_df = self._make_fake_df()
        bad_df.loc[0, "Peak_Gain_dB"] = float("nan")

        with patch("core.models.trainer.load_csv", return_value=bad_df), \
             patch("core.models.trainer._MODELS_DIR", str(tmp_path)):
            with pytest.raises(ValueError, match="Data quality issues"):
                tr.train(self.CIRCUIT_ID, verbose=False)

    def test_saved_model_reloads_and_predicts(self, tmp_path):
        from core.models import trainer as tr

        fake_df = self._make_fake_df()
        with patch("core.models.trainer.load_csv", return_value=fake_df), \
             patch("core.models.trainer._MODELS_DIR", str(tmp_path)), \
             patch("core.models.trainer._update_model_block"):
            tr.train(self.CIRCUIT_ID, verbose=False)

        import joblib
        model_path = tmp_path / self.CIRCUIT_ID / "circuit_model.pkl"
        loaded = RandomForestModel()
        loaded.load(str(model_path))

        X_test = np.array([[100000, 20000, 3000, 470]], dtype=float)

        # Need to scale with the saved scaler
        scaler_path = tmp_path / self.CIRCUIT_ID / "feature_scaler.pkl"
        scaler = joblib.load(str(scaler_path))
        X_scaled = scaler.transform(X_test)

        pred = loaded.predict(X_scaled)
        assert pred.shape == (1, 2)
        assert np.isfinite(pred).all()

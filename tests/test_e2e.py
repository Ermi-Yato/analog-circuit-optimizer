"""
Phase 13 — End-to-end pipeline test.

Covers the full train → optimize workflow without ngspice.
Uses a synthetic CSV that mirrors the common_emitter_amplifier schema,
so the tests are fully self-contained regardless of how many real
simulation rows exist on disk.
"""
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

import registry.circuit_registry as reg
from core.models.trainer import train, evaluate
from core.models.random_forest import RandomForestModel
from core.dataset.preprocessor import fit_transform
from core.optimization.genetic_algorithm import optimize

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CIRCUIT_ID   = "common_emitter_amplifier"


# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------

def _make_synthetic_csv(path: str, n: int = 400, seed: int = 0):
    """
    Write a synthetic common_emitter_amplifier CSV to *path*.

    Physics approximation (rough):
      Peak_Gain_dB  ~ Rc / Re  (collector/emitter resistor ratio)
      Bandwidth_Hz  ~ 1e8 / Rc  (inversely proportional to collector resistor)
    """
    rng = np.random.default_rng(seed)
    R1  = rng.uniform(50_000, 150_000, n)
    R2  = rng.uniform(10_000,  30_000, n)
    Rc  = rng.uniform(1_000,   5_000,  n)
    Re  = rng.uniform(100,     1_000,  n)

    gain_db = 20 * np.log10(Rc / Re + 1e-6) + rng.normal(0, 0.5, n)
    bw_hz   = 1e8 / Rc * 1_000 + rng.normal(0, 500, n)

    df = pd.DataFrame({
        "R1": R1, "R2": R2, "Rc": Rc, "Re": Re,
        "Peak_Gain_dB": gain_db,
        "Bandwidth_Hz": bw_hz,
    })
    df.to_csv(path, index=False)
    return df


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory):
    """Write once per module, return the file path."""
    p = tmp_path_factory.mktemp("data") / f"{_CIRCUIT_ID}_dataset.csv"
    _make_synthetic_csv(str(p), n=400)
    return str(p)


@pytest.fixture(scope="module")
def trained_rf(synthetic_csv, tmp_path_factory):
    """Train a RandomForest on the synthetic CSV; return (models_dir, metrics)."""
    models_dir = str(tmp_path_factory.mktemp("models"))
    with patch("core.models.trainer._MODELS_DIR", models_dir), \
         patch("core.models.trainer._update_model_block"), \
         patch("core.dataset.preprocessor.load_csv",
               side_effect=lambda cid, **kw: pd.read_csv(synthetic_csv)):
        metrics = train(_CIRCUIT_ID, model_type="random_forest", verbose=False)
    return models_dir, metrics


# ---------------------------------------------------------------------------
# 1. Training pipeline
# ---------------------------------------------------------------------------

class TestTrainPipeline:
    def test_train_returns_required_keys(self, synthetic_csv, tmp_path):
        with patch("core.models.trainer._MODELS_DIR", str(tmp_path / "m")), \
             patch("core.models.trainer._update_model_block"), \
             patch("core.models.trainer.load_csv",
                   side_effect=lambda cid, **kw: pd.read_csv(synthetic_csv)):
            result = train(_CIRCUIT_ID, model_type="random_forest", verbose=False)

        assert "r2"  in result
        assert "mae" in result
        assert "n_train"    in result
        assert "model_type" in result

    def test_train_r2_above_threshold(self, synthetic_csv, tmp_path):
        """R² must be positive on synthetic data — any useful model beats the mean."""
        with patch("core.models.trainer._MODELS_DIR", str(tmp_path / "m")), \
             patch("core.models.trainer._update_model_block"), \
             patch("core.models.trainer.load_csv",
                   side_effect=lambda cid, **kw: pd.read_csv(synthetic_csv)):
            result = train(_CIRCUIT_ID, model_type="random_forest", verbose=False)

        for name, r2 in result["r2"].items():
            assert r2 > 0.0, f"{name}: R²={r2:.4f} — model worse than baseline mean"

    def test_train_saves_pkl_files(self, synthetic_csv, tmp_path):
        models_dir = str(tmp_path / "m")
        with patch("core.models.trainer._MODELS_DIR", models_dir), \
             patch("core.models.trainer._update_model_block"), \
             patch("core.models.trainer.load_csv",
                   side_effect=lambda cid, **kw: pd.read_csv(synthetic_csv)):
            train(_CIRCUIT_ID, model_type="random_forest", verbose=False)

        assert os.path.isfile(os.path.join(models_dir, _CIRCUIT_ID, "circuit_model.pkl"))
        assert os.path.isfile(os.path.join(models_dir, _CIRCUIT_ID, "feature_scaler.pkl"))

    def test_mlp_train_completes(self, synthetic_csv, tmp_path):
        with patch("core.models.trainer._MODELS_DIR", str(tmp_path / "m")), \
             patch("core.models.trainer._update_model_block"), \
             patch("core.models.trainer.load_csv",
                   side_effect=lambda cid, **kw: pd.read_csv(synthetic_csv)):
            result = train(_CIRCUIT_ID, model_type="mlp", verbose=False)

        assert result["model_type"] == "mlp"
        # R² can be negative on tiny synthetic data; just verify the pipeline ran
        assert all(isinstance(r, float) for r in result["r2"].values())

    def test_auto_picks_a_model(self, synthetic_csv, tmp_path):
        with patch("core.models.trainer._MODELS_DIR", str(tmp_path / "m")), \
             patch("core.models.trainer._update_model_block"), \
             patch("core.models.trainer.load_csv",
                   side_effect=lambda cid, **kw: pd.read_csv(synthetic_csv)):
            result = train(_CIRCUIT_ID, model_type="auto", verbose=False)

        assert result["model_type"] in ("random_forest", "mlp")


# ---------------------------------------------------------------------------
# 2. Optimizer pipeline
# ---------------------------------------------------------------------------

def _circuit_with_model(models_dir: str) -> dict:
    """Return circuit dict patched to use temp model paths."""
    circuit = json.loads(json.dumps(reg.get(_CIRCUIT_ID)))
    circuit["model"] = {
        "surrogate_path": os.path.join(
            models_dir, _CIRCUIT_ID, "circuit_model.pkl"
        ).replace("\\", "/"),
        "scaler_path": os.path.join(
            models_dir, _CIRCUIT_ID, "feature_scaler.pkl"
        ).replace("\\", "/"),
        "model_type": "random_forest",
        "log_metrics": [],
    }
    return circuit


class TestOptimizerPipeline:
    def test_result_has_expected_keys(self, trained_rf):
        models_dir, _ = trained_rf
        circuit = _circuit_with_model(models_dir)

        with patch("core.optimization.genetic_algorithm._PROJECT_ROOT", _PROJECT_ROOT), \
             patch("core.optimization.genetic_algorithm.reg") as m:
            m.get.return_value = circuit
            result = optimize(
                _CIRCUIT_ID,
                {"Peak_Gain_dB": 20.0, "Bandwidth_Hz": 1e8},
                n_generations=15, pop_size=20,
            )

        assert "best_params"    in result
        assert "best_score"     in result
        assert "best_predicted" in result
        assert "population"     in result

    def test_bounds_respected(self, trained_rf):
        models_dir, _ = trained_rf
        circuit = _circuit_with_model(models_dir)
        param_bounds = {p["name"]: (p["min"], p["max"]) for p in reg.get(_CIRCUIT_ID)["parameters"]}

        with patch("core.optimization.genetic_algorithm._PROJECT_ROOT", _PROJECT_ROOT), \
             patch("core.optimization.genetic_algorithm.reg") as m:
            m.get.return_value = circuit
            result = optimize(
                _CIRCUIT_ID,
                {"Peak_Gain_dB": 20.0, "Bandwidth_Hz": 1e8},
                n_generations=15, pop_size=20,
            )

        for name, val in result["best_params"].items():
            lo, hi = param_bounds[name]
            assert lo <= val <= hi, f"{name}={val:.4g} outside [{lo}, {hi}]"

    def test_score_non_negative(self, trained_rf):
        models_dir, _ = trained_rf
        circuit = _circuit_with_model(models_dir)

        with patch("core.optimization.genetic_algorithm._PROJECT_ROOT", _PROJECT_ROOT), \
             patch("core.optimization.genetic_algorithm.reg") as m:
            m.get.return_value = circuit
            result = optimize(
                _CIRCUIT_ID,
                {"Peak_Gain_dB": 20.0},
                n_generations=15, pop_size=20,
            )

        assert result["best_score"] >= 0.0

    def test_population_sorted_by_score(self, trained_rf):
        models_dir, _ = trained_rf
        circuit = _circuit_with_model(models_dir)

        with patch("core.optimization.genetic_algorithm._PROJECT_ROOT", _PROJECT_ROOT), \
             patch("core.optimization.genetic_algorithm.reg") as m:
            m.get.return_value = circuit
            result = optimize(
                _CIRCUIT_ID,
                {"Peak_Gain_dB": 20.0, "Bandwidth_Hz": 1e8},
                n_generations=15, pop_size=20,
            )

        scores = [s for _, s, _ in result["population"]]
        assert scores == sorted(scores)

    def test_progress_callback_called_per_generation(self, trained_rf):
        models_dir, _ = trained_rf
        circuit = _circuit_with_model(models_dir)
        calls = []

        with patch("core.optimization.genetic_algorithm._PROJECT_ROOT", _PROJECT_ROOT), \
             patch("core.optimization.genetic_algorithm.reg") as m:
            m.get.return_value = circuit
            optimize(
                _CIRCUIT_ID,
                {"Peak_Gain_dB": 20.0},
                n_generations=8, pop_size=20,
                progress_callback=lambda g, s: calls.append(g),
            )

        assert calls == list(range(1, 9))

    def test_optimizer_improves_over_generations(self, trained_rf):
        """Best score at generation N should be <= best score at generation 1."""
        models_dir, _ = trained_rf
        circuit = _circuit_with_model(models_dir)
        scores = []

        with patch("core.optimization.genetic_algorithm._PROJECT_ROOT", _PROJECT_ROOT), \
             patch("core.optimization.genetic_algorithm.reg") as m:
            m.get.return_value = circuit
            optimize(
                _CIRCUIT_ID,
                {"Peak_Gain_dB": 20.0, "Bandwidth_Hz": 1e8},
                n_generations=30, pop_size=40,
                progress_callback=lambda g, s: scores.append(s),
            )

        # Score should not increase monotonically — best at end <= best at start
        assert scores[-1] <= scores[0] + 1e-9, (
            f"Optimizer did not improve: start={scores[0]:.4f}, end={scores[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. Model save/load consistency
# ---------------------------------------------------------------------------

class TestModelSaveLoad:
    def test_rf_reload_gives_identical_predictions(self, tmp_path):
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 1, (200, 4))
        y = rng.uniform(0, 10, (200, 2))

        model = RandomForestModel()
        model.fit(X, y)

        path = str(tmp_path / "model.pkl")
        model.save(path)

        model2 = RandomForestModel()
        model2.load(path)

        np.testing.assert_array_almost_equal(
            model.predict(X[:10]),
            model2.predict(X[:10]),
            decimal=5,
        )

    def test_evaluate_returns_r2_and_mae(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(0, 1, (100, 3))
        y = X @ np.array([[2.0], [1.0], [3.0]]) + rng.normal(0, 0.01, (100, 1))

        model = RandomForestModel()
        model.fit(X[:80], y[:80])

        result = evaluate(model, X[80:], y[80:], ["metric_a"])
        assert "r2"  in result
        assert "mae" in result
        assert "metric_a" in result["r2"]
        assert result["r2"]["metric_a"] > 0.5

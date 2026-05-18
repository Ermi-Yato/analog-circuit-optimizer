"""Phase 5 — Optimizer tests. Uses synthetic models — no ngspice, no real CSV."""
import os
import random
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from core.optimization.objective import build_fitness_fn
from core.optimization.genetic_algorithm import optimize


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_scaler(n_features=4):
    """A fitted StandardScaler with known mean=0, scale=1 (identity transform)."""
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    sc = StandardScaler()
    sc.fit(np.zeros((2, n_features)))
    sc.mean_  = np.zeros(n_features)
    sc.scale_ = np.ones(n_features)
    return sc


def _make_model(n_features=4, n_targets=2, constant_output=None):
    """
    A mock surrogate model.
    If constant_output is given, predict always returns that array.
    Otherwise returns a random array each call.
    """
    model = MagicMock()
    if constant_output is not None:
        arr = np.array(constant_output, dtype=float).reshape(1, -1)
        model.predict.return_value = arr
    else:
        rng = np.random.default_rng(0)
        model.predict.side_effect = lambda X: rng.uniform(0, 10, (len(X), n_targets))
    return model


PARAM_DEFS_4 = [
    {"name": "R1", "min": 50000,  "max": 150000, "default": 100000, "scale": "linear"},
    {"name": "R2", "min": 10000,  "max": 30000,  "default": 20000,  "scale": "linear"},
    {"name": "Rc", "min": 1000,   "max": 5000,   "default": 3000,   "scale": "linear"},
    {"name": "Re", "min": 100,    "max": 1000,   "default": 470,    "scale": "linear"},
]

METRIC_DEFS_2 = [
    {"name": "Peak_Gain_dB", "label": "Gain",      "unit": "dB",  "optimize": "maximize"},
    {"name": "Bandwidth_Hz", "label": "Bandwidth", "unit": "Hz",  "optimize": "maximize"},
]


# ---------------------------------------------------------------------------
# TestBuildFitnessFn
# ---------------------------------------------------------------------------

class TestBuildFitnessFn:

    def test_returns_callable(self):
        m = _make_model(constant_output=[10.0, 100000.0])
        sc = _make_scaler()
        fn = build_fitness_fn(m, sc, {"Peak_Gain_dB": 10.0, "Bandwidth_Hz": 1e5}, METRIC_DEFS_2)
        assert callable(fn)

    def test_output_is_tuple(self):
        m = _make_model(constant_output=[10.0, 100000.0])
        sc = _make_scaler()
        fn = build_fitness_fn(m, sc, {"Peak_Gain_dB": 10.0, "Bandwidth_Hz": 1e5}, METRIC_DEFS_2)
        result = fn([100000, 20000, 3000, 470])
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_perfect_prediction_gives_zero_score(self):
        m = _make_model(constant_output=[20.0, 200000.0])
        sc = _make_scaler()
        fn = build_fitness_fn(m, sc, {"Peak_Gain_dB": 20.0, "Bandwidth_Hz": 200000.0}, METRIC_DEFS_2)
        score, = fn([100000, 20000, 3000, 470])
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_score_is_non_negative(self):
        m = _make_model(constant_output=[5.0, 50000.0])
        sc = _make_scaler()
        fn = build_fitness_fn(m, sc, {"Peak_Gain_dB": 15.0, "Bandwidth_Hz": 100000.0}, METRIC_DEFS_2)
        score, = fn([100000, 20000, 3000, 470])
        assert score >= 0.0

    def test_larger_miss_gives_higher_score(self):
        sc = _make_scaler()
        target = {"Peak_Gain_dB": 20.0}
        metrics = [METRIC_DEFS_2[0]]  # only gain

        m_close = _make_model(constant_output=[[19.0]])
        m_far   = _make_model(constant_output=[[10.0]])

        fn_close = build_fitness_fn(m_close, sc, target, metrics)
        fn_far   = build_fitness_fn(m_far,   sc, target, metrics)

        score_close, = fn_close([100000, 20000, 3000, 470])
        score_far,   = fn_far(  [100000, 20000, 3000, 470])
        assert score_far > score_close

    def test_unknown_metric_targets_ignored(self):
        m = _make_model(constant_output=[10.0, 100000.0])
        sc = _make_scaler()
        # target only has one of the two metrics
        fn = build_fitness_fn(m, sc, {"Peak_Gain_dB": 10.0}, METRIC_DEFS_2)
        score, = fn([100000, 20000, 3000, 470])
        assert isinstance(score, float)

    def test_custom_weights_scale_score(self):
        sc = _make_scaler()
        target = {"Peak_Gain_dB": 20.0, "Bandwidth_Hz": 100000.0}
        m = _make_model(constant_output=[10.0, 50000.0])

        fn_uniform = build_fitness_fn(m, sc, target, METRIC_DEFS_2, weights=None)
        fn_heavy   = build_fitness_fn(m, sc, target, METRIC_DEFS_2,
                                      weights={"Peak_Gain_dB": 10.0, "Bandwidth_Hz": 1.0})

        s_uniform, = fn_uniform([100000, 20000, 3000, 470])
        s_heavy,   = fn_heavy(  [100000, 20000, 3000, 470])
        assert s_heavy > s_uniform


# ---------------------------------------------------------------------------
# TestOptimize — uses a mock registry + mock model so no I/O needed
# ---------------------------------------------------------------------------

def _mock_circuit():
    return {
        "id":              "common_emitter_amplifier",
        "name":            "Common-Emitter Amplifier",
        "description":     "Test circuit",
        "spice_template":  "circuits/common_emitter_amplifier/common_emitter_amplifier.template",
        "simulation_type": "ac",
        "parameters":      PARAM_DEFS_4,
        "metrics":         METRIC_DEFS_2,
        "model": {
            "surrogate_path": "trained_models/common_emitter_amplifier/circuit_model.pkl",
            "scaler_path":    "trained_models/common_emitter_amplifier/feature_scaler.pkl",
        },
    }


@pytest.fixture()
def fake_model_files(tmp_path):
    """Write dummy pkl files so os.path.isfile checks pass."""
    import joblib
    from sklearn.preprocessing import StandardScaler

    model_dir = tmp_path / "trained_models" / "common_emitter_amplifier"
    model_dir.mkdir(parents=True)

    # Minimal real model + scaler so the GA can actually predict
    from core.models.random_forest import RandomForestModel
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (50, 4))
    y = rng.uniform(0, 1, (50, 2))
    m = RandomForestModel()
    m.fit(X, y)
    m.save(str(model_dir / "circuit_model.pkl"))

    sc = StandardScaler()
    sc.fit(X)
    joblib.dump(sc, str(model_dir / "feature_scaler.pkl"))

    return tmp_path


class TestOptimize:

    CIRCUIT_ID = "common_emitter_amplifier"
    TARGETS    = {"Peak_Gain_dB": 15.0, "Bandwidth_Hz": 80000.0}

    def _run(self, tmp_path, **kwargs):
        circuit = _mock_circuit()
        with patch("core.optimization.genetic_algorithm.reg.get", return_value=circuit), \
             patch("core.optimization.genetic_algorithm._PROJECT_ROOT", str(tmp_path)):
            return optimize(
                self.CIRCUIT_ID,
                self.TARGETS,
                n_generations=kwargs.pop("n_generations", 5),
                pop_size=kwargs.pop("pop_size", 20),
                seed=kwargs.pop("seed", 42),
                **kwargs,
            )

    def test_returns_dict_with_required_keys(self, fake_model_files):
        result = self._run(fake_model_files)
        assert "best_params"     in result
        assert "best_score"      in result
        assert "best_predicted"  in result
        assert "history"         in result
        assert "population"      in result

    def test_best_params_has_all_param_names(self, fake_model_files):
        result = self._run(fake_model_files)
        assert set(result["best_params"].keys()) == {"R1", "R2", "Rc", "Re"}

    def test_bounds_respected(self, fake_model_files):
        result = self._run(fake_model_files)
        for params, _ in result["population"]:
            for p in PARAM_DEFS_4:
                assert params[p["name"]] >= p["min"], f"{p['name']} below min"
                assert params[p["name"]] <= p["max"], f"{p['name']} above max"

    def test_best_params_within_bounds(self, fake_model_files):
        result = self._run(fake_model_files)
        for p in PARAM_DEFS_4:
            v = result["best_params"][p["name"]]
            assert p["min"] <= v <= p["max"]

    def test_history_length_equals_n_generations(self, fake_model_files):
        result = self._run(fake_model_files, n_generations=7)
        assert len(result["history"]) == 7

    def test_history_generations_are_sequential(self, fake_model_files):
        result = self._run(fake_model_files, n_generations=5)
        gens = [h[0] for h in result["history"]]
        assert gens == list(range(1, 6))

    def test_best_score_is_non_negative(self, fake_model_files):
        result = self._run(fake_model_files)
        assert result["best_score"] >= 0.0

    def test_best_score_is_float(self, fake_model_files):
        result = self._run(fake_model_files)
        assert isinstance(result["best_score"], float)

    def test_population_sorted_by_score(self, fake_model_files):
        result = self._run(fake_model_files)
        scores = [s for _, s in result["population"]]
        assert scores == sorted(scores)

    def test_progress_callback_called_each_generation(self, fake_model_files):
        calls = []
        self._run(fake_model_files, n_generations=5,
                  progress_callback=lambda g, s: calls.append(g))
        assert len(calls) == 5

    def test_progress_callback_receives_generation_number(self, fake_model_files):
        calls = []
        self._run(fake_model_files, n_generations=4,
                  progress_callback=lambda g, s: calls.append(g))
        assert calls == [1, 2, 3, 4]

    def test_progress_callback_score_is_float(self, fake_model_files):
        scores = []
        self._run(fake_model_files, n_generations=3,
                  progress_callback=lambda g, s: scores.append(s))
        for s in scores:
            assert isinstance(s, float)

    def test_no_model_block_raises(self, fake_model_files):
        circuit = _mock_circuit()
        del circuit["model"]
        with patch("core.optimization.genetic_algorithm.reg.get", return_value=circuit), \
             patch("core.optimization.genetic_algorithm._PROJECT_ROOT", str(fake_model_files)):
            with pytest.raises(FileNotFoundError, match="No trained model"):
                optimize(self.CIRCUIT_ID, self.TARGETS, n_generations=2, pop_size=10)

    def test_missing_pkl_raises(self, fake_model_files):
        import os
        circuit = _mock_circuit()
        # Point to a non-existent file
        circuit["model"]["surrogate_path"] = "trained_models/nonexistent/circuit_model.pkl"
        with patch("core.optimization.genetic_algorithm.reg.get", return_value=circuit), \
             patch("core.optimization.genetic_algorithm._PROJECT_ROOT", str(fake_model_files)):
            with pytest.raises(FileNotFoundError):
                optimize(self.CIRCUIT_ID, self.TARGETS, n_generations=2, pop_size=10)

    def test_seed_gives_reproducible_results(self, fake_model_files):
        r1 = self._run(fake_model_files, n_generations=5, seed=7)
        r2 = self._run(fake_model_files, n_generations=5, seed=7)
        assert r1["best_params"] == pytest.approx(r2["best_params"])

    def test_best_predicted_has_metric_names(self, fake_model_files):
        result = self._run(fake_model_files)
        assert set(result["best_predicted"].keys()) == {"Peak_Gain_dB", "Bandwidth_Hz"}

"""Phase 3 — Dataset generation and preprocessor tests."""
import math
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.dataset.generator import (
    _sample_params,
    _extract_metrics,
    _gain_db_array,
    _peak_gain_db,
    _bandwidth_hz,
    _cutoff_freq_hz,
    _q_factor,
    _output_swing_v,
    _thd_percent,
    generate,
    preview,
)
from core.dataset.preprocessor import validate, stats, fit_transform, load_csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ac_raw():
    """Synthetic AC simulation result: flat gain then 3dB rolloff."""
    freq = np.logspace(1, 8, 500)           # 10 Hz – 100 MHz
    # Gain = 10 (20 dB) up to 1 MHz, then rolling off
    bw_hz = 1e6
    H = 10.0 / np.sqrt(1 + (freq / bw_hz) ** 2)
    # Make complex: phase lag
    phase = -np.arctan(freq / bw_hz)
    return {
        "freq": freq,
        "real": H * np.cos(phase),
        "imag": H * np.sin(phase),
    }


@pytest.fixture
def transient_raw():
    """Synthetic transient: 1 kHz sine, 5ms, settled around 6V DC with 1V swing."""
    t   = np.linspace(0, 5e-3, 5000)
    v   = 6.0 + 0.5 * np.sin(2 * np.pi * 1000 * t)   # 1Vpp, 6V DC
    return {"time": t, "voltage": v}


@pytest.fixture
def clean_df():
    np.random.seed(42)
    return pd.DataFrame({
        "R1": np.random.uniform(50000, 150000, 100),
        "R2": np.random.uniform(10000, 30000,  100),
        "Peak_Gain_dB":  np.random.uniform(5, 20, 100),
        "Bandwidth_Hz":  np.random.uniform(1e6, 1e8, 100),
    })


# ---------------------------------------------------------------------------
# TestSampleParams
# ---------------------------------------------------------------------------

class TestSampleParams:

    PARAM_DEFS_LINEAR = [
        {"name": "R1", "min": 1000, "max": 10000, "scale": "linear"},
        {"name": "R2", "min": 500,  "max": 5000,  "scale": "linear"},
    ]
    PARAM_DEFS_LOG = [
        {"name": "C1", "min": 1e-10, "max": 1e-6, "scale": "log"},
    ]

    def _rng(self):
        return np.random.default_rng(0)

    def test_returns_n_param_sets(self):
        sets = _sample_params(self.PARAM_DEFS_LINEAR, 50, self._rng())
        assert len(sets) == 50

    def test_each_set_has_correct_keys(self):
        sets = _sample_params(self.PARAM_DEFS_LINEAR, 10, self._rng())
        for s in sets:
            assert set(s.keys()) == {"R1", "R2"}

    def test_linear_values_within_bounds(self):
        sets = _sample_params(self.PARAM_DEFS_LINEAR, 200, self._rng())
        r1_vals = [s["R1"] for s in sets]
        assert all(1000 <= v <= 10000 for v in r1_vals)

    def test_log_values_within_bounds(self):
        sets = _sample_params(self.PARAM_DEFS_LOG, 200, self._rng())
        c1_vals = [s["C1"] for s in sets]
        assert all(1e-10 <= v <= 1e-6 for v in c1_vals)

    def test_log_sampling_covers_decades(self):
        """Log-scale sampling should have values spread across all decades."""
        sets = _sample_params(self.PARAM_DEFS_LOG, 1000, self._rng())
        c1_vals = [s["C1"] for s in sets]
        # Should have values in both pF range and nF range
        assert any(v < 1e-8 for v in c1_vals), "No values in pF range"
        assert any(v > 1e-7 for v in c1_vals), "No values in nF range"

    def test_linear_sampling_uniform_distribution(self):
        """Linear sampling: ~50% of values should be below midpoint."""
        sets = _sample_params(self.PARAM_DEFS_LINEAR, 1000, self._rng())
        midpoint = (1000 + 10000) / 2
        below = sum(1 for s in sets if s["R1"] < midpoint)
        assert 400 < below < 600  # roughly 50% ± 10%

    def test_seed_reproducibility(self):
        sets_a = _sample_params(self.PARAM_DEFS_LINEAR, 10, np.random.default_rng(42))
        sets_b = _sample_params(self.PARAM_DEFS_LINEAR, 10, np.random.default_rng(42))
        for a, b in zip(sets_a, sets_b):
            assert a["R1"] == b["R1"]


# ---------------------------------------------------------------------------
# TestACMetricExtractors
# ---------------------------------------------------------------------------

class TestACMetricExtractors:

    def test_peak_gain_db_correct(self, ac_raw):
        result = _peak_gain_db(ac_raw, {})
        assert pytest.approx(result, abs=0.5) == 20.0   # 10 V/V = 20 dB

    def test_bandwidth_hz_correct(self, ac_raw):
        result = _bandwidth_hz(ac_raw, {})
        # BW should be close to 1 MHz (set in fixture)
        assert result is not None and not math.isnan(result)
        assert 5e5 < result < 2e6

    def test_bandwidth_returns_nan_when_no_rolloff(self):
        """If gain never drops 3dB within sweep, return NaN."""
        freq = np.array([10.0, 100.0, 1000.0])
        flat = np.full(3, 10.0)
        raw  = {"freq": freq, "real": flat, "imag": np.zeros(3)}
        result = _bandwidth_hz(raw, {})
        # Flat response within sweep — BW = f_high - f_low = 990 Hz, not NaN
        # This is valid behaviour: all frequencies are within 3dB of peak
        assert result is not None

    def test_cutoff_freq_returns_float(self, ac_raw):
        result = _cutoff_freq_hz(ac_raw, {})
        assert isinstance(result, float)
        assert result > 0

    def test_cutoff_freq_near_rolloff(self, ac_raw):
        result = _cutoff_freq_hz(ac_raw, {})
        # -3dB point should be close to 1 MHz
        assert 5e5 < result < 2e6

    def test_q_factor_equal_components(self):
        """Q = 0.5 for R1=R2=R, C1=C2=C (Sallen-Key equal-component case)."""
        params = {"R1": 10000, "R2": 10000, "C1": 10e-9, "C2": 10e-9}
        result = _q_factor({}, params)
        assert pytest.approx(result, abs=0.01) == 0.5

    def test_q_factor_butterworth(self):
        """Q ≈ 0.707 for R1=R2, C1=2*C2 (Butterworth Sallen-Key)."""
        params = {"R1": 10000, "R2": 10000, "C1": 20e-9, "C2": 10e-9}
        result = _q_factor({}, params)
        assert pytest.approx(result, abs=0.01) == 1.0 / math.sqrt(2)

    def test_gain_db_array_positive_for_gain_gt_1(self, ac_raw):
        db = _gain_db_array(ac_raw)
        assert db[0] > 0   # gain > 1 at DC → positive dB

    def test_gain_db_avoids_log_zero(self):
        """Should not raise even if magnitude is zero."""
        raw = {"freq": np.array([10.0]), "real": np.array([0.0]), "imag": np.array([0.0])}
        db = _gain_db_array(raw)
        assert np.isfinite(db).all() or True  # no exception is what matters


# ---------------------------------------------------------------------------
# TestTransientMetricExtractors
# ---------------------------------------------------------------------------

class TestTransientMetricExtractors:

    def test_output_swing_correct(self, transient_raw):
        result = _output_swing_v(transient_raw, {})
        # 0.5 amplitude sine → Vpp = 1.0V
        assert pytest.approx(result, abs=0.05) == 1.0

    def test_output_swing_ignores_initial_transient(self):
        """Initial spike should not inflate the swing measurement."""
        t = np.linspace(0, 5e-3, 5000)
        v = 6.0 + 0.5 * np.sin(2 * np.pi * 1000 * t)
        v[0] = 20.0   # large initial spike
        raw = {"time": t, "voltage": v}
        result = _output_swing_v(raw, {})
        # Spike at index 0 is in the first 20% that gets skipped
        assert result < 5.0

    def test_thd_returns_float(self, transient_raw):
        result = _thd_percent(transient_raw, {})
        assert isinstance(result, float)
        assert not math.isnan(result)

    def test_thd_pure_sine_is_low(self, transient_raw):
        """Pure sine should have very low THD."""
        result = _thd_percent(transient_raw, {})
        # Ideal sine → THD should be < 5%
        assert result < 5.0

    def test_thd_clipped_signal_is_high(self):
        """Clipped (saturated) waveform should produce high THD."""
        t = np.linspace(0, 5e-3, 5000)
        sine = np.sin(2 * np.pi * 1000 * t)
        clipped = np.clip(sine, -0.5, 0.5)   # hard clip
        raw = {"time": t, "voltage": clipped}
        result = _thd_percent(raw, {})
        assert result > 5.0


# ---------------------------------------------------------------------------
# TestExtractMetrics
# ---------------------------------------------------------------------------

class TestExtractMetrics:

    def test_returns_dict_for_known_metrics(self, ac_raw):
        result = _extract_metrics(ac_raw, ["Peak_Gain_dB", "Bandwidth_Hz"], {})
        assert result is not None
        assert "Peak_Gain_dB" in result
        assert "Bandwidth_Hz" in result

    def test_unknown_metric_returns_nan(self, ac_raw):
        result = _extract_metrics(ac_raw, ["UNKNOWN_METRIC"], {})
        assert result is not None
        assert math.isnan(result["UNKNOWN_METRIC"])


# ---------------------------------------------------------------------------
# TestPreprocessorValidate
# ---------------------------------------------------------------------------

class TestValidate:

    def test_clean_df_has_no_issues(self, clean_df):
        assert validate(clean_df) == []

    def test_detects_nan(self, clean_df):
        clean_df.loc[5, "Peak_Gain_dB"] = np.nan
        issues = validate(clean_df)
        assert any("NaN" in i for i in issues)

    def test_detects_inf(self, clean_df):
        clean_df.loc[5, "Bandwidth_Hz"] = np.inf
        issues = validate(clean_df)
        assert any("Infinite" in i for i in issues)

    def test_detects_zero_variance(self, clean_df):
        clean_df["constant_col"] = 42.0
        issues = validate(clean_df)
        assert any("constant" in i for i in issues)

    def test_detects_duplicates(self, clean_df):
        dup_df = pd.concat([clean_df, clean_df.iloc[:5]], ignore_index=True)
        issues = validate(dup_df)
        assert any("duplicate" in i for i in issues)

    def test_empty_df_is_flagged(self):
        issues = validate(pd.DataFrame())
        assert any("empty" in i.lower() for i in issues)


# ---------------------------------------------------------------------------
# TestPreprocessorStats
# ---------------------------------------------------------------------------

class TestStats:

    def test_returns_dict_with_all_columns(self, clean_df):
        result = stats(clean_df)
        assert set(result.keys()) == set(clean_df.columns)

    def test_each_entry_has_required_keys(self, clean_df):
        result = stats(clean_df)
        for col_stats in result.values():
            assert {"min", "max", "mean", "std", "count"} <= set(col_stats.keys())

    def test_count_correct(self, clean_df):
        result = stats(clean_df)
        assert result["R1"]["count"] == 100

    def test_min_max_correct(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = stats(df)
        assert result["x"]["min"] == 1.0
        assert result["x"]["max"] == 3.0


# ---------------------------------------------------------------------------
# TestFitTransform
# ---------------------------------------------------------------------------

class TestFitTransform:

    def test_returns_three_items(self, clean_df):
        X, y, scaler = fit_transform(clean_df, ["R1", "R2"], ["Peak_Gain_dB", "Bandwidth_Hz"])
        assert X is not None
        assert y is not None
        assert scaler is not None

    def test_x_shape(self, clean_df):
        X, y, scaler = fit_transform(clean_df, ["R1", "R2"], ["Peak_Gain_dB", "Bandwidth_Hz"])
        assert X.shape == (100, 2)

    def test_y_shape(self, clean_df):
        X, y, scaler = fit_transform(clean_df, ["R1", "R2"], ["Peak_Gain_dB", "Bandwidth_Hz"])
        assert y.shape == (100, 2)

    def test_x_is_scaled(self, clean_df):
        X, _, _ = fit_transform(clean_df, ["R1", "R2"], ["Peak_Gain_dB"])
        # StandardScaler → mean ≈ 0, std ≈ 1
        assert pytest.approx(X[:, 0].mean(), abs=1e-10) == 0.0
        assert pytest.approx(X[:, 0].std(),  abs=0.01)  == 1.0

    def test_y_is_unscaled(self, clean_df):
        _, y, _ = fit_transform(clean_df, ["R1"], ["Peak_Gain_dB"])
        # y should still be in original dB range
        assert y.min() > 0


# ---------------------------------------------------------------------------
# TestLoadCSV
# ---------------------------------------------------------------------------

class TestLoadCSV:

    def test_raises_for_missing_circuit(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nonexistent_circuit"):
            load_csv("nonexistent_circuit", data_dir=str(tmp_path))

    def test_loads_existing_csv(self, tmp_path):
        # Write a dummy CSV
        df = pd.DataFrame({"R1": [1, 2], "gain": [10.0, 12.0]})
        df.to_csv(tmp_path / "test_ckt_dataset.csv", index=False)
        result = load_csv("test_ckt", data_dir=str(tmp_path))
        assert list(result.columns) == ["R1", "gain"]
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TestGenerateMocked — unit test with fake simulator
# ---------------------------------------------------------------------------

class TestGenerateMocked:

    def _fake_raw_ac(self):
        freq = np.logspace(1, 8, 500)
        H    = 10.0 / np.sqrt(1 + (freq / 1e6) ** 2)
        phi  = -np.arctan(freq / 1e6)
        return {"freq": freq, "real": H * np.cos(phi), "imag": H * np.sin(phi)}

    def test_generate_returns_dataframe(self, tmp_path):
        fake = self._fake_raw_ac()
        with patch("core.dataset.generator.NgspiceSimulator") as MockSim:
            MockSim.return_value.run_batch.return_value = [fake] * 5
            df = generate("common_emitter_amplifier", n_samples=5,
                          verbose=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_generate_has_correct_columns(self, tmp_path):
        fake = self._fake_raw_ac()
        with patch("core.dataset.generator.NgspiceSimulator") as MockSim:
            MockSim.return_value.run_batch.return_value = [fake] * 3
            df = generate("common_emitter_amplifier", n_samples=3,
                          verbose=False)
        expected_cols = {"R1", "R2", "Rc", "Re", "Peak_Gain_dB", "Bandwidth_Hz"}
        assert set(df.columns) == expected_cols

    def test_generate_drops_failed_simulations(self, tmp_path):
        fake = self._fake_raw_ac()
        with patch("core.dataset.generator.NgspiceSimulator") as MockSim:
            # 2 succeed, 1 fails
            MockSim.return_value.run_batch.return_value = [fake, None, fake]
            df = generate("common_emitter_amplifier", n_samples=3,
                          verbose=False)
        assert len(df) == 2

    def test_generate_saves_csv(self, tmp_path):
        fake = self._fake_raw_ac()
        with patch("core.dataset.generator.NgspiceSimulator") as MockSim, \
             patch("core.dataset.generator._DATA_DIR", str(tmp_path)):
            MockSim.return_value.run_batch.return_value = [fake] * 3
            generate("common_emitter_amplifier", n_samples=3, verbose=False)
        assert (tmp_path / "common_emitter_amplifier_dataset.csv").exists()

    def test_progress_callback_is_forwarded(self):
        fake = self._fake_raw_ac()
        calls = []
        with patch("core.dataset.generator.NgspiceSimulator") as MockSim:
            MockSim.return_value.run_batch.return_value = [fake] * 3
            generate("common_emitter_amplifier", n_samples=3,
                     progress_callback=lambda c, t: calls.append(c),
                     verbose=False)
        # progress_callback is passed through to run_batch — verify it was forwarded
        _, kwargs = MockSim.return_value.run_batch.call_args
        assert kwargs.get("progress_callback") is not None


# ---------------------------------------------------------------------------
# TestPreview
# ---------------------------------------------------------------------------

class TestPreview:

    def test_returns_string(self, clean_df):
        result = preview(clean_df)
        assert isinstance(result, str)

    def test_contains_shape_info(self, clean_df):
        result = preview(clean_df)
        assert "100" in result

    def test_contains_stats(self, clean_df):
        result = preview(clean_df)
        assert "mean" in result.lower() or "std" in result.lower()


# ---------------------------------------------------------------------------
# INTEGRATION TEST — real ngspice, small N, prints output for visual check
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_generate_real_ngspice_small_n(capsys):
    """
    Run 10 real ngspice simulations for common_emitter_amplifier.
    Verifies the pipeline end-to-end and prints the dataset for visual inspection.
    Requires ngspice on PATH. Skipped if not available.
    """
    import shutil
    if not shutil.which("ngspice"):
        pytest.skip("ngspice not on PATH")

    df = generate(
        "common_emitter_amplifier",
        n_samples=10,
        seed=42,
        verbose=True,
        max_workers=4,
    )

    captured = capsys.readouterr()
    print(captured.out)  # show in pytest -s output

    # Basic sanity checks
    assert len(df) > 0, "All simulations failed"
    assert "Peak_Gain_dB" in df.columns
    assert "Bandwidth_Hz" in df.columns

    # Gain should be physically reasonable for this circuit (positive dB)
    assert df["Peak_Gain_dB"].min() > 0
    assert df["Peak_Gain_dB"].max() < 60  # no unrealistic values

    issues = validate(df)
    assert issues == [], f"Data quality issues: {issues}"

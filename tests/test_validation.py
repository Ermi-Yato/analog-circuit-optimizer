"""Phase 6 — SPICE validation tests. Ngspice is fully mocked."""
import math
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from core.validation.spice_validator import validate, format_table


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CIRCUIT_ID = "common_emitter_amplifier"

PARAMS = {"R1": 100000.0, "R2": 20000.0, "Rc": 3000.0, "Re": 470.0}

PREDICTED = {"Peak_Gain_dB": 12.5, "Bandwidth_Hz": 98000.0}

# A minimal fake raw result that the metric extractors can work with
FAKE_RAW_AC = {
    "freq": np.linspace(10, 1e8, 500),
    "real": np.ones(500) * 4.0,   # |V| ≈ 4  → gain ≈ 12 dB
    "imag": np.zeros(500),
}


def _make_sim(return_value):
    """Return a mock NgspiceSimulator whose run_single returns return_value."""
    mock = MagicMock()
    mock.run_single.return_value = return_value
    return mock


# ---------------------------------------------------------------------------
# TestValidateSimulationOk
# ---------------------------------------------------------------------------

class TestValidateSimulationOk:

    def _run(self, raw=FAKE_RAW_AC, predicted=None):
        with patch(
            "core.validation.spice_validator.NgspiceSimulator",
            return_value=_make_sim(raw),
        ):
            return validate(CIRCUIT_ID, PARAMS, predicted=predicted)

    def test_returns_dict(self):
        assert isinstance(self._run(), dict)

    def test_simulation_ok_is_true(self):
        assert self._run()["simulation_ok"] is True

    def test_circuit_id_in_result(self):
        assert self._run()["circuit_id"] == CIRCUIT_ID

    def test_params_in_result(self):
        assert self._run()["params"] == PARAMS

    def test_metrics_list_length(self):
        result = self._run()
        # common_emitter_amplifier has 2 metrics
        assert len(result["metrics"]) == 2

    def test_metric_names_correct(self):
        result = self._run()
        names = {m["name"] for m in result["metrics"]}
        assert names == {"Peak_Gain_dB", "Bandwidth_Hz"}

    def test_actual_values_are_finite(self):
        result = self._run()
        for m in result["metrics"]:
            assert m["actual"] is not None
            assert math.isfinite(m["actual"])

    def test_predicted_none_when_not_provided(self):
        result = self._run(predicted=None)
        for m in result["metrics"]:
            assert m["predicted"] is None

    def test_predicted_values_filled_when_provided(self):
        result = self._run(predicted=PREDICTED)
        preds = {m["name"]: m["predicted"] for m in result["metrics"]}
        assert preds["Peak_Gain_dB"] == pytest.approx(12.5)
        assert preds["Bandwidth_Hz"] == pytest.approx(98000.0)

    def test_abs_error_computed_when_predicted_given(self):
        result = self._run(predicted=PREDICTED)
        for m in result["metrics"]:
            assert m["abs_error"] is not None
            assert m["abs_error"] >= 0.0

    def test_rel_error_computed_when_predicted_given(self):
        result = self._run(predicted=PREDICTED)
        for m in result["metrics"]:
            assert m["rel_error"] is not None
            assert m["rel_error"] >= 0.0

    def test_abs_error_none_when_no_predicted(self):
        result = self._run(predicted=None)
        for m in result["metrics"]:
            assert m["abs_error"] is None

    def test_rel_error_none_when_no_predicted(self):
        result = self._run(predicted=None)
        for m in result["metrics"]:
            assert m["rel_error"] is None

    def test_zero_abs_error_for_perfect_prediction(self):
        """If predicted == actual, abs_error should be ~0."""
        result = self._run(predicted=None)
        # Get actual values from result, then re-run with those as predicted
        actual_preds = {m["name"]: m["actual"] for m in result["metrics"]}
        result2 = self._run(predicted=actual_preds)
        for m in result2["metrics"]:
            assert m["abs_error"] == pytest.approx(0.0, abs=1e-10)

    def test_rel_error_is_percentage(self):
        """rel_error = |pred - actual| / |actual| * 100."""
        result = self._run(predicted=PREDICTED)
        for m in result["metrics"]:
            if m["actual"] and abs(m["actual"]) > 1e-12:
                expected = abs(m["predicted"] - m["actual"]) / abs(m["actual"]) * 100.0
                assert m["rel_error"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# TestValidateSimulationFailed
# ---------------------------------------------------------------------------

class TestValidateSimulationFailed:

    def _run(self, predicted=None):
        with patch(
            "core.validation.spice_validator.NgspiceSimulator",
            return_value=_make_sim(None),   # simulate ngspice failure
        ):
            return validate(CIRCUIT_ID, PARAMS, predicted=predicted)

    def test_simulation_ok_is_false(self):
        assert self._run()["simulation_ok"] is False

    def test_actual_values_are_none(self):
        result = self._run()
        for m in result["metrics"]:
            assert m["actual"] is None

    def test_abs_error_none_on_failure(self):
        result = self._run(predicted=PREDICTED)
        for m in result["metrics"]:
            assert m["abs_error"] is None

    def test_metrics_list_still_present(self):
        result = self._run()
        assert len(result["metrics"]) == 2

    def test_predicted_still_filled_on_failure(self):
        result = self._run(predicted=PREDICTED)
        preds = {m["name"]: m["predicted"] for m in result["metrics"]}
        assert preds["Peak_Gain_dB"] == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# TestFormatTable
# ---------------------------------------------------------------------------

class TestFormatTable:

    def _make_result(self, sim_ok=True, with_predicted=True):
        return {
            "circuit_id":    CIRCUIT_ID,
            "params":        PARAMS,
            "simulation_ok": sim_ok,
            "metrics": [
                {
                    "name":      "Peak_Gain_dB",
                    "predicted": 12.5   if with_predicted else None,
                    "actual":    12.31  if sim_ok else None,
                    "abs_error": 0.19   if (sim_ok and with_predicted) else None,
                    "rel_error": 1.54   if (sim_ok and with_predicted) else None,
                },
                {
                    "name":      "Bandwidth_Hz",
                    "predicted": 98000.0 if with_predicted else None,
                    "actual":    99725.0 if sim_ok else None,
                    "abs_error": 1725.0  if (sim_ok and with_predicted) else None,
                    "rel_error": 1.73    if (sim_ok and with_predicted) else None,
                },
            ],
        }

    def test_returns_string(self):
        assert isinstance(format_table(self._make_result()), str)

    def test_contains_metric_names(self):
        table = format_table(self._make_result())
        assert "Peak_Gain_dB" in table
        assert "Bandwidth_Hz" in table

    def test_contains_params(self):
        table = format_table(self._make_result())
        assert "R1" in table

    def test_contains_predicted_column(self):
        table = format_table(self._make_result(with_predicted=True))
        assert "Predicted" in table

    def test_contains_actual_column(self):
        table = format_table(self._make_result())
        assert "Actual" in table

    def test_warning_present_on_failure(self):
        table = format_table(self._make_result(sim_ok=False))
        assert "WARNING" in table or "failed" in table.lower()

    def test_no_warning_on_success(self):
        table = format_table(self._make_result(sim_ok=True))
        assert "WARNING" not in table

    def test_na_shown_for_none_values(self):
        table = format_table(self._make_result(sim_ok=False))
        assert "N/A" in table

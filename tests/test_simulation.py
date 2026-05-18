"""Phase 2 — Simulation layer tests. No ngspice binary required (fully mocked)."""
import os
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.simulation.ngspice import NgspiceSimulator

SIM = NgspiceSimulator()

# ---------------------------------------------------------------------------
# Fixture data: real ngspice output snippets captured from live runs
# ---------------------------------------------------------------------------

AC_OUTPUT = """\
 1.00000000e+01  1.00000000e+01  0.00000000e+00  1.00000000e+01  8.04768933e-02  8.11379306e-03
 1.01157945e+01  1.01157945e+01  0.00000000e+00  1.01157945e+01  8.06892773e-02  7.60907515e-03
 1.02329299e+01  1.02329299e+01  0.00000000e+00  1.02329299e+01  8.08977942e-02  7.10222842e-03
"""

TRANSIENT_OUTPUT = """\
 0.00000000e+00  0.00000000e+00  0.00000000e+00  6.28304849e+00
 1.00000000e-06  1.00000000e-06  1.00000000e-06  6.28291543e+00
 2.00000000e-06  2.00000000e-06  2.00000000e-06  6.28278238e+00
"""

TEMPLATE_CEA = """\
* Common Emitter
R1 vcc base {R1_VAL}
R2 vcc base {R2_VAL}
Rc vcc col {RC_VAL}
Re emit 0 {RE_VAL}
.end
"""

TEMPLATE_DIFF = """\
* Diff Amp
RL1 vcc col1 {RL_VAL}
RE1 emit tail {RE_VAL}
Rtail tail vee {RTAIL_VAL}
.end
"""

TEMPLATE_TIA = """\
* TIA
Rf out neg {RF_VAL}
Cf out neg {CF_VAL}
.end
"""

TEMPLATE_CLASS_A = """\
* Class-A
Rbias1 vcc base {RBIAS1_VAL}
Rbias2 base 0 {RBIAS2_VAL}
Rload vcc col {RLOAD_VAL}
Remitter emit 0 {REMITTER_VAL}
.end
"""


# ---------------------------------------------------------------------------
# TestCheckAvailable
# ---------------------------------------------------------------------------

class TestCheckAvailable:

    def test_returns_true_when_ngspice_on_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ngspice")
        assert SIM.check_available() is True

    def test_returns_false_when_ngspice_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        assert SIM.check_available() is False

    def test_returns_bool(self):
        result = SIM.check_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# TestInjectParams
# ---------------------------------------------------------------------------

class TestInjectParams:

    def test_basic_replacement(self):
        out = NgspiceSimulator.inject_params("R {R1_VAL} ohm", {"R1": 10000})
        assert "10000" in out
        assert "{R1_VAL}" not in out

    def test_uppercase_conversion(self):
        out = NgspiceSimulator.inject_params("R {RC_VAL}", {"Rc": 3000})
        assert "3000" in out

    def test_underscore_removal(self):
        # R_tail -> {RTAIL_VAL}
        out = NgspiceSimulator.inject_params("R {RTAIL_VAL}", {"R_tail": 22000})
        assert "22000" in out

    def test_all_cea_params(self):
        params = {"R1": 100000, "R2": 20000, "Rc": 3000, "Re": 470}
        out = NgspiceSimulator.inject_params(TEMPLATE_CEA, params)
        assert "100000" in out
        assert "20000" in out
        assert "3000" in out
        assert "470" in out
        assert "{" not in out  # all placeholders replaced

    def test_all_diff_amp_params(self):
        params = {"R_L": 4700, "R_E": 500, "R_tail": 22000}
        out = NgspiceSimulator.inject_params(TEMPLATE_DIFF, params)
        assert "4700" in out
        assert "500" in out
        assert "22000" in out
        assert "{" not in out

    def test_all_tia_params(self):
        params = {"R_f": 100000, "C_f": 1e-12}
        out = NgspiceSimulator.inject_params(TEMPLATE_TIA, params)
        assert "{" not in out

    def test_all_class_a_params(self):
        params = {"R_bias1": 47000, "R_bias2": 10000, "R_load": 2200, "R_emitter": 470}
        out = NgspiceSimulator.inject_params(TEMPLATE_CLASS_A, params)
        assert "{" not in out

    def test_raises_on_missing_placeholder(self):
        with pytest.raises(ValueError, match="not found in template"):
            NgspiceSimulator.inject_params("R {R1_VAL}", {"R_nonexistent": 999})

    def test_float_value_injected(self):
        out = NgspiceSimulator.inject_params("{R1_VAL}", {"R1": 1234.5})
        assert "1234.5" in out

    def test_scientific_notation_value(self):
        out = NgspiceSimulator.inject_params("{CF_VAL}", {"C_f": 1e-12})
        assert "1e-12" in out


# ---------------------------------------------------------------------------
# TestDetectSimType
# ---------------------------------------------------------------------------

class TestDetectSimType:

    def test_detects_ac_from_6_cols(self):
        assert NgspiceSimulator._detect_sim_type(AC_OUTPUT) == "ac"

    def test_detects_transient_from_4_cols(self):
        assert NgspiceSimulator._detect_sim_type(TRANSIENT_OUTPUT) == "transient"

    def test_returns_unknown_for_garbage(self):
        assert NgspiceSimulator._detect_sim_type("not a number\nstill not\n") == "unknown"

    def test_returns_unknown_for_empty(self):
        assert NgspiceSimulator._detect_sim_type("") == "unknown"

    def test_skips_header_lines(self):
        text = "# header line\n" + AC_OUTPUT
        assert NgspiceSimulator._detect_sim_type(text) == "ac"


# ---------------------------------------------------------------------------
# TestParseAC
# ---------------------------------------------------------------------------

class TestParseAC:

    def test_returns_three_arrays(self):
        result = NgspiceSimulator._parse_ac(AC_OUTPUT)
        assert set(result.keys()) == {"freq", "real", "imag"}

    def test_correct_row_count(self):
        result = NgspiceSimulator._parse_ac(AC_OUTPUT)
        assert len(result["freq"]) == 3

    def test_freq_values(self):
        result = NgspiceSimulator._parse_ac(AC_OUTPUT)
        assert pytest.approx(result["freq"][0], rel=1e-5) == 10.0

    def test_real_values_correct_column(self):
        result = NgspiceSimulator._parse_ac(AC_OUTPUT)
        # col 4 of first row = 8.04768933e-02
        assert pytest.approx(result["real"][0], rel=1e-5) == 8.04768933e-02

    def test_imag_values_correct_column(self):
        result = NgspiceSimulator._parse_ac(AC_OUTPUT)
        # col 5 of first row = 8.11379306e-03
        assert pytest.approx(result["imag"][0], rel=1e-5) == 8.11379306e-03

    def test_returns_numpy_arrays(self):
        result = NgspiceSimulator._parse_ac(AC_OUTPUT)
        for key in ("freq", "real", "imag"):
            assert isinstance(result[key], np.ndarray)

    def test_returns_empty_dict_for_garbage(self):
        result = NgspiceSimulator._parse_ac("not a number\n")
        assert result == {}

    def test_skips_rows_with_wrong_col_count(self):
        mixed = "1 2 3\n" + AC_OUTPUT  # 3-col row should be skipped
        result = NgspiceSimulator._parse_ac(mixed)
        assert len(result["freq"]) == 3  # only the 6-col rows counted


# ---------------------------------------------------------------------------
# TestParseTransient
# ---------------------------------------------------------------------------

class TestParseTransient:

    def test_returns_two_arrays(self):
        result = NgspiceSimulator._parse_transient(TRANSIENT_OUTPUT)
        assert set(result.keys()) == {"time", "voltage"}

    def test_correct_row_count(self):
        result = NgspiceSimulator._parse_transient(TRANSIENT_OUTPUT)
        assert len(result["time"]) == 3

    def test_time_starts_at_zero(self):
        result = NgspiceSimulator._parse_transient(TRANSIENT_OUTPUT)
        assert result["time"][0] == 0.0

    def test_voltage_correct_column(self):
        result = NgspiceSimulator._parse_transient(TRANSIENT_OUTPUT)
        # col 3 of first row = 6.28304849e+00
        assert pytest.approx(result["voltage"][0], rel=1e-5) == 6.28304849

    def test_returns_numpy_arrays(self):
        result = NgspiceSimulator._parse_transient(TRANSIENT_OUTPUT)
        for key in ("time", "voltage"):
            assert isinstance(result[key], np.ndarray)

    def test_returns_empty_dict_for_garbage(self):
        result = NgspiceSimulator._parse_transient("not a number\n")
        assert result == {}


# ---------------------------------------------------------------------------
# TestRunSingle (mocked subprocess + file I/O)
# ---------------------------------------------------------------------------

class TestRunSingle:

    def _make_mock_run(self):
        m = MagicMock()
        m.returncode = 0
        return m

    def test_returns_dict_on_success(self, tmp_path):
        output_file = tmp_path / "ngspice_simulation_output.txt"
        output_file.write_text(AC_OUTPUT)

        with patch("subprocess.run", return_value=self._make_mock_run()), \
             patch("tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__ = lambda s: str(tmp_path)
            mock_td.return_value.__exit__ = MagicMock(return_value=False)
            result = SIM.run_single(TEMPLATE_CEA, {"R1": 100000, "R2": 20000, "Rc": 3000, "Re": 470})

        assert result is not None
        assert "freq" in result

    def test_returns_none_on_bad_params(self):
        result = SIM.run_single(TEMPLATE_CEA, {"NONEXISTENT": 999})
        assert result is None

    def test_returns_none_when_output_file_missing(self, tmp_path):
        # No output file written — simulates a failed ngspice run
        with patch("subprocess.run", return_value=self._make_mock_run()), \
             patch("tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__ = lambda s: str(tmp_path)
            mock_td.return_value.__exit__ = MagicMock(return_value=False)
            result = SIM.run_single(TEMPLATE_CEA, {"R1": 100000, "R2": 20000, "Rc": 3000, "Re": 470})

        assert result is None

    def test_returns_none_on_timeout(self):
        import subprocess as sp
        with patch("subprocess.run", side_effect=sp.TimeoutExpired("ngspice", 60)):
            result = SIM.run_single(TEMPLATE_CEA, {"R1": 100000, "R2": 20000, "Rc": 3000, "Re": 470})
        assert result is None

    def test_returns_none_when_ngspice_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = SIM.run_single(TEMPLATE_CEA, {"R1": 100000, "R2": 20000, "Rc": 3000, "Re": 470})
        assert result is None


# ---------------------------------------------------------------------------
# TestRunBatch (mock _run_one)
# ---------------------------------------------------------------------------

class TestRunBatch:

    PARAMS_3 = [
        {"R1": 100000, "R2": 20000, "Rc": 3000, "Re": 470},
        {"R1": 80000,  "R2": 15000, "Rc": 2000, "Re": 300},
        {"R1": 120000, "R2": 25000, "Rc": 4000, "Re": 800},
    ]

    def _mock_run_one(self, result):
        SIM._run_one = MagicMock(return_value=result)

    def teardown_method(self):
        # Restore _run_one in case it was patched
        if isinstance(SIM._run_one, MagicMock):
            del SIM._run_one

    def test_returns_list_of_correct_length(self):
        fake = {"freq": np.array([1.0]), "real": np.array([0.1]), "imag": np.array([0.0])}
        with patch.object(SIM, "_run_one", return_value=fake):
            results = SIM.run_batch(TEMPLATE_CEA, self.PARAMS_3)
        assert len(results) == 3

    def test_calls_run_one_for_each_param_set(self):
        fake = {"freq": np.array([1.0]), "real": np.array([0.1]), "imag": np.array([0.0])}
        with patch.object(SIM, "_run_one", return_value=fake) as mock:
            SIM.run_batch(TEMPLATE_CEA, self.PARAMS_3)
        assert mock.call_count == 3

    def test_handles_partial_failures(self):
        # First call succeeds, rest fail
        fake = {"freq": np.array([1.0]), "real": np.array([0.1]), "imag": np.array([0.0])}
        side_effects = [fake, None, None]
        with patch.object(SIM, "_run_one", side_effect=side_effects):
            results = SIM.run_batch(TEMPLATE_CEA, self.PARAMS_3)
        assert sum(r is not None for r in results) == 1
        assert sum(r is None for r in results) == 2

    def test_progress_callback_called_correct_times(self):
        fake = {"freq": np.array([1.0]), "real": np.array([0.1]), "imag": np.array([0.0])}
        calls = []
        with patch.object(SIM, "_run_one", return_value=fake):
            SIM.run_batch(TEMPLATE_CEA, self.PARAMS_3, progress_callback=lambda c, t: calls.append((c, t)))
        assert len(calls) == 3

    def test_progress_callback_total_is_correct(self):
        fake = {"freq": np.array([1.0]), "real": np.array([0.1]), "imag": np.array([0.0])}
        calls = []
        with patch.object(SIM, "_run_one", return_value=fake):
            SIM.run_batch(TEMPLATE_CEA, self.PARAMS_3, progress_callback=lambda c, t: calls.append((c, t)))
        for _, total in calls:
            assert total == 3

    def test_progress_callback_reaches_total(self):
        fake = {"freq": np.array([1.0]), "real": np.array([0.1]), "imag": np.array([0.0])}
        final = []
        with patch.object(SIM, "_run_one", return_value=fake):
            SIM.run_batch(TEMPLATE_CEA, self.PARAMS_3, progress_callback=lambda c, t: final.append(c))
        assert max(final) == 3

    def test_empty_param_list_returns_empty(self):
        with patch.object(SIM, "_run_one", return_value=None):
            results = SIM.run_batch(TEMPLATE_CEA, [])
        assert results == []

    def test_all_none_on_complete_failure(self):
        with patch.object(SIM, "_run_one", return_value=None):
            results = SIM.run_batch(TEMPLATE_CEA, self.PARAMS_3)
        assert all(r is None for r in results)

    def test_results_are_in_input_order(self):
        """Results must map 1:1 to param_list order, not arrival order."""
        # Return a unique marker in each result so we can verify ordering
        call_order = []

        def fake_run_one(template, params):
            call_order.append(params["R1"])
            return {"marker": params["R1"]}

        with patch.object(SIM, "_run_one", side_effect=fake_run_one):
            results = SIM.run_batch(TEMPLATE_CEA, self.PARAMS_3, max_workers=1)

        assert results[0]["marker"] == 100000
        assert results[1]["marker"] == 80000
        assert results[2]["marker"] == 120000

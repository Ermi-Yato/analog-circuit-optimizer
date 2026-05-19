"""Phase 1 — Registry System tests."""
import json
import os
import pytest
import registry.circuit_registry as reg

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_cache():
    """Clear the registry cache before each test for isolation."""
    reg._cache.clear()
    yield
    reg._cache.clear()


@pytest.fixture
def valid_circuit():
    return {
        "id": "test_circuit",
        "name": "Test Circuit",
        "description": "For testing only",
        "spice_template": "circuits/test_circuit/test.template",
        "simulation_type": "ac",
        "parameters": [
            {
                "name": "R1",
                "label": "Resistor 1",
                "unit": "Ω",
                "min": 1000,
                "max": 10000,
                "default": 5000,
                "scale": "linear",
            }
        ],
        "metrics": [
            {
                "name": "Gain_dB",
                "label": "Gain",
                "unit": "dB",
                "optimize": "maximize",
            }
        ],
    }


# ---------------------------------------------------------------------------
# TestLoadAll
# ---------------------------------------------------------------------------

class TestLoadAll:

    def test_returns_five_circuits(self):
        circuits = reg.load_all()
        assert len(circuits) == 5

    def test_all_expected_ids_present(self):
        circuits = reg.load_all()
        expected = {
            "common_emitter_amplifier",
            "differential_amplifier",
            "sallen_key_filter",
            "transimpedance_amplifier",
            "class_a_amplifier",
        }
        assert set(circuits.keys()) == expected

    def test_id_matches_filename_for_all(self):
        circuits = reg.load_all()
        for cid, circuit in circuits.items():
            assert circuit["id"] == cid

    def test_repeated_call_repopulates_cache(self):
        c1 = reg.load_all()
        c2 = reg.load_all()
        assert set(c1.keys()) == set(c2.keys())

    def test_returns_dict_of_dicts(self):
        circuits = reg.load_all()
        for circuit in circuits.values():
            assert isinstance(circuit, dict)


# ---------------------------------------------------------------------------
# TestSchemaContent
# ---------------------------------------------------------------------------

class TestSchemaContent:

    def setup_method(self):
        self.circuits = reg.load_all()

    def test_common_emitter_has_four_parameters(self):
        c = self.circuits["common_emitter_amplifier"]
        assert len(c["parameters"]) == 4

    def test_common_emitter_parameter_names(self):
        c = self.circuits["common_emitter_amplifier"]
        names = {p["name"] for p in c["parameters"]}
        assert names == {"R1", "R2", "Rc", "Re"}

    def test_common_emitter_has_two_metrics(self):
        c = self.circuits["common_emitter_amplifier"]
        assert len(c["metrics"]) == 2

    def test_class_a_is_transient(self):
        assert self.circuits["class_a_amplifier"]["simulation_type"] == "transient"

    def test_all_others_are_ac(self):
        ac_ids = {
            "common_emitter_amplifier",
            "differential_amplifier",
            "sallen_key_filter",
            "transimpedance_amplifier",
        }
        for cid in ac_ids:
            assert self.circuits[cid]["simulation_type"] == "ac"

    def test_param_min_less_than_max_all_circuits(self):
        for cid, c in self.circuits.items():
            for p in c["parameters"]:
                assert p["min"] < p["max"], f"{cid}/{p['name']}: min >= max"

    def test_param_default_within_bounds_all_circuits(self):
        for cid, c in self.circuits.items():
            for p in c["parameters"]:
                assert p["min"] <= p["default"] <= p["max"], (
                    f"{cid}/{p['name']}: default out of bounds"
                )

    def test_thd_is_minimize(self):
        c = self.circuits["class_a_amplifier"]
        thd = next(m for m in c["metrics"] if m["name"] == "THD_percent")
        assert thd["optimize"] == "minimize"

    def test_sallen_key_capacitors_are_log_scale(self):
        c = self.circuits["sallen_key_filter"]
        cap_params = [p for p in c["parameters"] if p["name"] in ("C1", "C2")]
        assert len(cap_params) == 2
        for p in cap_params:
            assert p["scale"] == "log"

    def test_tia_params_are_log_scale(self):
        c = self.circuits["transimpedance_amplifier"]
        for p in c["parameters"]:
            assert p["scale"] == "log", f"TIA param {p['name']} should be log scale"

    def test_common_emitter_has_model_block(self):
        c = self.circuits["common_emitter_amplifier"]
        assert "model" in c
        assert "surrogate_path" in c["model"]
        assert "scaler_path" in c["model"]

    def test_all_circuits_have_model_block(self):
        # All 5 circuits are now trained (Phase 12 complete)
        for cid in ["common_emitter_amplifier", "differential_amplifier",
                    "sallen_key_filter", "transimpedance_amplifier", "class_a_amplifier"]:
            assert "model" in self.circuits[cid], f"{cid} should have a model block"
            assert "surrogate_path" in self.circuits[cid]["model"]
            assert "scaler_path" in self.circuits[cid]["model"]


# ---------------------------------------------------------------------------
# TestGet
# ---------------------------------------------------------------------------

class TestGet:

    def test_returns_dict_for_known_id(self):
        result = reg.get("common_emitter_amplifier")
        assert isinstance(result, dict)
        assert result["id"] == "common_emitter_amplifier"

    def test_returns_copy_not_reference(self):
        c1 = reg.get("common_emitter_amplifier")
        c1["name"] = "MUTATED"
        c2 = reg.get("common_emitter_amplifier")
        assert c2["name"] != "MUTATED"

    def test_raises_key_error_for_unknown_id(self):
        reg.load_all()
        with pytest.raises(KeyError, match="nonexistent_circuit"):
            reg.get("nonexistent_circuit")

    def test_lazy_loads_when_cache_empty(self):
        reg._cache.clear()
        result = reg.get("common_emitter_amplifier")
        assert result["id"] == "common_emitter_amplifier"


# ---------------------------------------------------------------------------
# TestModelExists
# ---------------------------------------------------------------------------

class TestModelExists:

    def test_model_exists_returns_bool(self):
        # Smoke test: model_exists always returns a bool, never raises
        reg.load_all()
        for cid in ["common_emitter_amplifier", "differential_amplifier",
                    "sallen_key_filter", "transimpedance_amplifier", "class_a_amplifier"]:
            assert isinstance(reg.model_exists(cid), bool)

    def test_false_for_unknown_id(self):
        assert reg.model_exists("nonexistent") is False

    def test_false_when_no_model_block(self):
        # Use a synthetic circuit dict with no model block
        from unittest.mock import patch
        no_model = {"id": "fake", "name": "Fake", "parameters": [], "metrics": []}
        with patch.object(reg, "_cache", {"fake": no_model}):
            assert reg.model_exists("fake") is False

    def test_true_when_both_pkls_present(self):
        circuit = reg.get("common_emitter_amplifier")
        model = circuit["model"]
        surrogate = os.path.join(reg._PROJECT_ROOT, model["surrogate_path"])
        scaler    = os.path.join(reg._PROJECT_ROOT, model["scaler_path"])

        os.makedirs(os.path.dirname(surrogate), exist_ok=True)
        os.makedirs(os.path.dirname(scaler), exist_ok=True)
        try:
            open(surrogate, "wb").write(b"dummy")
            open(scaler, "wb").write(b"dummy")
            assert reg.model_exists("common_emitter_amplifier") is True
        finally:
            for f in (surrogate, scaler):
                if os.path.exists(f):
                    os.remove(f)

    def test_false_when_only_surrogate_present(self):
        circuit = reg.get("common_emitter_amplifier")
        model = circuit["model"]
        surrogate = os.path.join(reg._PROJECT_ROOT, model["surrogate_path"])

        os.makedirs(os.path.dirname(surrogate), exist_ok=True)
        try:
            open(surrogate, "wb").write(b"dummy")
            assert reg.model_exists("common_emitter_amplifier") is False
        finally:
            if os.path.exists(surrogate):
                os.remove(surrogate)


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_valid_circuit_does_not_raise(self, valid_circuit):
        reg._validate(valid_circuit)

    def test_missing_top_level_key_raises(self, valid_circuit):
        del valid_circuit["metrics"]
        with pytest.raises(ValueError, match="missing required fields"):
            reg._validate(valid_circuit)

    def test_id_with_space_raises(self, valid_circuit):
        valid_circuit["id"] = "test circuit"
        with pytest.raises(ValueError, match="spaces"):
            reg._validate(valid_circuit)

    def test_empty_name_raises(self, valid_circuit):
        valid_circuit["name"] = ""
        with pytest.raises(ValueError):
            reg._validate(valid_circuit)

    def test_invalid_simulation_type_raises(self, valid_circuit):
        valid_circuit["simulation_type"] = "dc_sweep"
        with pytest.raises(ValueError, match="simulation_type"):
            reg._validate(valid_circuit)

    def test_empty_parameters_raises(self, valid_circuit):
        valid_circuit["parameters"] = []
        with pytest.raises(ValueError, match="parameters"):
            reg._validate(valid_circuit)

    def test_param_min_equals_max_raises(self, valid_circuit):
        valid_circuit["parameters"][0]["max"] = 1000
        with pytest.raises(ValueError, match="min"):
            reg._validate(valid_circuit)

    def test_param_default_below_min_raises(self, valid_circuit):
        valid_circuit["parameters"][0]["default"] = 500
        with pytest.raises(ValueError, match="default"):
            reg._validate(valid_circuit)

    def test_param_default_above_max_raises(self, valid_circuit):
        valid_circuit["parameters"][0]["default"] = 99999
        with pytest.raises(ValueError, match="default"):
            reg._validate(valid_circuit)

    def test_invalid_scale_raises(self, valid_circuit):
        valid_circuit["parameters"][0]["scale"] = "exponential"
        with pytest.raises(ValueError, match="scale"):
            reg._validate(valid_circuit)

    def test_invalid_optimize_raises(self, valid_circuit):
        valid_circuit["metrics"][0]["optimize"] = "neutral"
        with pytest.raises(ValueError, match="optimize"):
            reg._validate(valid_circuit)

    def test_missing_param_subfield_raises(self, valid_circuit):
        del valid_circuit["parameters"][0]["unit"]
        with pytest.raises(ValueError, match="missing fields"):
            reg._validate(valid_circuit)

    def test_model_block_missing_surrogate_raises(self, valid_circuit):
        valid_circuit["model"] = {"scaler_path": "trained_models/x/scaler.pkl"}
        with pytest.raises(ValueError, match="surrogate_path"):
            reg._validate(valid_circuit)


# ---------------------------------------------------------------------------
# TestRegister
# ---------------------------------------------------------------------------

class TestRegister:

    def test_writes_json_file(self, valid_circuit, tmp_path, monkeypatch):
        monkeypatch.setattr(reg, "_CIRCUITS_DIR", str(tmp_path))
        reg.register(valid_circuit)
        assert (tmp_path / "test_circuit.json").exists()

    def test_updates_cache(self, valid_circuit, tmp_path, monkeypatch):
        monkeypatch.setattr(reg, "_CIRCUITS_DIR", str(tmp_path))
        reg.register(valid_circuit)
        assert "test_circuit" in reg._cache

    def test_invalid_circuit_raises_before_writing(self, valid_circuit, tmp_path, monkeypatch):
        monkeypatch.setattr(reg, "_CIRCUITS_DIR", str(tmp_path))
        valid_circuit["simulation_type"] = "bogus"
        with pytest.raises(ValueError):
            reg.register(valid_circuit)
        assert not (tmp_path / "test_circuit.json").exists()

    def test_written_json_is_loadable(self, valid_circuit, tmp_path, monkeypatch):
        monkeypatch.setattr(reg, "_CIRCUITS_DIR", str(tmp_path))
        reg.register(valid_circuit)
        with open(tmp_path / "test_circuit.json", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["id"] == "test_circuit"

    def test_unicode_unit_preserved(self, valid_circuit, tmp_path, monkeypatch):
        monkeypatch.setattr(reg, "_CIRCUITS_DIR", str(tmp_path))
        reg.register(valid_circuit)
        content = (tmp_path / "test_circuit.json").read_text(encoding="utf-8")
        assert "Ω" in content

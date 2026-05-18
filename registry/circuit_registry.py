"""
Circuit Registry — single source of truth for all circuit metadata.

Users never edit JSON directly. The GUI's CircuitManagerView calls register()
to write circuits. All other modules call load_all() or get() to read.
"""
import json
import os
from typing import Optional

_REGISTRY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_REGISTRY_DIR)
_CIRCUITS_DIR = os.path.join(_REGISTRY_DIR, "circuits")

_REQUIRED_TOP_LEVEL = {
    "id", "name", "description", "spice_template",
    "simulation_type", "parameters", "metrics",
}
_REQUIRED_PARAM_FIELDS  = {"name", "label", "unit", "min", "max", "default", "scale"}
_REQUIRED_METRIC_FIELDS = {"name", "label", "unit", "optimize"}

_VALID_SIM_TYPES    = {"ac", "transient", "dc"}
_VALID_OPTIMIZE     = {"maximize", "minimize"}
_VALID_SCALE        = {"linear", "log"}

# Module-level cache. Not thread-safe — acceptable because the GUI uses
# QThread workers that each import this module independently.
_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(circuit: dict) -> None:
    """
    Validate a circuit dict against the required schema.
    Raises ValueError with a descriptive message on the first violation found.
    """
    missing = _REQUIRED_TOP_LEVEL - set(circuit.keys())
    if missing:
        raise ValueError(f"Circuit dict missing required fields: {missing}")

    cid = circuit["id"]
    if not isinstance(cid, str) or not cid.strip():
        raise ValueError("'id' must be a non-empty string")
    if " " in cid:
        raise ValueError(f"'id' must not contain spaces, got: {cid!r}")

    for field in ("name", "description"):
        val = circuit[field]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"'{field}' must be a non-empty string")

    if circuit["simulation_type"] not in _VALID_SIM_TYPES:
        raise ValueError(
            f"'simulation_type' must be one of {_VALID_SIM_TYPES}, "
            f"got {circuit['simulation_type']!r}"
        )

    params = circuit["parameters"]
    if not isinstance(params, list) or len(params) == 0:
        raise ValueError("'parameters' must be a non-empty list")

    for i, p in enumerate(params):
        missing_p = _REQUIRED_PARAM_FIELDS - set(p.keys())
        if missing_p:
            raise ValueError(f"Parameter[{i}] missing fields: {missing_p}")
        if not isinstance(p["min"], (int, float)):
            raise ValueError(f"Parameter[{i}]['min'] must be numeric")
        if not isinstance(p["max"], (int, float)):
            raise ValueError(f"Parameter[{i}]['max'] must be numeric")
        if p["min"] >= p["max"]:
            raise ValueError(
                f"Parameter[{i}] '{p['name']}': min ({p['min']}) must be "
                f"strictly less than max ({p['max']})"
            )
        if not isinstance(p["default"], (int, float)):
            raise ValueError(f"Parameter[{i}]['default'] must be numeric")
        if not (p["min"] <= p["default"] <= p["max"]):
            raise ValueError(
                f"Parameter[{i}] '{p['name']}': default ({p['default']}) must "
                f"be within [min={p['min']}, max={p['max']}]"
            )
        if p["scale"] not in _VALID_SCALE:
            raise ValueError(
                f"Parameter[{i}] '{p['name']}': scale must be one of "
                f"{_VALID_SCALE}, got {p['scale']!r}"
            )

    metrics = circuit["metrics"]
    if not isinstance(metrics, list) or len(metrics) == 0:
        raise ValueError("'metrics' must be a non-empty list")

    for i, m in enumerate(metrics):
        missing_m = _REQUIRED_METRIC_FIELDS - set(m.keys())
        if missing_m:
            raise ValueError(f"Metric[{i}] missing fields: {missing_m}")
        if m["optimize"] not in _VALID_OPTIMIZE:
            raise ValueError(
                f"Metric[{i}] '{m['name']}': optimize must be one of "
                f"{_VALID_OPTIMIZE}, got {m['optimize']!r}"
            )

    model = circuit.get("model")
    if model is not None:
        if not isinstance(model, dict):
            raise ValueError("'model' must be a dict if present")
        for key in ("surrogate_path", "scaler_path"):
            if key not in model:
                raise ValueError(f"'model' block missing required key: '{key}'")
            if not isinstance(model[key], str):
                raise ValueError(f"'model.{key}' must be a string path")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all() -> dict[str, dict]:
    """
    Scan registry/circuits/*.json, validate every file, and return a dict
    mapping circuit_id -> circuit_dict.

    Repopulates the in-memory cache on every call. Use get() for repeated
    single-circuit lookups after the initial load.

    Raises:
        FileNotFoundError: if the circuits directory doesn't exist.
        ValueError: if any JSON fails schema validation or id/filename mismatch.
    """
    global _cache
    _cache = {}

    if not os.path.isdir(_CIRCUITS_DIR):
        raise FileNotFoundError(
            f"Registry circuits directory not found: {_CIRCUITS_DIR}"
        )

    for filename in sorted(os.listdir(_CIRCUITS_DIR)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(_CIRCUITS_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filename}: {e}") from e

        _validate(data)

        expected_id = filename[:-5]  # strip ".json"
        if data["id"] != expected_id:
            raise ValueError(
                f"File '{filename}': 'id' field ({data['id']!r}) does not "
                f"match filename stem ({expected_id!r})"
            )

        _cache[data["id"]] = data

    return dict(_cache)


def get(circuit_id: str) -> dict:
    """
    Return a copy of the circuit dict for the given id.
    Lazy-loads the cache if it is empty.

    Returns a copy so callers cannot accidentally corrupt the cache.

    Raises:
        KeyError: if circuit_id is not found.
    """
    if not _cache:
        load_all()
    if circuit_id not in _cache:
        raise KeyError(f"Circuit '{circuit_id}' not found in registry")
    return dict(_cache[circuit_id])


def register(circuit: dict) -> None:
    """
    Validate circuit and write it to registry/circuits/<id>.json.
    Updates the in-memory cache on success.

    This is the only function that writes to disk. Called by the GUI's
    CircuitManagerView when a user saves a new or edited circuit.

    Raises:
        ValueError: if validation fails (no file is written).
    """
    _validate(circuit)

    os.makedirs(_CIRCUITS_DIR, exist_ok=True)

    circuit_id = circuit["id"]
    filepath = os.path.join(_CIRCUITS_DIR, f"{circuit_id}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(circuit, f, indent=2, ensure_ascii=False)

    _cache[circuit_id] = dict(circuit)


def model_exists(circuit_id: str) -> bool:
    """
    Return True only if BOTH the surrogate .pkl and scaler .pkl files exist
    on disk at the paths specified in the circuit's 'model' block.

    Returns False if:
    - circuit_id is unknown
    - circuit has no 'model' block
    - either .pkl file is missing
    """
    try:
        circuit = get(circuit_id)
    except KeyError:
        return False

    model_block = circuit.get("model")
    if model_block is None:
        return False

    surrogate = os.path.join(_PROJECT_ROOT, model_block["surrogate_path"])
    scaler    = os.path.join(_PROJECT_ROOT, model_block["scaler_path"])

    return os.path.isfile(surrogate) and os.path.isfile(scaler)

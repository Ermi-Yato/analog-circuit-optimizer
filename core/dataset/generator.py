"""
Dataset generator.

Workflow:
  1. Load circuit definition from registry
  2. Sample N random parameter sets (log or linear scale per parameter)
  3. Run all simulations in parallel via NgspiceSimulator.run_batch()
  4. Extract circuit-specific metrics from each raw simulation result
  5. Build a pandas DataFrame and save to data/<circuit_id>_dataset.csv

Metric extraction is dispatch-based: each metric name maps to a Python
function that receives (raw_data_dict, params_dict) and returns a float.
"""
import os
import math
from typing import Callable

import numpy as np
import pandas as pd

import registry.circuit_registry as reg
from core.simulation.ngspice import NgspiceSimulator

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR      = os.path.join(_PROJECT_ROOT, "data")


# ---------------------------------------------------------------------------
# Metric extraction functions
# Each takes (raw: dict, params: dict) and returns float or np.nan.
# raw keys:
#   AC:        {"freq", "real", "imag"}
#   Transient: {"time", "voltage"}
# params: the component values used for this simulation run
# ---------------------------------------------------------------------------

def _magnitude(raw: dict) -> np.ndarray:
    return np.sqrt(raw["real"] ** 2 + raw["imag"] ** 2)


def _gain_db_array(raw: dict) -> np.ndarray:
    mag = _magnitude(raw)
    return 20.0 * np.log10(np.maximum(mag, 1e-15))


def _peak_gain_db(raw: dict, params: dict) -> float:
    return float(np.max(_gain_db_array(raw)))


def _bandwidth_hz(raw: dict, params: dict) -> float:
    """
    3dB bandwidth for an amplifier (bandpass-style):
    BW = f_high - f_low  where both are within -3dB of peak gain.
    Returns np.nan if the gain never drops 3dB within the sweep range.
    """
    gain_db = _gain_db_array(raw)
    peak    = np.max(gain_db)
    above   = np.where(gain_db >= peak - 3.0)[0]
    if len(above) == 0:
        return np.nan
    bw = raw["freq"][above[-1]] - raw["freq"][above[0]]
    return float(bw) if bw > 0 else np.nan


def _cutoff_freq_hz(raw: dict, params: dict) -> float:
    """
    -3dB cutoff for a low-pass filter:
    The frequency where gain first drops 3dB below DC gain.
    """
    gain_db  = _gain_db_array(raw)
    dc_gain  = gain_db[0]           # gain at lowest simulated frequency
    target   = dc_gain - 3.0
    below    = np.where(gain_db <= target)[0]
    if len(below) == 0:
        # Gain never drops 3dB — return the sweep upper limit
        return float(raw["freq"][-1])
    return float(raw["freq"][below[0]])


def _q_factor(raw: dict, params: dict) -> float:
    """
    Quality factor for the Sallen-Key filter computed analytically from params:
      Q = sqrt(R1 * R2 * C1 * C2) / (C2 * (R1 + R2))
    Simulation data is not used — this is more reliable than estimating Q from
    a noisy frequency response.
    """
    R1, R2 = params["R1"], params["R2"]
    C1, C2 = params["C1"], params["C2"]
    denom = C2 * (R1 + R2)
    if denom == 0:
        return np.nan
    return float(math.sqrt(R1 * R2 * C1 * C2) / denom)


def _transimpedance_dbohm(raw: dict, params: dict) -> float:
    """
    Transimpedance gain in dBΩ.
    Input AC current = 1A, so |Z_t| = |V_out| / 1 = |V_out|.
    """
    return _peak_gain_db(raw, params)  # same formula since Iin = 1A AC


def _phase_margin_tia(raw: dict, params: dict) -> float:
    """
    Analytical phase margin calculation for transimpedance amplifier.
    
    For a TIA with feedback Rf||Cf and photodiode capacitance Cpd, the
    closed-loop response is a second-order system characterized by:
    
    Quality factor Q = (1/Cf) × sqrt(Cpd / (2π × Rf × GBW))
    Damping ratio ζ = 1 / (2Q)
    
    Phase margin is related to damping ratio by:
    PM = arctan(2ζ / sqrt(sqrt(1 + 4ζ⁴) - 2ζ²))
    
    For practical ranges:
    - Q ≈ 0.5 (ζ ≈ 1.0)   → PM ≈ 76° (overdamped)
    - Q ≈ 0.707 (ζ ≈ 0.707) → PM ≈ 65° (critically damped, Butterworth)
    - Q ≈ 1.0 (ζ ≈ 0.5)   → PM ≈ 52° (slightly underdamped)
    - Q ≈ 1.3 (ζ ≈ 0.38)  → PM ≈ 45° 
    - Q ≈ 2.0 (ζ ≈ 0.25)  → PM ≈ 30° (underdamped, marginal stability)
    
    Fixed circuit parameters (from transimpedance_amplifier.json):
    - Cpd = 10pF (photodiode junction capacitance)
    - GBW = 100MHz (op-amp gain-bandwidth product)
    """
    # Extract variable parameters
    Rf = params.get("R_f")
    Cf = params.get("C_f")
    
    if Rf is None or Cf is None:
        return np.nan
    
    # Fixed parameters for TIA (from registry fixed_components)
    Cpd = 10e-12   # 10pF photodiode capacitance
    GBW = 100e6    # 100MHz op-amp GBW
    
    # Calculate quality factor Q
    # Q = (1/Cf) × sqrt(Cpd / (2π × Rf × GBW))
    denominator = 2 * math.pi * Rf * GBW
    if denominator <= 0:
        return np.nan
    
    Q = (1.0 / Cf) * math.sqrt(Cpd / denominator)
    
    # Calculate damping ratio ζ = 1/(2Q)
    if Q <= 0:
        return np.nan
    zeta = 1.0 / (2.0 * Q)
    
    # Calculate phase margin using exact formula:
    # PM = arctan(2ζ / sqrt(sqrt(1 + 4ζ⁴) - 2ζ²))
    zeta_sq = zeta * zeta
    zeta_4th = zeta_sq * zeta_sq
    
    inner = math.sqrt(1.0 + 4.0 * zeta_4th) - 2.0 * zeta_sq
    
    # Handle edge cases where inner could be negative or zero
    if inner <= 0:
        # Very high damping ratio - system is heavily overdamped
        # Phase margin approaches 90°
        return 90.0
    
    argument = 2.0 * zeta / math.sqrt(inner)
    phase_margin = math.degrees(math.atan(argument))
    
    # Clamp to reasonable range [0, 90] for this formula
    # (PM > 90° means heavily overdamped, effectively very stable)
    return float(max(0.0, min(90.0, phase_margin)))


def _output_swing_v(raw: dict, params: dict) -> float:
    """
    Peak-to-peak output swing of the steady-state waveform.
    Skips the first 20% of the transient to allow initial conditions to settle.
    """
    v    = raw["voltage"]
    skip = max(1, int(0.2 * len(v)))
    steady = v[skip:]
    return float(np.max(steady) - np.min(steady))


def _thd_percent(raw: dict, params: dict) -> float:
    """
    Total Harmonic Distortion via FFT on the steady-state portion.
    THD = sqrt(sum(H2..HN)^2) / H1 * 100
    Falls back to NaN for very short or constant signals.
    """
    v    = raw["voltage"]
    skip = max(1, int(0.2 * len(v)))
    steady = v[skip:]

    if len(steady) < 16:
        return np.nan

    spectrum  = np.abs(np.fft.rfft(steady - np.mean(steady)))
    if len(spectrum) < 3:
        return np.nan

    fund_idx  = int(np.argmax(spectrum[1:])) + 1  # skip DC bin
    h1        = spectrum[fund_idx]
    if h1 < 1e-12:
        return np.nan

    # Collect harmonics at 2f, 3f, 4f, ... up to Nyquist
    harmonic_indices = [fund_idx * k for k in range(2, 10)
                        if fund_idx * k < len(spectrum)]
    if not harmonic_indices:
        return 0.0

    h_rms = float(np.sqrt(np.sum(spectrum[harmonic_indices] ** 2)))
    return float(h_rms / h1 * 100.0)


def _efficiency_percent(raw: dict, params: dict) -> float:
    """
    Approximate DC-to-AC power efficiency for the Class-A stage.
    P_out  = V_rms_ac² / R_load
    P_dc   = VCC * I_dc  ≈  VCC * (VCC - V_dc_mean) / R_load
    """
    VCC   = 12.0
    r_load = params.get("R_load")
    if r_load is None or r_load <= 0:
        return np.nan

    v     = raw["voltage"]
    skip  = max(1, int(0.2 * len(v)))
    steady = v[skip:]

    v_mean   = float(np.mean(steady))
    v_ac_rms = float(np.std(steady))

    p_out = (v_ac_rms ** 2) / r_load
    i_dc  = (VCC - v_mean) / r_load
    p_dc  = VCC * i_dc

    if p_dc <= 0:
        return np.nan
    return float(min(p_out / p_dc * 100.0, 100.0))


# Dispatch table: metric name -> extraction function
_EXTRACTORS: dict[str, Callable] = {
    "Peak_Gain_dB":          _peak_gain_db,
    "Diff_Gain_dB":          _peak_gain_db,
    "Transimpedance_dBOhm":  _transimpedance_dbohm,
    "Bandwidth_Hz":          _bandwidth_hz,
    "Cutoff_Freq_Hz":        _cutoff_freq_hz,
    "Q_factor":              _q_factor,
    "Phase_Margin_deg":      _phase_margin_tia,
    "Output_Swing_V":        _output_swing_v,
    "THD_percent":           _thd_percent,
    "Efficiency_percent":    _efficiency_percent,
}

# Pattern-based fallback: (lowercase fragment, sim_type_or_None) -> extractor
_PATTERN_FALLBACKS: list[tuple[str, str | None, Callable]] = [
    # AC patterns
    ("gain",      "ac",        _peak_gain_db),
    ("db",        "ac",        _peak_gain_db),
    ("bandwidth", "ac",        _bandwidth_hz),
    ("_bw",       "ac",        _bandwidth_hz),
    ("_hz",       "ac",        _bandwidth_hz),
    ("freq",      "ac",        _cutoff_freq_hz),
    ("cutoff",    "ac",        _cutoff_freq_hz),
    ("phase",     "ac",        _phase_margin_tia),
    ("impedance", "ac",        _transimpedance_dbohm),
    # Transient patterns
    ("swing",     "transient", _output_swing_v),
    ("_v",        "transient", _output_swing_v),
    ("thd",       "transient", _thd_percent),
    ("distortion","transient", _thd_percent),
    ("efficiency","transient", _efficiency_percent),
    # Generic fallbacks (any sim type)
    ("gain",      None,        _peak_gain_db),
    ("db",        None,        _peak_gain_db),
    ("_hz",       None,        _bandwidth_hz),
    ("_v",        None,        _output_swing_v),
]


def _resolve_extractor(metric_name: str, sim_type: str) -> Callable | None:
    """
    Return the best extractor for a metric name.
    1. Exact match in _EXTRACTORS
    2. Pattern match in _PATTERN_FALLBACKS (sim-type-specific first, then generic)
    3. None if nothing matches
    """
    fn = _EXTRACTORS.get(metric_name)
    if fn is not None:
        return fn

    name_lower = metric_name.lower()
    # sim-specific patterns first
    for fragment, stype, extractor in _PATTERN_FALLBACKS:
        if stype == sim_type and fragment in name_lower:
            return extractor
    # generic fallbacks
    for fragment, stype, extractor in _PATTERN_FALLBACKS:
        if stype is None and fragment in name_lower:
            return extractor

    return None


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def _sample_params(param_defs: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    """
    Generate N random parameter sets from circuit parameter definitions.

    Linear-scale parameters: uniform(min, max)
    Log-scale parameters:    10 ** uniform(log10(min), log10(max))

    All N sets are generated with numpy (vectorised) before any simulation
    starts, so the generation overhead is negligible even for N=10000.
    """
    # Build N values per parameter in one numpy call each
    columns: dict[str, np.ndarray] = {}
    for p in param_defs:
        if p["scale"] == "log":
            lo = math.log10(p["min"])
            hi = math.log10(p["max"])
            columns[p["name"]] = 10 ** rng.uniform(lo, hi, n)
        else:
            columns[p["name"]] = rng.uniform(p["min"], p["max"], n)

    # Transpose: list of N dicts
    names = list(columns.keys())
    return [
        {name: float(columns[name][i]) for name in names}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _extract_metrics(
    raw: dict,
    metric_names: list[str],
    params: dict,
    sim_type: str = "ac",
) -> tuple[dict | None, str | None]:
    """
    Apply the appropriate extractor for each metric name.
    Returns (result_dict, None) on success, or (None, reason_str) on failure.
    """
    result: dict[str, float] = {}
    for name in metric_names:
        extractor = _resolve_extractor(name, sim_type)
        if extractor is None:
            return None, f"no extractor for metric '{name}'"
        try:
            result[name] = extractor(raw, params)
        except Exception as exc:
            return None, f"extractor error for '{name}': {exc}"
    return result, None


# ---------------------------------------------------------------------------
# Validity checking
# ---------------------------------------------------------------------------

def _check_metric_ranges(metrics: dict, metric_defs: list[dict]) -> str | None:
    """
    Check if all metrics are within their valid_range bounds.
    Returns None if valid, or a reason string if invalid.
    """
    for m_def in metric_defs:
        name = m_def["name"]
        if name not in metrics:
            continue
        value = metrics[name]
        valid_range = m_def.get("valid_range")
        if valid_range is None:
            continue
        
        min_val = valid_range.get("min")
        max_val = valid_range.get("max")
        
        if min_val is not None and value < min_val:
            return f"{name}={value:.4g} below valid min {min_val}"
        if max_val is not None and value > max_val:
            return f"{name}={value:.4g} above valid max {max_val}"
    
    return None


def _check_validity_rules(metrics: dict, validity_checks: dict | None) -> str | None:
    """
    Check metrics against validity_checks rules from registry.
    Each rule has a 'condition' that when True indicates INVALID data.
    
    Supports simple comparisons like:
      - "Peak_Gain_dB < 5.0"
      - "Q_factor > 10.0"
      - "Phase_Margin_deg < 30.0"
      - "Output_Swing_V < 2.0 AND THD_percent > 40.0"
    
    Returns None if valid, or the check name + description if invalid.
    """
    if not validity_checks:
        return None
    
    for check_name, check_def in validity_checks.items():
        condition = check_def.get("condition", "")
        description = check_def.get("description", check_name)
        
        try:
            # Parse and evaluate the condition
            if _evaluate_condition(condition, metrics):
                return f"{check_name}: {description}"
        except Exception:
            # If condition can't be evaluated, skip it
            continue
    
    return None


def _evaluate_condition(condition: str, metrics: dict) -> bool:
    """
    Evaluate a simple condition string against metric values.
    Supports: <, >, <=, >=, ==, AND, OR
    Returns True if the condition matches (i.e., data is INVALID).
    """
    if not condition.strip():
        return False
    
    # Handle AND/OR
    if " AND " in condition:
        parts = condition.split(" AND ")
        return all(_evaluate_condition(p.strip(), metrics) for p in parts)
    if " OR " in condition:
        parts = condition.split(" OR ")
        return any(_evaluate_condition(p.strip(), metrics) for p in parts)
    
    # Parse comparison: "metric_name op value"
    import re
    match = re.match(r'(\w+)\s*([<>=!]+)\s*([\d.eE+-]+)', condition.strip())
    if not match:
        return False
    
    metric_name, op, value_str = match.groups()
    
    if metric_name not in metrics:
        return False
    
    metric_val = metrics[metric_name]
    threshold = float(value_str)
    
    if op == "<":
        return metric_val < threshold
    elif op == ">":
        return metric_val > threshold
    elif op == "<=":
        return metric_val <= threshold
    elif op == ">=":
        return metric_val >= threshold
    elif op == "==" or op == "=":
        return abs(metric_val - threshold) < 1e-9
    elif op == "!=" or op == "<>":
        return abs(metric_val - threshold) >= 1e-9
    
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    circuit_id: str,
    n_samples: int,
    progress_callback=None,
    max_workers: int | None = None,
    seed: int | None = None,
    verbose: bool = True,
    filter_invalid: bool = True,
) -> pd.DataFrame:
    """
    Generate a simulation dataset for a circuit.

    Args:
        circuit_id:        Registry circuit ID (e.g. "common_emitter_amplifier").
        n_samples:         Number of random parameter combinations to simulate.
        progress_callback: Called as (completed: int, total: int) after each sim.
                           Suitable for GUI progress bars.
        max_workers:       Parallel worker count (default: ThreadPoolExecutor default).
        seed:              Random seed for reproducibility.
        verbose:           Print progress summary and dataset stats when done.
        filter_invalid:    If True, filter out rows that fail validity checks
                           defined in the circuit registry (saturation, clipping, etc.)

    Returns:
        DataFrame with columns [param1, param2, ..., metric1, metric2, ...]
        Rows with failed simulations or invalid metrics are dropped.
        Also saved to data/<circuit_id>_dataset.csv.
    """
    circuit      = reg.get(circuit_id)
    param_defs   = circuit["parameters"]
    metric_defs  = circuit["metrics"]
    metric_names = [m["name"] for m in metric_defs]
    param_names  = [p["name"] for p in param_defs]
    sim_type     = circuit.get("simulation_type", "ac")
    
    # Get validity checking configuration from registry
    validity_checks = circuit.get("validity_checks")

    # Read SPICE template
    template_path = os.path.join(_PROJECT_ROOT, circuit["spice_template"])
    with open(template_path, "r", encoding="utf-8") as f:
        template_str = f.read()

    # Sample parameter sets
    rng = np.random.default_rng(seed)
    param_sets = _sample_params(param_defs, n_samples, rng)

    if verbose:
        print(f"\nGenerating dataset: {circuit['name']}")
        print(f"  Parameters : {param_names}")
        print(f"  Metrics    : {metric_names}")
        print(f"  Samples    : {n_samples}")
        print(f"  Workers    : {max_workers or 'auto'}")
        print(f"  Validity   : {'enabled' if filter_invalid else 'disabled'}")
        print()

    # Run simulations in parallel
    sim = NgspiceSimulator()
    raw_results = sim.run_batch(
        template_str,
        param_sets,
        progress_callback=progress_callback,
        max_workers=max_workers,
    )

    # Validate metric extractors before running any sims
    unresolved = [
        name for name in metric_names
        if _resolve_extractor(name, sim_type) is None
    ]
    if unresolved:
        raise ValueError(
            f"No extractor found for metric(s): {unresolved}.\n"
            f"Rename them to match a known pattern (e.g. containing 'Gain', 'dB', "
            f"'Bandwidth', 'Hz', 'Phase', 'Swing', 'THD') or add a custom extractor "
            f"to core/dataset/generator.py."
        )

    # Extract metrics from each result
    rows = []
    failed_sim = 0
    failed_nan = 0
    failed_validity = 0
    failed_range = 0
    failed_extract: dict[str, int] = {}
    validity_reasons: dict[str, int] = {}
    
    for params, raw in zip(param_sets, raw_results):
        if raw is None:
            failed_sim += 1
            continue
        metrics, reason = _extract_metrics(raw, metric_names, params, sim_type)
        if metrics is None:
            key = reason or "extraction error"
            failed_extract[key] = failed_extract.get(key, 0) + 1
            continue
        # Drop rows where any metric is NaN (unusable for training)
        if any(math.isnan(v) for v in metrics.values() if isinstance(v, float)):
            failed_nan += 1
            continue
        
        # Validity filtering (if enabled)
        if filter_invalid:
            # Check metric valid_range bounds
            range_reason = _check_metric_ranges(metrics, metric_defs)
            if range_reason:
                failed_range += 1
                validity_reasons[range_reason] = validity_reasons.get(range_reason, 0) + 1
                continue
            
            # Check validity rules (saturation, clipping, instability, etc.)
            validity_reason = _check_validity_rules(metrics, validity_checks)
            if validity_reason:
                failed_validity += 1
                validity_reasons[validity_reason] = validity_reasons.get(validity_reason, 0) + 1
                continue
        
        row = {name: params[name] for name in param_names}
        row.update(metrics)
        rows.append(row)

    failed = failed_sim + failed_nan + failed_validity + failed_range + sum(failed_extract.values())

    df = pd.DataFrame(rows, columns=param_names + metric_names)

    # Save CSV
    os.makedirs(_DATA_DIR, exist_ok=True)
    out_path = os.path.join(_DATA_DIR, f"{circuit_id}_dataset.csv")
    df.to_csv(out_path, index=False)

    if verbose:
        success = len(rows)
        print(f"\n  Completed  : {success}/{n_samples} successful ({failed} failed)")
        if failed_sim:
            print(f"  Sim errors : {failed_sim} (ngspice returned no output)")
        if failed_nan:
            print(f"  NaN rows   : {failed_nan} (metric returned NaN)")
        if failed_range:
            print(f"  Out of range: {failed_range} (metric outside valid_range)")
        if failed_validity:
            print(f"  Invalid    : {failed_validity} (failed validity_checks)")
        for reason, count in failed_extract.items():
            print(f"  Extract err: {count}x — {reason}")
        if validity_reasons and verbose:
            print(f"\n  Validity rejection breakdown:")
            for reason, count in sorted(validity_reasons.items(), key=lambda x: -x[1])[:5]:
                print(f"    {count}x — {reason}")
        print(f"\n  Saved to   : {out_path}")
        print()
        if len(df) > 0:
            print(preview(df))

    return df


def preview(df: pd.DataFrame, n_rows: int = 5) -> str:
    """Return a formatted string showing dataset stats + first N rows."""
    lines = []
    lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    lines.append("")
    lines.append("Stats:")
    lines.append(df.describe().to_string())
    lines.append("")
    lines.append(f"First {n_rows} rows:")
    lines.append(df.head(n_rows).to_string(index=False))
    return "\n".join(lines)

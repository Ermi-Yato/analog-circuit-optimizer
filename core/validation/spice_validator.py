"""
SPICE validation — Block 3 of the thesis pipeline.

Takes optimized component values, runs a real ngspice simulation, extracts
metrics using the same extractors as the dataset generator, and returns a
comparison table of surrogate-model predictions vs SPICE ground truth.

Public API:
    validate(circuit_id, params, predicted=None) -> validation_result_dict

result_dict keys:
    "circuit_id"  : str
    "params"      : {param_name: value, ...}
    "metrics"     : [
        {
            "name":      str,
            "predicted": float | None,  # surrogate prediction (if provided)
            "actual":    float,         # ngspice ground truth
            "abs_error": float | None,  # |predicted - actual|
            "rel_error": float | None,  # |predicted - actual| / |actual| * 100
        },
        ...
    ]
    "simulation_ok": bool   # False if ngspice failed or returned no data
"""
from __future__ import annotations

import os

import registry.circuit_registry as reg
from core.simulation.ngspice import NgspiceSimulator
from core.dataset.generator import _extract_metrics

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def validate(
    circuit_id: str,
    params: dict[str, float],
    predicted: dict[str, float] | None = None,
) -> dict:
    """
    Run SPICE validation on a set of component values.

    Args:
        circuit_id: Registry circuit ID.
        params:     {param_name: value} — component values to simulate.
        predicted:  Optional {metric_name: value} from the surrogate model.
                    If provided, abs_error and rel_error are computed.

    Returns:
        validation_result_dict — see module docstring.

    Raises:
        KeyError: circuit_id not in registry.
        FileNotFoundError: SPICE template not found.
    """
    circuit      = reg.get(circuit_id)
    metric_defs  = circuit["metrics"]
    metric_names = [m["name"] for m in metric_defs]

    template_path = os.path.join(_PROJECT_ROOT, circuit["spice_template"])
    with open(template_path, "r", encoding="utf-8") as f:
        template_str = f.read()

    sim = NgspiceSimulator()
    raw = sim.run_single(template_str, params)

    if raw is None:
        return {
            "circuit_id":    circuit_id,
            "params":        params,
            "metrics":       [
                {
                    "name":      name,
                    "predicted": predicted.get(name) if predicted else None,
                    "actual":    None,
                    "abs_error": None,
                    "rel_error": None,
                }
                for name in metric_names
            ],
            "simulation_ok": False,
        }

    extracted = _extract_metrics(raw, metric_names, params)

    metrics_out = []
    for name in metric_names:
        actual    = extracted.get(name) if extracted else None
        pred_val  = predicted.get(name) if predicted else None

        abs_err = None
        rel_err = None
        if pred_val is not None and actual is not None:
            abs_err = abs(pred_val - actual)
            rel_err = (abs_err / abs(actual) * 100.0) if abs(actual) > 1e-12 else None

        metrics_out.append({
            "name":      name,
            "predicted": pred_val,
            "actual":    actual,
            "abs_error": abs_err,
            "rel_error": rel_err,
        })

    return {
        "circuit_id":    circuit_id,
        "params":        params,
        "metrics":       metrics_out,
        "simulation_ok": extracted is not None,
    }


def format_table(result: dict) -> str:
    """
    Return a human-readable comparison table from a validation result dict.

    Example output:
        Circuit : Common-Emitter Amplifier
        Params  : R1=100000  R2=20000  Rc=3000  Re=470

        Metric              Predicted     Actual     Abs Err    Rel Err
        ─────────────────────────────────────────────────────────────────
        Peak_Gain_dB          12.50       12.31       0.19     1.54 %
        Bandwidth_Hz       98000.00    99725.00    1725.00     1.73 %
    """
    lines = []

    try:
        circuit = reg.get(result["circuit_id"])
        lines.append(f"Circuit : {circuit['name']}")
    except KeyError:
        lines.append(f"Circuit : {result['circuit_id']}")

    param_str = "  ".join(f"{k}={v:.4g}" for k, v in result["params"].items())
    lines.append(f"Params  : {param_str}")
    lines.append("")

    header = f"{'Metric':<25} {'Predicted':>12} {'Actual':>12} {'Abs Err':>12} {'Rel Err':>10}"
    lines.append(header)
    lines.append("─" * len(header))

    for m in result["metrics"]:
        pred_s = f"{m['predicted']:12.4g}" if m["predicted"] is not None else f"{'N/A':>12}"
        act_s  = f"{m['actual']:12.4g}"    if m["actual"]    is not None else f"{'N/A':>12}"
        abs_s  = f"{m['abs_error']:12.4g}" if m["abs_error"] is not None else f"{'N/A':>12}"
        rel_s  = f"{m['rel_error']:9.2f} %" if m["rel_error"] is not None else f"{'N/A':>10}"
        lines.append(f"{m['name']:<25}{pred_s}{act_s}{abs_s}{rel_s}")

    if not result["simulation_ok"]:
        lines.append("")
        lines.append("WARNING: ngspice simulation failed — actual values unavailable.")

    return "\n".join(lines)

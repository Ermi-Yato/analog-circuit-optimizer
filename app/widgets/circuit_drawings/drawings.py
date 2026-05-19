"""
Schemdraw circuit schematic drawings for all 5 registered topologies.

Each _draw_<id>() function builds a schemdraw.Drawing and returns the
resulting matplotlib Figure, styled for the dark GUI theme.

Usage:
    from app.widgets.circuit_drawings import get_drawing
    fig = get_drawing("sallen_key_filter")   # matplotlib.figure.Figure or None
"""
from __future__ import annotations

import warnings
import matplotlib
matplotlib.use("Agg")           # must be set before pyplot import
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm

# ── Design tokens (matches GUI theme) ────────────────────────────────────────
_BG    = "#0d1117"
_FG    = "#e6edf3"
_SUB   = "#8b949e"
_BLUE  = "#388bfd"
_GREEN = "#3fb950"

_DEFAULT_COLOR = _FG     # wire / element color
_LABEL_COLOR   = _SUB


def _make_drawing(**kwargs) -> schemdraw.Drawing:
    """Return a Drawing configured for the dark theme."""
    return schemdraw.Drawing(
        fontsize=10,
        color=_DEFAULT_COLOR,
        lw=1.4,
        **kwargs,
    )


def _finish(d: schemdraw.Drawing, title: str) -> matplotlib.figure.Figure:
    """Render the drawing, apply dark theme, return the matplotlib Figure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scd_fig = d.draw()

    fig: matplotlib.figure.Figure = scd_fig.fig
    fig.patch.set_facecolor(_BG)
    ax = scd_fig.ax
    ax.set_facecolor(_BG)
    if title:
        fig.suptitle(title, color=_SUB, fontsize=9, y=0.97)
    fig.tight_layout(pad=0.4)
    return fig


# ── Common-Emitter Amplifier ──────────────────────────────────────────────────

def _draw_common_emitter_amplifier() -> matplotlib.figure.Figure:
    d = _make_drawing()

    # VCC line at top
    vcc = d.add(elm.Dot().at((3, 6)).color(_FG))
    d.add(elm.Line().left(1).color(_FG))
    d.add(elm.Resistor().down(3).label("R1", loc="right").color(_FG))
    base_node = d.add(elm.Dot().color(_FG))
    d.add(elm.Resistor().down(2.5).label("R2", loc="right").color(_FG))
    d.add(elm.Ground().color(_FG))

    # BJT
    d.add(elm.Line().right(1.5).at(base_node.end).color(_FG))
    Q = d.add(elm.BjtNpn(circle=True).anchor("base").color(_FG))
    d.add(elm.Dot().at(Q.base).color(_FG))

    # Collector resistor to VCC
    d.add(elm.Resistor().up(2.5).at(Q.collector).label("Rc", loc="right").color(_FG))
    d.add(elm.Line().left().tox(3).color(_FG))
    d.add(elm.Dot().at((3, 6)).color(_FG))
    d.add(elm.Label().at((3, 6.3)).label("VCC", color=_BLUE))

    # Emitter resistor to GND
    d.add(elm.Resistor().down(2).at(Q.emitter).label("Re", loc="right").color(_FG))
    d.add(elm.Ground().color(_FG))

    # Input (left side, coupling cap)
    d.add(elm.Capacitor().left(1.5).at(Q.base).label("Cin", loc="top").color(_FG))
    d.add(elm.Dot().color(_FG))
    d.add(elm.Label().label("Vin", color=_SUB))

    # Output (right side of collector)
    d.add(elm.Line().right(0.8).at(Q.collector).color(_FG))
    out_n = d.add(elm.Dot().color(_FG))
    d.add(elm.Capacitor().right(1.2).at(out_n.end).label("Cout", loc="top").color(_FG))
    d.add(elm.Label().label("Vout", color=_SUB))

    return _finish(d, "Common-Emitter Amplifier")


# ── Differential Amplifier ────────────────────────────────────────────────────

def _draw_differential_amplifier() -> matplotlib.figure.Figure:
    d = _make_drawing()

    # Q1 (left BJT)
    Q1 = d.add(elm.BjtNpn(circle=True).at((1.5, 3)).color(_FG))
    # Q2 (right BJT)
    Q2 = d.add(elm.BjtNpn(circle=True).at((4.5, 3)).color(_FG))

    # Collector resistors to VCC
    d.add(elm.Resistor().up(2).at(Q1.collector).label("R_L", loc="right").color(_FG))
    rl1_top = d.add(elm.Dot().color(_FG))
    d.add(elm.Resistor().up(2).at(Q2.collector).label("R_L", loc="right").color(_FG))
    rl2_top = d.add(elm.Dot().color(_FG))

    # VCC rail connecting both collector loads
    d.add(elm.Line().right().at(rl1_top.end).tox(rl2_top.end).color(_FG))
    vcc_x = (rl1_top.end[0] + rl2_top.end[0]) / 2
    d.add(elm.Label().at((vcc_x, rl1_top.end[1] + 0.3)).label("VCC", color=_BLUE))

    # Emitter connection to tail
    e_y = Q1.emitter[1]
    d.add(elm.Line().right().at(Q1.emitter).tox(Q2.emitter).color(_FG))
    mid_e = ((Q1.emitter[0] + Q2.emitter[0]) / 2, e_y)
    d.add(elm.Dot().at(mid_e).color(_FG))
    d.add(elm.Resistor().down(2).at(mid_e).label("R_tail", loc="right").color(_FG))
    d.add(elm.Ground().color(_FG))

    # Inputs
    d.add(elm.Line().left(1.2).at(Q1.base).color(_FG))
    d.add(elm.Label().label("Vin+", color=_SUB))
    d.add(elm.Line().left(1.2).at(Q2.base).color(_FG))
    d.add(elm.Label().label("Vin−", color=_SUB))

    # Outputs
    d.add(elm.Line().right(0.8).at(Q1.collector).color(_FG))
    d.add(elm.Label().label("Vout+", color=_SUB))
    d.add(elm.Line().right(0.8).at(Q2.collector).color(_FG))
    d.add(elm.Label().label("Vout−", color=_SUB))

    return _finish(d, "Differential Amplifier (Long-Tail Pair)")


# ── Sallen-Key Active Filter ──────────────────────────────────────────────────

def _draw_sallen_key_filter() -> matplotlib.figure.Figure:
    d = _make_drawing()

    # Input
    d.add(elm.Dot().at((0, 3)).color(_FG))
    d.add(elm.Label().at((-0.3, 3)).label("Vin", color=_SUB))

    # R1
    d.add(elm.Resistor().right(3).at((0, 3)).label("R1", loc="top").color(_FG))
    nodeA = d.add(elm.Dot().color(_FG))

    # R2
    d.add(elm.Resistor().right(3).at(nodeA.end).label("R2", loc="top").color(_FG))
    nodeB = d.add(elm.Dot().color(_FG))

    # Op-amp
    op = d.add(elm.Opamp().anchor("in1").at(nodeB.end).right().color(_FG))

    # C1: nodeA → output (feedback)
    out_x = op.out[0]
    out_y = op.out[1]
    d.add(elm.Line().up(1.8).at(nodeA.end).color(_FG))
    c1_top = d.add(elm.Capacitor().right().tox(out_x).label("C1", loc="top").color(_FG))
    d.add(elm.Line().down().toy(out_y).color(_FG))
    d.add(elm.Dot().at(op.out).color(_FG))

    # C2: nodeB → GND
    d.add(elm.Capacitor().down(2).at(nodeB.end).label("C2", loc="right").color(_FG))
    d.add(elm.Ground().color(_FG))

    # Op-amp unity gain: output → inverting input
    d.add(elm.Line().down(0.8).at(op.out).color(_FG))
    inv_y = op.in2[1]
    d.add(elm.Line().left().tox(op.in2[0]).color(_FG))
    d.add(elm.Line().up().toy(inv_y).color(_FG))
    d.add(elm.Dot().at(op.in2).color(_FG))

    # Output label
    d.add(elm.Line().right(1).at(op.out).color(_FG))
    d.add(elm.Label().label("Vout", color=_SUB))

    return _finish(d, "Sallen-Key 2nd-Order Low-Pass Filter")


# ── Transimpedance Amplifier ──────────────────────────────────────────────────

def _draw_transimpedance_amplifier() -> matplotlib.figure.Figure:
    d = _make_drawing()

    # Op-amp
    op = d.add(elm.Opamp().at((3, 3)).right().color(_FG))

    # Non-inverting input to GND
    d.add(elm.Line().left(0.6).at(op.in1).color(_FG))
    d.add(elm.Ground().color(_FG))

    # Photodiode current source at inverting input
    d.add(elm.Line().left(1.5).at(op.in2).color(_FG))
    inv_node = d.add(elm.Dot().color(_FG))
    d.add(elm.SourceI().down(2).at(inv_node.end).label("I_PD", loc="right").color(_FG))
    d.add(elm.Ground().color(_FG))

    # Feedback: Rf || Cf from output to inverting input
    fb_start_x = op.out[0]
    fb_start_y = op.out[1]
    inv_x = inv_node.end[0]
    inv_y = inv_node.end[1]

    # Top feedback path
    d.add(elm.Line().up(1.5).at(op.out).color(_FG))
    top_r = d.add(elm.Dot().color(_FG))
    d.add(elm.Resistor().left().tox(inv_x).label("R_f", loc="top").color(_FG))
    d.add(elm.Line().down().toy(inv_y).color(_FG))
    d.add(elm.Dot().at(inv_node.end).color(_FG))

    # Cf in parallel: slightly above the resistor
    d.add(elm.Line().up(0.7).at(op.out).color(_FG))
    cf_r = d.add(elm.Dot().color(_FG))
    d.add(elm.Capacitor().left().tox(inv_x).label("C_f", loc="bottom").color(_FG))
    d.add(elm.Line().down().toy(inv_y).color(_FG))

    # Output
    d.add(elm.Line().right(1).at(op.out).color(_FG))
    d.add(elm.Label().label("Vout", color=_SUB))

    return _finish(d, "Transimpedance Amplifier (TIA)")


# ── Class-A Amplifier ─────────────────────────────────────────────────────────

def _draw_class_a_amplifier() -> matplotlib.figure.Figure:
    d = _make_drawing()

    # VCC
    d.add(elm.Label().at((3, 6.4)).label("VCC", color=_BLUE))
    d.add(elm.Dot().at((3, 6)).color(_FG))

    # Bias divider
    d.add(elm.Line().left(1.5).at((3, 6)).color(_FG))
    d.add(elm.Resistor().down(2.5).label("R_bias1", loc="right").color(_FG))
    base_node = d.add(elm.Dot().color(_FG))
    d.add(elm.Resistor().down(2.5).label("R_bias2", loc="right").color(_FG))
    d.add(elm.Ground().color(_FG))

    # BJT
    d.add(elm.Line().right(1.5).at(base_node.end).color(_FG))
    Q = d.add(elm.BjtNpn(circle=True).anchor("base").color(_FG))
    d.add(elm.Dot().at(Q.base).color(_FG))

    # Collector load to VCC
    d.add(elm.Resistor().up(2.5).at(Q.collector).label("R_load", loc="right").color(_FG))
    d.add(elm.Line().left().tox(3).color(_FG))
    d.add(elm.Dot().at((3, 6)).color(_FG))

    # Emitter degeneration
    d.add(elm.Resistor().down(2).at(Q.emitter).label("R_emitter", loc="right").color(_FG))
    d.add(elm.Ground().color(_FG))

    # Input coupling cap
    d.add(elm.Capacitor().left(1.5).at(Q.base).label("Cin", loc="top").color(_FG))
    d.add(elm.Label().label("Vin", color=_SUB))

    # Output
    d.add(elm.Line().right(0.8).at(Q.collector).color(_FG))
    cout_n = d.add(elm.Dot().color(_FG))
    d.add(elm.Capacitor().right(1.2).at(cout_n.end).label("Cout", loc="top").color(_FG))
    d.add(elm.Resistor().down(2).label("R_L", loc="right").color(_FG))
    d.add(elm.Ground().color(_FG))
    d.add(elm.Label().at(cout_n.end).label("Vout", color=_SUB))

    return _finish(d, "Class-A Amplifier")


# ── Dispatch ─────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, object] = {
    "common_emitter_amplifier":  _draw_common_emitter_amplifier,
    "differential_amplifier":    _draw_differential_amplifier,
    "sallen_key_filter":         _draw_sallen_key_filter,
    "transimpedance_amplifier":  _draw_transimpedance_amplifier,
    "class_a_amplifier":         _draw_class_a_amplifier,
}


def get_drawing(circuit_id: str) -> matplotlib.figure.Figure | None:
    """
    Return a dark-themed schematic Figure for the given circuit ID.
    Returns None if no drawing is registered for that circuit.
    """
    fn = _REGISTRY.get(circuit_id)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None

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

    # 1. Establish a single, clean VCC Power Rail at the top
    vcc_node = d.add(elm.Dot().at((2, 6.5)).color(_FG))
    d.add(elm.Label().at(vcc_node.start).label("VCC", loc="top", color=_BLUE))
    
    # Draw horizontal rail extension to the right for Rc
    rail_line = d.add(elm.Line().right(2.5).at(vcc_node.start).color(_FG))
    rc_top_node = d.add(elm.Dot().color(_FG))

    # 2. Left Branch: Bias Voltage Divider (Labels moved left to prevent collisions)
    r1 = d.add(elm.Resistor().down(2.5).at(vcc_node.start).label("R1", loc="left").color(_FG))
    base_node = d.add(elm.Dot().color(_FG))
    d.add(elm.Resistor().down(2.5).at(base_node.start).label("R2", loc="left").color(_FG))
    d.add(elm.Ground().color(_FG))

    # 3. Input path (matches SPICE: Vin -> Rs -> Cin -> base)
    d.add(elm.Capacitor().left(1.4).at(base_node.start).label("Cin", loc="top").color(_FG))
    cin_left = d.add(elm.Dot().color(_FG))
    d.add(elm.Resistor().left(1.2).at(cin_left.start).label("Rs", loc="top").color(_FG))
    vin_node = d.add(elm.Dot().color(_FG))
    d.add(elm.Label().at(vin_node.start).label("Vin", loc="left", color=_SUB))

    # 4. Transistor Interconnection
    d.add(elm.Line().right(1.2).at(base_node.start).color(_FG))
    Q = d.add(elm.BjtNpn(circle=True).anchor("base").color(_FG))
    d.add(elm.Dot().at(Q.base).color(_FG))

    # 5. Collector Branch (Drops cleanly from the right side of the VCC rail)
    rc = d.add(elm.Resistor().down(2.5).at(rc_top_node.start).to(Q.collector).label("Rc", loc="right").color(_FG))
    coll_node = d.add(elm.Dot().at(Q.collector).color(_FG))

    # 6. Output Branch & Separated Labels
    d.add(elm.Line().right(1.2).at(coll_node.start).color(_FG))
    out_node = d.add(elm.Dot().color(_FG))
    d.add(elm.Capacitor().right(1.5).at(out_node.start).label("Cout", loc="top").color(_FG))
    out_terminal = d.add(elm.Dot().color(_FG))
    
    # Offset Vout label cleanly to the right side of the node terminal
    d.add(elm.Label().at(out_terminal.start).label("Vout", loc="right", color=_SUB))
    
    # Drop RL down from the node with its label cleanly on the left side
    d.add(elm.Resistor().down(2.5).at(out_terminal.start).label("RL", loc="left").color(_FG))
    d.add(elm.Ground().color(_FG))

    # 7. Emitter Network transformation (Shifted labels downwards to stop BJT overlap)
    re_min = d.add(elm.Resistor().down(1.5).at(Q.emitter).label("Re_min", loc="right").color(_FG))
    emit_int = d.add(elm.Dot().color(_FG))

    # Swept Re Branch
    d.add(elm.Resistor().down(1.5).at(emit_int.start).label("Re", loc="right").color(_FG))
    d.add(elm.Ground().color(_FG))

    # Ce Bypass Capacitor Branch
    d.add(elm.Line().right(1.0).at(emit_int.start).color(_FG))
    d.add(elm.Capacitor().down(1.5).label("Ce", loc="right").color(_FG))
    d.add(elm.Ground().color(_FG))

    return _finish(d, "Common-Emitter Amplifier")


# ── Differential Amplifier ────────────────────────────────────────────────────

def _draw_differential_amplifier() -> matplotlib.figure.Figure:
    d = _make_drawing()

    # Top VCC rail
    d.add(elm.Line().at((1.0, 8.0)).right(10.0).color(_FG))
    d.add(elm.Label().at((6.0, 8.0)).label("VCC", loc="top", color=_BLUE))

    # Differential pair transistors
    q1 = d.add(elm.BjtNpn(circle=True).at((4.0, 4.2)).color(_FG))
    q2 = d.add(elm.BjtNpn(circle=True).at((8.0, 4.2)).color(_FG))

    # Collector loads to VCC
    d.add(elm.Resistor().down(3.0).at((4.0, 8.0)).to(q1.collector).label("RL1", loc="left").color(_FG))
    d.add(elm.Resistor().down(3.0).at((8.0, 8.0)).to(q2.collector).label("RL2", loc="right").color(_FG))

    # Output branch: col1 -> Cout -> Vout -> Rload
    d.add(elm.Line().right(1.0).at(q1.collector).color(_FG))
    d.add(elm.Capacitor().right(1.6).label("Cout", loc="top").color(_FG))
    vout = d.add(elm.Dot().color(_FG))
    d.add(elm.Label().at(vout.start).label("Vout", loc="right", color=_SUB))
    d.add(elm.Resistor().down(2.0).at(vout.start).label("Rload", loc="left").color(_FG))
    d.add(elm.Ground().color(_FG))

    # Left input/bias network
    b1 = d.add(elm.Dot().at(q1.base).color(_FG))
    d.add(elm.Resistor().up(2.2).at(b1.start).label("Rb1_q1", loc="left").color(_FG))
    d.add(elm.Line().right(0.4).color(_FG))
    d.add(elm.Resistor().down(2.0).at(b1.start).label("Rb2_q1", loc="left").color(_FG))
    d.add(elm.Ground().color(_FG))
    d.add(elm.Capacitor().left(1.2).at(b1.start).label("Cin_p", loc="top").color(_FG))
    d.add(elm.Resistor().left(1.2).label("Rs_p", loc="top").color(_FG))
    d.add(elm.Label().label("Vin+", loc="left", color=_SUB))

    # Right input/bias network
    b2 = d.add(elm.Dot().at(q2.base).color(_FG))
    d.add(elm.Resistor().up(2.2).at(b2.start).label("Rb1_q2", loc="right").color(_FG))
    d.add(elm.Line().left(0.4).color(_FG))
    d.add(elm.Resistor().down(2.0).at(b2.start).label("Rb2_q2", loc="right").color(_FG))
    d.add(elm.Ground().color(_FG))
    d.add(elm.Capacitor().right(1.2).at(b2.start).label("Cin_n", loc="top").color(_FG))
    d.add(elm.Resistor().right(1.2).label("Rs_n", loc="top").color(_FG))
    d.add(elm.Label().label("Vin−", loc="right", color=_SUB))

    # Tail bus
    tail_x, tail_y = 6.0, 1.6
    d.add(elm.Line().at((4.8, tail_y)).to((7.2, tail_y)).color(_FG))
    tail = d.add(elm.Dot().at((tail_x, tail_y)).color(_FG))
    d.add(elm.Label().at(tail.start).label("tail", loc="top", color=_SUB))

    # Left emitter to tail: RE1 || Ce1
    d.add(elm.Line().left(0.5).at(q1.emitter).color(_FG))
    d.add(elm.Resistor().down(2.0).label("RE1", loc="left").color(_FG))
    d.add(elm.Line().right(1.3).color(_FG))
    d.add(elm.Line().to((tail_x, tail_y)).color(_FG))

    d.add(elm.Line().right(0.5).at(q1.emitter).color(_FG))
    d.add(elm.Capacitor().down(2.0).label("Ce1", loc="right").color(_FG))
    d.add(elm.Line().left(0.3).color(_FG))
    d.add(elm.Line().to((tail_x, tail_y)).color(_FG))

    # Right emitter to tail: RE2 || Ce2
    d.add(elm.Line().right(0.5).at(q2.emitter).color(_FG))
    d.add(elm.Resistor().down(2.0).label("RE2", loc="right").color(_FG))
    d.add(elm.Line().left(1.3).color(_FG))
    d.add(elm.Line().to((tail_x, tail_y)).color(_FG))

    d.add(elm.Line().left(0.5).at(q2.emitter).color(_FG))
    d.add(elm.Capacitor().down(2.0).label("Ce2", loc="left").color(_FG))
    d.add(elm.Line().right(0.3).color(_FG))
    d.add(elm.Line().to((tail_x, tail_y)).color(_FG))

    # Tail resistor to VEE
    d.add(elm.Resistor().down(1.8).at((tail_x, tail_y)).label("R_tail", loc="right").color(_FG))
    vee = d.add(elm.Dot().color(_FG))
    d.add(elm.Label().at(vee.start).label("VEE", loc="bottom", color=_BLUE))

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
    # Photodiode junction capacitance to ground (Cpd in SPICE)
    d.add(elm.Line().left(0.9).at(inv_node.end).color(_FG))
    d.add(elm.Capacitor().down(1.3).label("Cpd", loc="left").color(_FG))
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

    # Emitter degeneration + bypass capacitor (matches Remitter || Ce in SPICE)
    d.add(elm.Resistor().down(2).at(Q.emitter).label("R_emitter", loc="right").color(_FG))
    d.add(elm.Ground().color(_FG))
    d.add(elm.Line().right(1.0).at(Q.emitter).color(_FG))
    d.add(elm.Capacitor().down(2).label("Ce", loc="right").color(_FG))
    d.add(elm.Ground().color(_FG))

    # Input coupling cap
    d.add(elm.Capacitor().left(1.5).at(Q.base).label("Cin", loc="top").color(_FG))
    d.add(elm.Label().label("Vin", color=_SUB))

    # Output
    d.add(elm.Line().right(0.8).at(Q.collector).color(_FG))
    cout_n = d.add(elm.Dot().color(_FG))
    d.add(elm.Capacitor().right(1.2).at(cout_n.end).label("Cout", loc="top").color(_FG))
    d.add(elm.Resistor().down(2).label("RL_ext", loc="right").color(_FG))
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

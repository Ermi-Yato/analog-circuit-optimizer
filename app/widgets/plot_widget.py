"""
Reusable embedded matplotlib canvas for PySide6.

Usage:
    chart = PlotWidget(bg="#0d1117")
    chart.plot_line(x, y, label="fitness", color="#388bfd")
    chart.set_labels(xlabel="Generation", ylabel="Score")
    chart.clear()
"""
from __future__ import annotations

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from app.design_system import BG0, BG1, BORDER, TEXT, TEXT_SUB as TEXT_S


class SchematicWidget(FigureCanvasQTAgg):
    """Read-only canvas that displays an externally-provided matplotlib Figure."""

    def __init__(self, parent=None):
        fig = Figure(facecolor=BG0, tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)

    def set_figure(self, fig: Figure):
        """Replace the displayed figure with *fig* and redraw."""
        self.figure = fig
        self.figure.set_canvas(self)
        self.draw_idle()


class PlotWidget(FigureCanvasQTAgg):
    def __init__(self, bg: str = BG0, parent=None):
        self._bg = bg
        fig = Figure(facecolor=bg, tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)

        self._ax = fig.add_subplot(111)
        self._style_axes()

    # ── Public API ────────────────────────────────────────────────────────────

    def plot_line(
        self,
        x: list,
        y: list,
        label: str = "",
        color: str = "#388bfd",
        linewidth: float = 1.8,
        ymin: float | None = None,
        ymax: float | None = None,
    ):
        """Replace current line data (single-line chart)."""
        self._ax.clear()
        self._style_axes()
        self._ax.plot(x, y, color=color, linewidth=linewidth,
                      label=label, solid_capstyle="round")
        if label:
            self._ax.legend(
                facecolor=BG1, edgecolor=BORDER,
                labelcolor=TEXT, fontsize=9,
            )
        # Apply explicit axis limits after auto-scale
        lo, hi = self._ax.get_ylim()
        if ymin is not None:
            lo = ymin
        if ymax is not None:
            hi = ymax
        # Always add 5% headroom above the data
        if y:
            data_max = max(y)
            hi = max(hi, data_max * 1.05 + 1e-9)
        self._ax.set_ylim(lo, hi)
        self.draw_idle()

    def set_labels(self, xlabel: str = "", ylabel: str = "", title: str = ""):
        self._ax.set_xlabel(xlabel, color=TEXT_S, fontsize=10)
        self._ax.set_ylabel(ylabel, color=TEXT_S, fontsize=10)
        if title:
            self._ax.set_title(title, color=TEXT, fontsize=11)
        self.draw_idle()

    def clear(self):
        self._ax.clear()
        self._style_axes()
        self.draw_idle()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor(BG1)
        ax.tick_params(colors=TEXT_S, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.xaxis.label.set_color(TEXT_S)
        ax.yaxis.label.set_color(TEXT_S)
        ax.grid(True, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.6)

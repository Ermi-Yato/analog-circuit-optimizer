"""
Dashboard - landing view shown on startup.

Each circuit card shows its pipeline status (Data / Train / Optimize)
and routes the user to the next needed step on click.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QGridLayout,
    QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap

import registry.circuit_registry as reg

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOGO_PATH    = os.path.join(_PROJECT_ROOT, "logo.jpg")

from app.design_system import (
    BG0, BG1, BG2, BORDER, TEXT, TEXT_SUB as TEXT_S, TEXT_DIM as TEXT_D,
    BLUE, GREEN, YELLOW, RED, divider,
)


def _sep() -> QFrame:
    return divider()


def _dataset_path(circuit_id: str) -> str:
    return os.path.join(_PROJECT_ROOT, "data", f"{circuit_id}_dataset.csv")


def _circuit_next_step(circuit_id: str) -> tuple[int, str]:
    """Return (view_index, cta_label) for the next needed step."""
    if not os.path.isfile(_dataset_path(circuit_id)):
        return 2, "Generate Data"
    if not reg.model_exists(circuit_id):
        return 3, "Train Model"
    return 4, "Optimize"


# ── Pipeline step indicator ───────────────────────────────────────────────────

class _PipelineBar(QWidget):
    """Horizontal  Data → Train → Optimize  indicator."""

    _STEPS = ["Data", "Train", "Optimize"]

    def __init__(self, has_data: bool, has_model: bool, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        states = [has_data, has_model, False]   # Optimize never "done" — always next action
        if has_model:
            states[2] = True   # model trained = optimize is reachable/done

        # actually: show optimize as "ready" (blue) when model exists, not green
        # green = done, blue = current/ready, gray = locked
        for i, (step, done) in enumerate(zip(self._STEPS, [has_data, has_model, has_model])):
            dot_color = GREEN if done else (BLUE if i == self._next_idx(has_data, has_model) else TEXT_D)
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {dot_color}; font-size: 10px; background: transparent;")
            lbl = QLabel(step)
            lbl.setStyleSheet(
                f"color: {TEXT_S if done else (TEXT if i == self._next_idx(has_data, has_model) else TEXT_D)}; "
                f"font-size: 10px; background: transparent;"
            )
            row.addWidget(dot)
            row.addSpacing(3)
            row.addWidget(lbl)
            if i < len(self._STEPS) - 1:
                arrow = QLabel("  →  ")
                arrow.setStyleSheet(f"color: {TEXT_D}; font-size: 10px; background: transparent;")
                row.addWidget(arrow)
        row.addStretch()

    @staticmethod
    def _next_idx(has_data: bool, has_model: bool) -> int:
        if not has_data:
            return 0
        if not has_model:
            return 1
        return 2


# ── Circuit card ─────────────────────────────────────────────────────────────

class _CircuitCard(QFrame):
    def __init__(self, circuit: dict, navigate_cb, parent=None):
        super().__init__(parent)
        cid = circuit["id"]
        has_data  = os.path.isfile(_dataset_path(cid))
        has_model = reg.model_exists(cid)
        nav_idx, cta_label = _circuit_next_step(cid)

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._nav_idx = nav_idx
        self._cid     = cid
        self._nav_cb  = navigate_cb

        border_color = GREEN if has_model else (BLUE if has_data else BORDER)
        self.setStyleSheet(
            f"QFrame {{ background: {BG1}; border: 1px solid {border_color}22; "
            f"border-radius: 10px; }}"
            f"QFrame:hover {{ border-color: {border_color}; }}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        # Row 1: name + sim chip
        row1 = QHBoxLayout()
        name_lbl = QLabel(circuit["name"])
        name_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        name_lbl.setStyleSheet(f"color: {TEXT}; background: transparent; border: none;")
        row1.addWidget(name_lbl, 1)

        sim_type = circuit.get("simulation_type", "ac").upper()
        sim_chip = QLabel(f" {sim_type} ")
        sim_chip.setStyleSheet(
            f"color: {TEXT_D}; background: {BG2}; border: 1px solid {BORDER}; "
            f"border-radius: 3px; font-size: 9px; padding: 1px 4px;"
        )
        row1.addWidget(sim_chip)
        root.addLayout(row1)

        # Row 2: description
        desc = circuit.get("description", "")
        if desc:
            dl = QLabel(desc[:80] + ("..." if len(desc) > 80 else ""))
            dl.setStyleSheet(f"color: {TEXT_S}; font-size: 11px; background: transparent; border: none;")
            dl.setWordWrap(True)
            root.addWidget(dl)

        # Row 3: pipeline status bar
        root.addWidget(_PipelineBar(has_data, has_model))

        # Row 4: R² scores if trained
        model_block = circuit.get("model") or {}
        r2_scores = model_block.get("r2_scores") or {}
        valid_r2 = {k: v for k, v in r2_scores.items() if isinstance(v, (int, float))}
        if has_model and valid_r2:
            sr = QHBoxLayout()
            sr.setSpacing(14)
            for metric, r2 in list(valid_r2.items())[:3]:
                clr = GREEN if r2 >= 0.95 else (YELLOW if r2 >= 0.80 else RED)
                lbl = QLabel(f"{metric.replace('_', ' ')}: R²={r2:.3f}")
                lbl.setStyleSheet(f"color: {clr}; font-size: 10px; background: transparent; border: none;")
                sr.addWidget(lbl)
            sr.addStretch()
            root.addLayout(sr)

        root.addWidget(_sep())

        # Row 5: CTA + details link
        bottom = QHBoxLayout()
        details_btn = QPushButton("Settings")
        details_btn.setFixedHeight(28)
        details_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        details_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {TEXT_D}; border: none; "
            f"font-size: 11px; text-decoration: underline; padding: 0; }}"
            f"QPushButton:hover {{ color: {TEXT_S}; }}"
        )
        details_btn.clicked.connect(lambda: navigate_cb(1, cid))
        bottom.addWidget(details_btn)
        bottom.addStretch()

        cta = QPushButton(f"{cta_label}  →")
        cta.setFixedHeight(30)
        cta.setCursor(Qt.CursorShape.PointingHandCursor)
        cta_bg   = GREEN  if has_model else (BLUE if has_data else BLUE)
        cta_text = "#fff"
        cta.setStyleSheet(
            f"QPushButton {{ background: {cta_bg}; color: {cta_text}; border: none; "
            f"border-radius: 6px; font-size: 11px; font-weight: 700; padding: 0 14px; }}"
            f"QPushButton:hover {{ opacity: 0.85; }}"
        )
        cta.clicked.connect(lambda: navigate_cb(nav_idx, cid))
        bottom.addWidget(cta)
        root.addLayout(bottom)

    def mousePressEvent(self, event):
        self._nav_cb(self._nav_idx, self._cid)


# ── Add-circuit card ──────────────────────────────────────────────────────────

class _AddCard(QFrame):
    def __init__(self, navigate_cb, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._nav_cb = navigate_cb
        self.setStyleSheet(
            f"QFrame {{ background: transparent; border: 2px dashed {BORDER}; "
            f"border-radius: 10px; }}"
            f"QFrame:hover {{ border-color: {BLUE}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 24, 18, 24)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        plus = QLabel("+")
        plus.setFont(QFont("Segoe UI", 24))
        plus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plus.setStyleSheet(f"color: {TEXT_D}; background: transparent; border: none;")

        lbl = QLabel("Add Circuit")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"color: {TEXT_D}; font-size: 12px; background: transparent; border: none;")

        layout.addWidget(plus)
        layout.addWidget(lbl)

    def mousePressEvent(self, event):
        self._nav_cb(1, None)   # Circuit Manager, no pre-selection


# ── Dashboard view ────────────────────────────────────────────────────────────

class DashboardView(QWidget):
    def __init__(self, navigate_cb=None, ngspice_available: bool = False, parent=None):
        super().__init__(parent)
        self._navigate_cb     = navigate_cb or (lambda i, cid=None: None)
        self._ngspice_available = ngspice_available
        self.setStyleSheet(f"background: {BG0};")
        self._build_ui()

    def refresh(self):
        old = self.layout()
        if old:
            QWidget().setLayout(old)
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            f"QScrollBar:vertical {{ background: {BG1}; width: 8px; }}"
            f"QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; }}"
        )

        content = QWidget()
        content.setStyleSheet(f"background: {BG0};")
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(40, 36, 40, 40)
        vbox.setSpacing(28)

        vbox.addLayout(self._build_header())
        if not self._ngspice_available:
            vbox.addWidget(self._build_ngspice_banner())
        vbox.addWidget(self._build_circuits_section())
        vbox.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll)

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(14)

        logo_lbl = QLabel()
        pix = QPixmap(_LOGO_PATH)
        if not pix.isNull():
            logo_lbl.setPixmap(pix.scaled(52, 52,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        logo_lbl.setStyleSheet("background: transparent;")
        row.addWidget(logo_lbl, 0, Qt.AlignmentFlag.AlignVCenter)

        txt = QVBoxLayout()
        txt.setSpacing(3)
        title = QLabel("Analog Circuits Optimization using ML")
        title.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT};")
        sub = QLabel("ML-Assisted Analog Circuit Optimizer. Set a target, get component values.")
        sub.setStyleSheet(f"color: {TEXT_S}; font-size: 14px;")
        txt.addWidget(title)
        txt.addWidget(sub)
        row.addLayout(txt, 1)

        return row

    def _build_ngspice_banner(self) -> QFrame:
        banner = QFrame()
        banner.setStyleSheet(
            f"QFrame {{ background: #2d1717; border: 1px solid {RED}; border-radius: 8px; }}"
        )
        row = QHBoxLayout(banner)
        row.setContentsMargins(16, 10, 16, 10)
        icon = QLabel("!")
        icon.setFixedSize(20, 20)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(
            f"color: {RED}; font-size: 13px; font-weight: bold; "
            f"background: transparent; border: none;"
        )
        row.addWidget(icon)
        msg = QLabel(
            "ngspice not found. SPICE simulation and dataset generation are disabled. "
            "Install ngspice and ensure it is on your PATH, then restart."
        )
        msg.setStyleSheet(f"color: {RED}; font-size: 12px; background: transparent; border: none;")
        msg.setWordWrap(True)
        row.addWidget(msg, 1)
        return banner

    def _build_circuits_section(self) -> QWidget:
        section = QWidget()
        section.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(section)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(14)

        # Header row
        hdr_row = QHBoxLayout()
        hdr = QLabel("Circuits")
        hdr.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        hdr.setStyleSheet(f"color: {TEXT};")
        hdr_row.addWidget(hdr, 1)
        vbox.addLayout(hdr_row)

        try:
            circuits = list(reg.load_all().values())
        except Exception:
            circuits = []

        # Summary line
        if circuits:
            trained  = sum(1 for c in circuits if reg.model_exists(c["id"]))
            with_data = sum(1 for c in circuits if os.path.isfile(_dataset_path(c["id"])))
            summary = QLabel(
                f"{len(circuits)} circuits   {with_data} with data   {trained} trained"
            )
            summary.setStyleSheet(f"color: {TEXT_D}; font-size: 12px;")
            vbox.addWidget(summary)

        # 2-column grid
        grid = QGridLayout()
        grid.setSpacing(12)

        for i, circuit in enumerate(circuits):
            card = _CircuitCard(circuit, self._navigate_cb)
            grid.addWidget(card, i // 2, i % 2)

        # Add-circuit card fills next slot
        next_i = len(circuits)
        add_card = _AddCard(self._navigate_cb)
        add_card.setMinimumHeight(100)
        grid.addWidget(add_card, next_i // 2, next_i % 2)

        vbox.addLayout(grid)
        return section

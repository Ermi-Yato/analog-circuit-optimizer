"""
Model Training View — Phase 9.

Lets the user pick a circuit, train the surrogate model,
and see R² / MAE per metric after training.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QGridLayout, QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

import registry.circuit_registry as reg
from app.workers.training_worker import TrainingWorker

# Design tokens
BG0 = "#0d1117"; BG1 = "#161b22"; BG2 = "#1c2128"
BORDER = "#30363d"; BORDER_F = "#388bfd"
TEXT = "#e6edf3"; TEXT_SUB = "#8b949e"; TEXT_DIM = "#484f58"
BLUE = "#388bfd"; BLUE_HOV = "#1f6feb"; BLUE_LT = "#58a6ff"
GREEN = "#3fb950"; RED = "#f85149"; YELLOW = "#d29922"


def _divider() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {BORDER}; border: none;")
    return f


def _eyebrow(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: {TEXT_DIM}; font-size: 10px; font-weight: 600; letter-spacing: 1.2px;"
    )
    return lbl


def _input_ss() -> str:
    return f"""
        QComboBox {{
            background: {BG2}; color: {TEXT};
            border: 1px solid {BORDER}; border-radius: 6px;
            padding: 0 10px; font-size: 12px; min-height: 34px;
        }}
        QComboBox:focus {{ border-color: {BORDER_F}; }}
        QComboBox::drop-down {{ border: none; padding-right: 8px; }}
        QComboBox QAbstractItemView {{
            background: {BG2}; color: {TEXT};
            border: 1px solid {BORDER}; selection-background-color: {BLUE};
        }}
    """


def _quality_label(r2: float) -> tuple[str, str]:
    """Returns (text, color) quality badge for an R² value."""
    if r2 >= 0.95:
        return "Excellent", GREEN
    if r2 >= 0.85:
        return "Good", BLUE
    if r2 >= 0.70:
        return "Fair", YELLOW
    return "Poor", RED


class _MetricCard(QWidget):
    """Displays R² and MAE for one metric with plain-English context."""

    def __init__(self, metric_name: str, label: str, unit: str, parent=None):
        super().__init__(parent)
        self._unit = unit
        self.setStyleSheet(f"""
            QWidget {{
                background: {BG1};
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Coloured left accent + header ─────────────────────────────────────
        self._accent = QFrame()
        self._accent.setFixedWidth(4)
        self._accent.setStyleSheet(f"background: {BORDER}; border: none; border-radius: 2px;")

        header_lbl = QLabel(label or metric_name)
        header_lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        header_lbl.setStyleSheet(f"color: {TEXT}; border: none; background: transparent;")

        unit_lbl = QLabel(f"({unit})" if unit else "")
        unit_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; border: none; background: transparent;")

        self._badge = QLabel("—")
        self._badge.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; font-weight: 700; "
            f"border: none; background: transparent; padding: 0;"
        )

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        header_row.addWidget(header_lbl)
        header_row.addWidget(unit_lbl)
        header_row.addStretch()
        header_row.addWidget(self._badge)

        # ── Stats grid ────────────────────────────────────────────────────────
        stats = QWidget()
        stats.setStyleSheet("background: transparent;")
        stats_layout = QVBoxLayout(stats)
        stats_layout.setContentsMargins(0, 8, 0, 0)
        stats_layout.setSpacing(10)

        # R² row
        r2_row = QHBoxLayout()
        r2_row.setSpacing(12)

        r2_key = QLabel("R²")
        r2_key.setFixedWidth(36)
        r2_key.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; font-weight: 700; "
            f"letter-spacing: 1px; border: none; background: transparent;"
        )
        self._r2_val = QLabel("—")
        self._r2_val.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self._r2_val.setStyleSheet(f"color: {TEXT}; border: none; background: transparent;")
        self._r2_sub = QLabel("how well the model fits the data  (1.0 = perfect)")
        self._r2_sub.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; border: none; background: transparent;")

        r2_row.addWidget(r2_key)
        r2_row.addWidget(self._r2_val)
        r2_row.addWidget(self._r2_sub, stretch=1)

        # MAE row
        mae_row = QHBoxLayout()
        mae_row.setSpacing(12)

        mae_key = QLabel("MAE")
        mae_key.setFixedWidth(36)
        mae_key.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; font-weight: 700; "
            f"letter-spacing: 1px; border: none; background: transparent;"
        )
        self._mae_val = QLabel("—")
        self._mae_val.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self._mae_val.setStyleSheet(f"color: {TEXT}; border: none; background: transparent;")
        self._mae_sub = QLabel("average prediction error")
        self._mae_sub.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; border: none; background: transparent;")

        mae_row.addWidget(mae_key)
        mae_row.addWidget(self._mae_val)
        mae_row.addWidget(self._mae_sub, stretch=1)

        stats_layout.addLayout(r2_row)
        stats_layout.addLayout(mae_row)

        # ── Assemble with left accent ─────────────────────────────────────────
        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(16, 14, 16, 14)
        inner_layout.setSpacing(0)
        inner_layout.addLayout(header_row)
        inner_layout.addWidget(stats)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._accent)
        body.addWidget(inner, stretch=1)

        root.addLayout(body)

    def update_scores(self, r2: float, mae: float):
        r2_color = GREEN if r2 >= 0.9 else (YELLOW if r2 >= 0.7 else RED)
        quality, badge_color = _quality_label(r2)

        self._accent.setStyleSheet(
            f"background: {r2_color}; border: none; border-radius: 2px;"
        )
        self._badge.setText(quality)
        self._badge.setStyleSheet(
            f"color: {badge_color}; font-size: 10px; font-weight: 700; "
            f"border: none; background: transparent; padding: 0;"
        )

        self._r2_val.setText(f"{r2 * 100:.1f}%")
        self._r2_val.setStyleSheet(
            f"color: {r2_color}; font-size: 18px; font-weight: 700; "
            f"border: none; background: transparent;"
        )

        unit = self._unit
        mae_text = f"±{mae:.4g} {unit}" if unit else f"±{mae:.4g}"
        self._mae_val.setText(mae_text)
        self._mae_val.setStyleSheet(f"color: {TEXT}; font-size: 18px; font-weight: 700; border: none; background: transparent;")


class TrainingView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {BG0};")
        self._worker: TrainingWorker | None = None
        self._cards: dict[str, _MetricCard] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar())
        root.addWidget(_divider())
        root.addWidget(self._build_status_bar())
        root.addWidget(_divider())
        root.addWidget(self._build_results_area(), stretch=1)

        self._refresh_circuits()

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(64)
        bar.setStyleSheet(f"background: {BG1};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(16)

        lbl = QLabel("Circuit")
        lbl.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
        self._combo_circuit = QComboBox()
        self._combo_circuit.setMinimumWidth(220)
        self._combo_circuit.setStyleSheet(_input_ss())
        self._combo_circuit.currentIndexChanged.connect(self._on_circuit_changed)

        self._btn_train = QPushButton("Train Model")
        self._btn_train.setFixedHeight(36)
        self._btn_train.setMinimumWidth(130)
        self._btn_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_train.setStyleSheet(f"""
            QPushButton {{
                background: {BLUE}; color: #fff;
                border: none; border-radius: 6px;
                font-size: 13px; font-weight: 700; padding: 0 20px;
            }}
            QPushButton:hover  {{ background: {BLUE_HOV}; border: 1px solid {BLUE_LT}; }}
            QPushButton:pressed {{ background: {BG2}; }}
            QPushButton:disabled {{ background: {BG2}; color: {TEXT_DIM}; }}
        """)
        self._btn_train.clicked.connect(self._on_train)

        layout.addWidget(lbl)
        layout.addWidget(self._combo_circuit)
        layout.addStretch()
        layout.addWidget(self._btn_train)

        return bar

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(40)
        bar.setStyleSheet(f"background: {BG1};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(24, 0, 24, 0)

        self._status_label = QLabel("Select a circuit and click Train Model.")
        self._status_label.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
        layout.addWidget(self._status_label)

        return bar

    # ── Results area ──────────────────────────────────────────────────────────

    def _build_results_area(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet(f"background: {BG0};")
        self._results_layout = QVBoxLayout(container)
        self._results_layout.setContentsMargins(24, 20, 24, 20)
        self._results_layout.setSpacing(12)

        self._eyebrow_lbl = _eyebrow("Model Performance")
        self._results_layout.addWidget(self._eyebrow_lbl)

        self._cards_container = QWidget()
        self._cards_container.setStyleSheet(f"background: transparent;")
        self._cards_grid = QGridLayout(self._cards_container)
        self._cards_grid.setSpacing(12)
        self._cards_grid.setContentsMargins(0, 0, 0, 0)
        self._results_layout.addWidget(self._cards_container)

        self._results_layout.addStretch()

        self._placeholder = QLabel(
            "Train a model to see R² and MAE scores here."
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(f"color: {TEXT_DIM}; font-size: 13px;")
        self._results_layout.addWidget(self._placeholder)
        self._results_layout.addStretch()

        return container

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _refresh_circuits(self):
        self._combo_circuit.blockSignals(True)
        self._combo_circuit.clear()
        try:
            for cid, c in reg.load_all().items():
                self._combo_circuit.addItem(c["name"], userData=cid)
        except Exception:
            pass
        self._combo_circuit.blockSignals(False)
        self._on_circuit_changed()

    def _on_circuit_changed(self):
        self._clear_cards()

    def _clear_cards(self):
        self._cards.clear()
        while self._cards_grid.count():
            item = self._cards_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _rebuild_cards(self, circuit_id: str):
        self._clear_cards()
        try:
            circuit = reg.get(circuit_id)
        except KeyError:
            return
        for i, m in enumerate(circuit["metrics"]):
            card = _MetricCard(m["name"], m.get("label", m["name"]), m.get("unit", ""))
            self._cards[m["name"]] = card
            self._cards_grid.addWidget(card, i // 2, i % 2)

    def _set_busy(self, busy: bool):
        self._combo_circuit.setEnabled(not busy)
        if busy:
            self._status_label.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
            self._btn_train.setText("Stop")
            self._btn_train.setStyleSheet(f"""
                QPushButton {{
                    background: {RED}; color: #fff;
                    border: none; border-radius: 6px;
                    font-size: 13px; font-weight: 700; padding: 0 20px;
                }}
                QPushButton:hover  {{ background: #da3633; border: 1px solid {RED}; }}
                QPushButton:pressed {{ background: {BG2}; }}
            """)
            try:
                self._btn_train.clicked.disconnect()
            except RuntimeError:
                pass
            self._btn_train.clicked.connect(self._on_stop)
        else:
            self._btn_train.setText("Train Model")
            self._btn_train.setStyleSheet(f"""
                QPushButton {{
                    background: {BLUE}; color: #fff;
                    border: none; border-radius: 6px;
                    font-size: 13px; font-weight: 700; padding: 0 20px;
                }}
                QPushButton:hover  {{ background: {BLUE_HOV}; border: 1px solid {BLUE_LT}; }}
                QPushButton:pressed {{ background: {BG2}; }}
                QPushButton:disabled {{ background: {BG2}; color: {TEXT_DIM}; }}
            """)
            try:
                self._btn_train.clicked.disconnect()
            except RuntimeError:
                pass
            self._btn_train.clicked.connect(self._on_train)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_train(self):
        circuit_id = self._combo_circuit.currentData()
        if not circuit_id:
            return

        self._set_busy(True)
        self._placeholder.hide()
        self._rebuild_cards(circuit_id)
        self._status_label.setText("Training in progress...")

        self._worker = TrainingWorker(circuit_id)
        self._worker.status.connect(self._status_label.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._status_label.setStyleSheet(f"color: {YELLOW}; font-size: 12px;")
            self._status_label.setText("Stopping — waiting for current step to finish...")
            self._btn_train.setEnabled(False)

    def _on_finished(self, metrics: dict):
        self._set_busy(False)
        n_test = metrics.get("n_test", "?")
        self._status_label.setText(
            f"Training complete — evaluated on {n_test} held-out samples."
        )
        self._status_label.setStyleSheet(f"color: {GREEN}; font-size: 12px;")

        for name, card in self._cards.items():
            r2  = metrics["r2"].get(name,  0.0)
            mae = metrics["mae"].get(name, 0.0)
            card.update_scores(r2, mae)

    def _on_stopped(self):
        self._set_busy(False)
        self._clear_cards()
        self._placeholder.show()
        self._status_label.setStyleSheet(f"color: {YELLOW}; font-size: 12px;")
        self._status_label.setText("Cancelled.")

    def _on_error(self, msg: str):
        self._set_busy(False)
        self._status_label.setStyleSheet(f"color: {RED}; font-size: 12px;")
        self._status_label.setText(f"Error: {msg}")

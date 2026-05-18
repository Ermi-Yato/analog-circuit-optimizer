"""
Optimizer View — Phase 10.

Layout:
  Toolbar      : circuit selector (trained only) + Optimize button
  Left panel   : target spec inputs + GA settings
  Right panel  : convergence chart + top-5 results table
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QFrame, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QScrollArea, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

import registry.circuit_registry as reg
from app.workers.optimizer_worker import OptimizerWorker
from app.widgets.plot_widget import PlotWidget

BG0 = "#0d1117"; BG1 = "#161b22"; BG2 = "#1c2128"
BORDER = "#30363d"; BORDER_F = "#388bfd"
TEXT = "#e6edf3"; TEXT_SUB = "#8b949e"; TEXT_DIM = "#484f58"
BLUE = "#388bfd"; BLUE_HOV = "#1f6feb"; BLUE_LT = "#58a6ff"
GREEN = "#3fb950"; RED = "#f85149"; YELLOW = "#d29922"


def _divider(vertical: bool = False) -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine if vertical else QFrame.Shape.HLine)
    f.setFixedWidth(1) if vertical else f.setFixedHeight(1)
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
        QComboBox, QDoubleSpinBox, QSpinBox {{
            background: {BG2}; color: {TEXT};
            border: 1px solid {BORDER}; border-radius: 6px;
            padding: 0 10px; font-size: 12px; min-height: 32px;
        }}
        QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
            border-color: {BORDER_F};
        }}
        QComboBox::drop-down {{ border: none; padding-right: 8px; }}
        QComboBox QAbstractItemView {{
            background: {BG2}; color: {TEXT};
            border: 1px solid {BORDER}; selection-background-color: {BLUE};
        }}
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
        QSpinBox::up-button, QSpinBox::down-button {{
            width: 18px; border: none; background: {BG2};
        }}
    """


class OptimizerView(QWidget):
    # Emitted when optimization finishes: (circuit_id, best_params, best_predicted)
    optimization_complete = Signal(str, dict, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {BG0};")
        self._worker: OptimizerWorker | None = None
        self._target_inputs: dict[str, QDoubleSpinBox] = {}
        self._history_x: list[int]   = []
        self._history_y: list[float] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar())
        root.addWidget(_divider())
        root.addWidget(self._build_status_bar())
        root.addWidget(_divider())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([280, 820])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        root.addWidget(splitter, stretch=1)

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

        self._btn_optimize = QPushButton("Optimize")
        self._btn_optimize.setFixedHeight(36)
        self._btn_optimize.setMinimumWidth(130)
        self._btn_optimize.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_optimize.setStyleSheet(f"""
            QPushButton {{
                background: {BLUE}; color: #fff;
                border: none; border-radius: 6px;
                font-size: 13px; font-weight: 700; padding: 0 20px;
            }}
            QPushButton:hover  {{ background: {BLUE_HOV}; border: 1px solid {BLUE_LT}; }}
            QPushButton:pressed {{ background: {BG2}; }}
            QPushButton:disabled {{ background: {BG2}; color: {TEXT_DIM}; }}
        """)
        self._btn_optimize.clicked.connect(self._on_optimize)

        layout.addWidget(lbl)
        layout.addWidget(self._combo_circuit)
        layout.addStretch()
        layout.addWidget(self._btn_optimize)
        return bar

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(40)
        bar.setStyleSheet(f"background: {BG1};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(24, 0, 24, 0)
        self._status_label = QLabel("Select a trained circuit and set target specs.")
        self._status_label.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
        layout.addWidget(self._status_label)
        return bar

    # ── Left panel: targets + GA settings ────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        outer = QWidget()
        outer.setStyleSheet(f"background: {BG1};")
        outer.setMinimumWidth(240)

        scroll = QScrollArea()
        scroll.setWidget(outer)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"background: {BG1}; border: none;")

        layout = QVBoxLayout(outer)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Target specs
        layout.addWidget(_eyebrow("Target Specs"))
        self._targets_container = QWidget()
        self._targets_container.setStyleSheet("background: transparent;")
        self._targets_layout = QVBoxLayout(self._targets_container)
        self._targets_layout.setContentsMargins(0, 0, 0, 0)
        self._targets_layout.setSpacing(10)
        layout.addWidget(self._targets_container)

        layout.addWidget(_divider())

        # GA settings
        layout.addWidget(_eyebrow("GA Settings"))
        ga_form = QVBoxLayout()
        ga_form.setSpacing(10)

        self._spin_generations = self._labeled_spin(
            ga_form, "Generations", 10, 2000, 100
        )
        self._spin_pop = self._labeled_spin(
            ga_form, "Population size", 20, 2000, 200
        )
        layout.addLayout(ga_form)
        layout.addStretch()

        return scroll

    def _labeled_spin(
        self, layout: QVBoxLayout, label: str, lo: int, hi: int, default: int
    ) -> QSpinBox:
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px; background: transparent;")
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(default)
        spin.setSingleStep(max(1, (hi - lo) // 20))
        spin.setStyleSheet(_input_ss())
        layout.addWidget(lbl)
        layout.addWidget(spin)
        return spin

    # ── Right panel: chart + results table ───────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {BG0};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        # Convergence chart
        layout.addWidget(_eyebrow("Convergence"))
        self._chart = PlotWidget(bg=BG0)
        self._chart.setMinimumHeight(220)
        self._chart.set_labels(xlabel="Generation", ylabel="Best Score")
        layout.addWidget(self._chart, stretch=2)

        layout.addWidget(_divider())

        # Results table
        layout.addWidget(_eyebrow("Top Candidates"))
        self._results_table = QTableWidget(0, 0)
        self._results_table.setStyleSheet(f"""
            QTableWidget {{
                background: {BG1}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 6px;
                gridline-color: {BORDER}; font-size: 12px;
                alternate-background-color: {BG0};
            }}
            QHeaderView::section {{
                background: {BG2}; color: {TEXT_SUB};
                border: none; border-bottom: 1px solid {BORDER};
                padding: 0 10px; height: 32px;
                font-size: 11px; font-weight: 600;
            }}
            QTableWidget::item {{ padding: 0 10px; border: none; }}
            QTableWidget::item:selected {{ background: {BLUE}22; color: {TEXT}; }}
        """)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._results_table, stretch=1)

        return panel

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _refresh_circuits(self):
        self._combo_circuit.blockSignals(True)
        self._combo_circuit.clear()
        try:
            for cid, c in reg.load_all().items():
                label = c["name"]
                if not reg.model_exists(cid):
                    label += "  (untrained)"
                self._combo_circuit.addItem(label, userData=cid)
        except Exception:
            pass
        self._combo_circuit.blockSignals(False)
        self._on_circuit_changed()

    def _on_circuit_changed(self):
        self._build_target_inputs()
        circuit_id = self._combo_circuit.currentData()
        trained = circuit_id and reg.model_exists(circuit_id)
        self._btn_optimize.setEnabled(bool(trained))
        if circuit_id and not trained:
            self._status_label.setStyleSheet(f"color: {YELLOW}; font-size: 12px;")
            self._status_label.setText(
                "This circuit has no trained model — go to Training to train it first."
            )
        else:
            self._status_label.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
            self._status_label.setText("Set target specs and click Optimize.")

    def _build_target_inputs(self):
        # Clear old widgets
        self._target_inputs.clear()
        while self._targets_layout.count():
            item = self._targets_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        circuit_id = self._combo_circuit.currentData()
        if not circuit_id:
            return

        try:
            circuit = reg.get(circuit_id)
        except KeyError:
            return

        for m in circuit["metrics"]:
            lbl = QLabel(f"{m['label']}  ({m['unit']})" if m.get("unit") else m["label"])
            lbl.setStyleSheet(
                f"color: {TEXT_SUB}; font-size: 12px; background: transparent;"
            )
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(-1e9, 1e9)
            spin.setSingleStep(1.0)
            spin.setStyleSheet(_input_ss())

            spin.setValue(m.get("target_default", 0.0))

            self._targets_layout.addWidget(lbl)
            self._targets_layout.addWidget(spin)
            self._target_inputs[m["name"]] = spin

    def _set_busy(self, busy: bool):
        self._combo_circuit.setEnabled(not busy)
        self._spin_generations.setEnabled(not busy)
        self._spin_pop.setEnabled(not busy)
        for sp in self._target_inputs.values():
            sp.setEnabled(not busy)

        if busy:
            self._btn_optimize.setEnabled(True)
            self._btn_optimize.setText("Stop")
            self._btn_optimize.setStyleSheet(f"""
                QPushButton {{
                    background: {RED}; color: #fff;
                    border: none; border-radius: 6px;
                    font-size: 13px; font-weight: 700; padding: 0 20px;
                }}
                QPushButton:hover  {{ background: #da3633; border: 1px solid {RED}; }}
                QPushButton:pressed {{ background: {BG2}; }}
            """)
            try:
                self._btn_optimize.clicked.disconnect()
            except RuntimeError:
                pass
            self._btn_optimize.clicked.connect(self._on_stop)
        else:
            circuit_id = self._combo_circuit.currentData()
            trained = circuit_id and reg.model_exists(circuit_id)
            self._btn_optimize.setEnabled(bool(trained))
            self._btn_optimize.setText("Optimize")
            self._btn_optimize.setStyleSheet(f"""
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
                self._btn_optimize.clicked.disconnect()
            except RuntimeError:
                pass
            self._btn_optimize.clicked.connect(self._on_optimize)

    def _reset_chart(self):
        self._history_x.clear()
        self._history_y.clear()
        self._chart.clear()
        self._chart.set_labels(xlabel="Generation", ylabel="Best Score")

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_optimize(self):
        circuit_id = self._combo_circuit.currentData()
        if not circuit_id:
            return

        targets = {name: sp.value() for name, sp in self._target_inputs.items()}
        if not targets:
            return

        self._set_busy(True)
        self._reset_chart()
        self._results_table.setRowCount(0)
        self._results_table.setColumnCount(0)
        self._status_label.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
        self._status_label.setText("Starting optimizer...")

        self._worker = OptimizerWorker(
            circuit_id=circuit_id,
            targets=targets,
            n_generations=self._spin_generations.value(),
            pop_size=self._spin_pop.value(),
        )
        self._worker.generation.connect(self._on_generation)
        self._worker.status.connect(self._status_label.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._status_label.setStyleSheet(f"color: {YELLOW}; font-size: 12px;")
            self._status_label.setText("Stopping — finishing current generation...")
            self._btn_optimize.setEnabled(False)

    def _on_stopped(self):
        self._set_busy(False)
        self._status_label.setStyleSheet(f"color: {YELLOW}; font-size: 12px;")
        self._status_label.setText("Cancelled.")

    def _on_generation(self, gen: int, best_score: float):
        self._history_x.append(gen)
        self._history_y.append(best_score)
        self._chart.plot_line(
            self._history_x, self._history_y,
            label="Best score", color=BLUE,
            ymin=0,
        )
        self._chart.set_labels(xlabel="Generation", ylabel="Best Score")

    def _on_finished(self, result: dict):
        self._set_busy(False)
        self._status_label.setStyleSheet(f"color: {GREEN}; font-size: 12px;")
        self._status_label.setText(
            f"Done — best score: {result['best_score']:.4f}"
        )
        self._populate_results(result)
        circuit_id = self._combo_circuit.currentData()
        if circuit_id:
            self.optimization_complete.emit(
                circuit_id, result["best_params"], result["best_predicted"]
            )

    def _on_error(self, msg: str):
        self._set_busy(False)
        self._status_label.setStyleSheet(f"color: {RED}; font-size: 12px;")
        self._status_label.setText(f"Error: {msg}")

    def _populate_results(self, result: dict):
        circuit_id = self._combo_circuit.currentData()
        try:
            circuit    = reg.get(circuit_id)
            param_defs = circuit["parameters"]
            metric_defs = circuit["metrics"]
        except (KeyError, TypeError):
            return

        param_names  = [p["name"] for p in param_defs]
        metric_names = [m["name"] for m in metric_defs]
        col_headers  = param_names + metric_names + ["Score"]

        top5 = result["population"][:5]

        self._results_table.setColumnCount(len(col_headers))
        self._results_table.setRowCount(len(top5))
        self._results_table.setHorizontalHeaderLabels(col_headers)

        from math import isnan
        from PySide6.QtGui import QColor

        for row, entry in enumerate(top5):
            params, score, predicted = entry if len(entry) == 3 else (*entry, {})
            self._results_table.setRowHeight(row, 32)
            col = 0

            for pname in param_names:
                v = params.get(pname, float("nan"))
                item = QTableWidgetItem(f"{v:.4g}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._results_table.setItem(row, col, item)
                col += 1

            for mname in metric_names:
                v = predicted.get(mname, float("nan"))
                text = f"{v:.4g}" if not isnan(v) else "—"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._results_table.setItem(row, col, item)
                col += 1

            score_item = QTableWidgetItem(f"{score:.4f}")
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if row == 0:
                score_item.setForeground(QColor(GREEN))
            self._results_table.setItem(row, col, score_item)

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
        self._last_chart_draw: float = 0.0   # throttle: time of last chart redraw

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

        algo_lbl = QLabel("Algorithm")
        algo_lbl.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
        self._combo_algo = QComboBox()
        self._combo_algo.addItem("Genetic Algorithm",         userData="ga")
        self._combo_algo.addItem("Differential Evolution",   userData="differential_evolution")
        self._combo_algo.addItem("Dual Annealing",           userData="dual_annealing")
        self._combo_algo.addItem("Bayesian / TPE",           userData="bayesian_tpe")
        self._combo_algo.addItem("Bayesian / CMA-ES",        userData="bayesian_cmaes")
        self._combo_algo.setMinimumWidth(190)
        self._combo_algo.setStyleSheet(_input_ss())
        self._combo_algo.currentIndexChanged.connect(self._on_algo_changed)

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
        layout.addSpacing(8)
        layout.addWidget(algo_lbl)
        layout.addWidget(self._combo_algo)
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

        # Algorithm settings container — swapped by _on_algo_changed
        self._algo_settings_container = QWidget()
        self._algo_settings_container.setStyleSheet("background: transparent;")
        self._algo_settings_layout = QVBoxLayout(self._algo_settings_container)
        self._algo_settings_layout.setContentsMargins(0, 0, 0, 0)
        self._algo_settings_layout.setSpacing(0)
        layout.addWidget(self._algo_settings_container)

        # Build algo-specific panels; only one visible at a time
        self._panel_ga   = self._build_ga_panel()
        self._panel_de   = self._build_de_panel()
        self._panel_da   = self._build_da_panel()
        self._panel_bo   = self._build_bo_panel()
        for p in (self._panel_ga, self._panel_de, self._panel_da, self._panel_bo):
            self._algo_settings_layout.addWidget(p)
            p.setVisible(False)
        self._panel_ga.setVisible(True)

        layout.addWidget(_divider())

        # Fitness function settings — shared across all algorithms
        self._panel_fitness = self._build_fitness_panel()
        layout.addWidget(self._panel_fitness)

        layout.addStretch()

        return scroll

    def _build_ga_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        form = QVBoxLayout(panel)
        form.setContentsMargins(0, 0, 0, 10)
        form.setSpacing(10)

        form.addWidget(_eyebrow("GA Settings"))
        self._spin_generations = self._labeled_spin(form, "Generations",  10, 2000, 100)
        self._spin_pop         = self._labeled_spin(form, "Population",   20, 2000, 200)

        form.addWidget(_divider())

        btn_adv = QPushButton("Advanced  ▸")
        btn_adv.setFlat(True)
        btn_adv.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_adv.setStyleSheet(
            f"color: {TEXT_SUB}; font-size: 11px; font-weight: 600; "
            f"text-align: left; padding: 0; background: transparent; border: none;"
        )
        adv_container = QWidget()
        adv_container.setStyleSheet("background: transparent;")
        adv_container.setVisible(False)
        adv_form = QVBoxLayout(adv_container)
        adv_form.setContentsMargins(0, 0, 0, 0)
        adv_form.setSpacing(10)
        self._spin_mutation   = self._labeled_spin_float(adv_form, "Mutation rate",   0.0, 1.0, 0.2)
        self._spin_crossover  = self._labeled_spin_float(adv_form, "Crossover rate",  0.0, 1.0, 0.7)
        self._spin_elite      = self._labeled_spin(adv_form, "Elite size",     1, 50,  4)
        self._spin_tournament = self._labeled_spin(adv_form, "Tournament size", 2, 20, 3)

        def _toggle():
            v = not adv_container.isVisible()
            adv_container.setVisible(v)
            btn_adv.setText("Advanced  ▾" if v else "Advanced  ▸")

        btn_adv.clicked.connect(_toggle)
        form.addWidget(btn_adv)
        form.addWidget(adv_container)
        return panel

    def _build_de_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        form = QVBoxLayout(panel)
        form.setContentsMargins(0, 0, 0, 10)
        form.setSpacing(10)

        form.addWidget(_eyebrow("Differential Evolution"))
        self._spin_de_maxiter      = self._labeled_spin(form, "Max iterations",    10, 5000, 300)
        self._spin_de_popsize      = self._labeled_spin(form, "Pop size / dim",     5,   50,  15)
        self._spin_de_mutation     = self._labeled_spin_float(form, "Mutation F",    0.0, 2.0, 0.7)
        self._spin_de_recombination= self._labeled_spin_float(form, "Recombination", 0.0, 1.0, 0.7)
        return panel

    def _build_da_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        form = QVBoxLayout(panel)
        form.setContentsMargins(0, 0, 0, 10)
        form.setSpacing(10)

        form.addWidget(_eyebrow("Dual Annealing"))
        self._spin_da_maxiter      = self._labeled_spin(form, "Max iterations",   100, 10000, 1000)
        self._spin_da_initial_temp = self._labeled_spin_float(form, "Initial temp", 0.1, 50000.0, 5230.0)
        return panel

    def _build_bo_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        form = QVBoxLayout(panel)
        form.setContentsMargins(0, 0, 0, 10)
        form.setSpacing(10)

        form.addWidget(_eyebrow("Bayesian Optimization"))
        self._spin_bo_trials  = self._labeled_spin(form, "Trials",          50, 5000, 300)
        self._spin_bo_startup = self._labeled_spin(form, "Startup (random)", 5,  200,  20)
        return panel

    def _build_fitness_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(_eyebrow("Fitness Function"))

        loss_lbl = QLabel("Loss type")
        loss_lbl.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px; background: transparent;")
        self._combo_loss = QComboBox()
        self._combo_loss.addItem("MAE — L1 (default)",  userData="mae")
        self._combo_loss.addItem("MSE — L2",            userData="mse")
        self._combo_loss.addItem("Huber — L1+L2 mix",  userData="huber")
        self._combo_loss.addItem("Log — compress outliers", userData="log")
        self._combo_loss.setStyleSheet(_input_ss())
        layout.addWidget(loss_lbl)
        layout.addWidget(self._combo_loss)

        self._spin_dir_penalty = self._labeled_spin_float(
            layout, "Direction penalty",
            1.0, 10.0, 1.0,
        )
        self._spin_dir_penalty.setToolTip(
            "Multiplier applied when prediction misses in the wrong direction.\n"
            "1.0 = symmetric. 2.0 = double penalty for e.g. gain below target."
        )
        self._spin_dir_penalty.setSingleStep(0.25)

        self._spin_tolerance = self._labeled_spin_float(
            layout, "Tolerance band (%)",
            0.0, 50.0, 0.0,
        )
        self._spin_tolerance.setToolTip(
            "Dead-zone around target: errors within ±X% count as zero penalty.\n"
            "0 = disabled (any deviation is penalised)."
        )
        self._spin_tolerance.setSingleStep(1.0)

        return panel

    def _on_algo_changed(self):
        algo = self._combo_algo.currentData()
        self._panel_ga.setVisible(algo == "ga")
        self._panel_de.setVisible(algo == "differential_evolution")
        self._panel_da.setVisible(algo == "dual_annealing")
        self._panel_bo.setVisible(algo in ("bayesian_tpe", "bayesian_cmaes"))

    def _labeled_spin_float(
        self, layout, label: str, lo: float, hi: float, default: float
    ) -> QDoubleSpinBox:
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px; background: transparent;")
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(default)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setStyleSheet(_input_ss())
        layout.addWidget(lbl)
        layout.addWidget(spin)
        return spin

    def _labeled_spin(
        self, layout, label: str, lo: int, hi: int, default: int
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

    def select_circuit(self, circuit_id: str):
        for i in range(self._combo_circuit.count()):
            if self._combo_circuit.itemData(i) == circuit_id:
                self._combo_circuit.setCurrentIndex(i)
                return

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

        # Adapt GA defaults to model type (only if GA panel exists yet)
        if trained and circuit_id and hasattr(self, "_spin_pop"):
            try:
                model_type = reg.get(circuit_id).get("model", {}).get("model_type", "random_forest")
                if model_type == "mlp":
                    self._spin_pop.setValue(100)
                    self._spin_generations.setValue(200)
                else:
                    self._spin_pop.setValue(200)
                    self._spin_generations.setValue(100)
            except Exception:
                pass

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
        self._combo_algo.setEnabled(not busy)
        self._algo_settings_container.setEnabled(not busy)
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
        self._last_chart_draw = 0.0
        self._chart.clear()
        algo = self._combo_algo.currentData() if hasattr(self, "_combo_algo") else "ga"
        xlabel = "Generation" if algo == "ga" else "Iteration"
        self._chart.set_labels(xlabel=xlabel, ylabel="Best Score")

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

        algo = self._combo_algo.currentData() or "ga"
        self._worker = OptimizerWorker(
            circuit_id=circuit_id,
            targets=targets,
            algorithm=algo,
            # GA
            n_generations=self._spin_generations.value(),
            pop_size=self._spin_pop.value(),
            mutation_prob=self._spin_mutation.value(),
            crossover_prob=self._spin_crossover.value(),
            elite_size=self._spin_elite.value(),
            tournament_size=self._spin_tournament.value(),
            # DE
            de_maxiter=self._spin_de_maxiter.value(),
            de_popsize=self._spin_de_popsize.value(),
            de_mutation=self._spin_de_mutation.value(),
            de_recombination=self._spin_de_recombination.value(),
            # DA
            da_maxiter=self._spin_da_maxiter.value(),
            da_initial_temp=self._spin_da_initial_temp.value(),
            # Bayesian
            bo_n_trials=self._spin_bo_trials.value(),
            bo_n_startup=self._spin_bo_startup.value(),
            # Fitness
            loss_type=self._combo_loss.currentData() or "mae",
            direction_penalty=self._spin_dir_penalty.value(),
            tolerance_pct=self._spin_tolerance.value(),
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
        import time
        self._history_x.append(gen)
        self._history_y.append(best_score)
        # Throttle chart redraws to ~15 fps — avoids Qt signal queue backlog
        # that causes the generation counter to jump (e.g. 1→5→12→...)
        now = time.monotonic()
        if now - self._last_chart_draw >= 0.067:   # ~15 fps cap
            self._last_chart_draw = now
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
        # Final chart flush — ensures last generation always renders
        if self._history_x:
            self._chart.plot_line(
                self._history_x, self._history_y,
                label="Best score", color=BLUE, ymin=0,
            )
            self._chart.set_labels(xlabel="Generation", ylabel="Best Score")
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

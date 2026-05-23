"""
Results View — Phase 11.

Tabs:
  1. Predicted vs Actual  — scatter per metric + perfect-prediction line
  2. Feature Importance   — horizontal bar chart
  3. Error Residuals      — histogram per metric
  4. SPICE Validation     — run real ngspice on best candidate, compare vs surrogate
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

import registry.circuit_registry as reg
from app.workers.results_worker import ResultsWorker
from app.widgets.plot_widget import PlotWidget

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BG0 = "#0d1117"; BG1 = "#161b22"; BG2 = "#1c2128"
BORDER = "#30363d"; BORDER_F = "#388bfd"
TEXT = "#e6edf3"; TEXT_SUB = "#8b949e"; TEXT_DIM = "#484f58"
BLUE = "#388bfd"; BLUE_HOV = "#1f6feb"; BLUE_LT = "#58a6ff"
GREEN = "#3fb950"; RED = "#f85149"; YELLOW = "#d29922"

# Per-metric accent colors for scatter / residual plots
_METRIC_COLORS = ["#388bfd", "#3fb950", "#f9e2af", "#f38ba8", "#cba6f7"]


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


def _table_ss() -> str:
    return f"""
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
    """


class ResultsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {BG0};")
        self._worker: ResultsWorker | None = None
        self._last_result: dict | None = None
        self._optimized_circuit: str | None = None
        self._optimized_params: dict | None = None
        self._optimized_predicted: dict | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar())
        root.addWidget(_divider())
        root.addWidget(self._build_status_bar())
        root.addWidget(_divider())
        root.addWidget(self._build_tabs(), stretch=1)

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

        self._btn_load = QPushButton("Load Results")
        self._btn_load.setFixedHeight(36)
        self._btn_load.setMinimumWidth(130)
        self._btn_load.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_load.setStyleSheet(f"""
            QPushButton {{
                background: {BLUE}; color: #fff;
                border: none; border-radius: 6px;
                font-size: 13px; font-weight: 700; padding: 0 20px;
            }}
            QPushButton:hover  {{ background: {BLUE_HOV}; border: 1px solid {BLUE_LT}; }}
            QPushButton:pressed {{ background: {BG2}; }}
            QPushButton:disabled {{ background: {BG2}; color: {TEXT_DIM}; }}
        """)
        self._btn_load.clicked.connect(self._on_load)

        self._btn_export = QPushButton("Export PNGs")
        self._btn_export.setFixedHeight(36)
        self._btn_export.setEnabled(False)
        self._btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_export.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {BLUE_LT};
                border: 1px solid {BORDER}; border-radius: 6px;
                font-size: 12px; padding: 0 16px;
            }}
            QPushButton:hover  {{ border-color: {BLUE_LT}; background: {BLUE}18; }}
            QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {BORDER}; }}
        """)
        self._btn_export.clicked.connect(self._on_export)

        layout.addWidget(lbl)
        layout.addWidget(self._combo_circuit)
        layout.addStretch()
        layout.addWidget(self._btn_export)
        layout.addWidget(self._btn_load)
        return bar

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(40)
        bar.setStyleSheet(f"background: {BG1};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(24, 0, 24, 0)
        self._status = QLabel("Select a trained circuit and click Load Results.")
        self._status.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
        layout.addWidget(self._status)
        return bar

    # ── Tabs ──────────────────────────────────────────────────────────────────

    def _build_tabs(self) -> QTabWidget:
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane  {{ border: none; background: {BG0}; }}
            QTabBar           {{ background: {BG1}; }}
            QTabBar::tab {{
                background: transparent; color: {TEXT_SUB};
                min-height: 44px; padding: 0 24px; border: none; font-size: 12px;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {TEXT}; font-weight: 600;
                border-bottom: 2px solid {BLUE};
            }}
            QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
        """)

        self._tabs.addTab(self._build_scatter_tab(),     "Predicted vs Actual")
        self._tabs.addTab(self._build_importance_tab(),  "Feature Importance")
        self._tabs.addTab(self._build_residuals_tab(),   "Error Residuals")
        self._tabs.addTab(self._build_validation_tab(),  "SPICE Validation")

        return self._tabs

    def _chart_tab(self) -> tuple[QWidget, PlotWidget]:
        tab = QWidget()
        tab.setStyleSheet(f"background: {BG0};")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 16, 20, 16)
        chart = PlotWidget(bg=BG0)
        layout.addWidget(chart)
        return tab, chart

    def _build_scatter_tab(self) -> QWidget:
        tab, self._scatter_chart = self._chart_tab()
        return tab

    def _build_importance_tab(self) -> QWidget:
        tab, self._importance_chart = self._chart_tab()
        return tab

    def _build_residuals_tab(self) -> QWidget:
        tab, self._residuals_chart = self._chart_tab()
        return tab

    def _build_validation_tab(self) -> QWidget:
        tab = QWidget()
        tab.setStyleSheet(f"background: {BG0};")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.addWidget(_eyebrow("Surrogate Prediction vs Ngspice Ground Truth"))
        header.addStretch()
        self._btn_validate = QPushButton("Run SPICE Validation")
        self._btn_validate.setFixedHeight(32)
        self._btn_validate.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_validate.setEnabled(False)
        self._btn_validate.setStyleSheet(f"""
            QPushButton {{
                background: {BG2}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 6px;
                font-size: 11px; padding: 0 14px;
            }}
            QPushButton:hover {{ border-color: {TEXT_SUB}; }}
            QPushButton:enabled:hover {{ border-color: {BLUE_LT}; color: {BLUE_LT}; }}
            QPushButton:disabled {{ color: {TEXT_DIM}; }}
        """)
        self._btn_validate.clicked.connect(self._on_validate)
        header.addWidget(self._btn_validate)
        layout.addLayout(header)

        self._val_table = QTableWidget(0, 0)
        self._val_table.setStyleSheet(_table_ss())
        self._val_table.setAlternatingRowColors(True)
        self._val_table.verticalHeader().setVisible(False)
        self._val_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._val_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._val_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._val_table, stretch=1)

        self._val_status = QLabel("")
        self._val_status.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        layout.addWidget(self._val_status)

        return tab

    # ── Helpers ───────────────────────────────────────────────────────────────

    def set_optimized_result(self, circuit_id: str, params: dict, predicted: dict):
        """Called by MainWindow when the optimizer finishes — stores params for SPICE validation."""
        self._optimized_circuit   = circuit_id
        self._optimized_params    = params
        self._optimized_predicted = predicted
        # Switch the combo to match and update the validation button hint
        for i in range(self._combo_circuit.count()):
            if self._combo_circuit.itemData(i) == circuit_id:
                self._combo_circuit.setCurrentIndex(i)
                break
        self._btn_validate.setEnabled(True)
        self._val_status.setStyleSheet(f"color: {BLUE}; font-size: 11px;")
        self._val_status.setText(
            "Optimized params ready — click Run SPICE Validation to verify."
        )

    def _refresh_circuits(self):
        self._combo_circuit.clear()
        try:
            for cid, c in reg.load_all().items():
                if reg.model_exists(cid):
                    self._combo_circuit.addItem(c["name"], userData=cid)
                else:
                    self._combo_circuit.addItem(f"{c['name']}  (untrained)", userData=cid)
        except Exception:
            pass

    def _set_busy(self, busy: bool):
        self._btn_load.setEnabled(not busy)
        self._combo_circuit.setEnabled(not busy)

    # ── Chart renderers ───────────────────────────────────────────────────────

    def _style_ax(self, ax):
        ax.set_facecolor(BG1)
        ax.tick_params(colors=TEXT_SUB, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.xaxis.label.set_color(TEXT_SUB)
        ax.yaxis.label.set_color(TEXT_SUB)
        ax.grid(True, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.5)

    def _render_scatter(self, data: dict):
        y_test       = data["y_test"]
        y_pred       = data["y_pred"]
        metric_names = data["metric_names"]
        n = len(metric_names)

        fig = self._scatter_chart.figure
        fig.clear()
        fig.set_facecolor(BG0)

        for i, name in enumerate(metric_names):
            ax  = fig.add_subplot(1, n, i + 1)
            self._style_ax(ax)
            col    = _METRIC_COLORS[i % len(_METRIC_COLORS)]
            actual = y_test[:, i]
            pred   = y_pred[:, i]
            lo, hi = actual.min(), actual.max()
            ax.scatter(actual, pred, alpha=0.55, color=col, s=18, edgecolors="none")
            ax.plot([lo, hi], [lo, hi], "--", color=TEXT_DIM, linewidth=1)
            ax.set_title(name, color=TEXT, fontsize=9, pad=6)
            ax.set_xlabel("Actual", color=TEXT_SUB, fontsize=9)
            if i == 0:
                ax.set_ylabel("Predicted", color=TEXT_SUB, fontsize=9)

        fig.tight_layout(pad=1.5)
        self._scatter_chart.draw()

    def _render_importance(self, data: dict):
        fi          = data["feature_importances"]
        param_names = data["param_names"]
        if fi is None:
            fig = self._importance_chart.figure
            fig.clear()
            fig.set_facecolor(BG0)
            ax = fig.add_subplot(111)
            self._style_ax(ax)
            ax.text(0.5, 0.5, "Feature importance\nnot available for MLP",
                    ha="center", va="center", transform=ax.transAxes,
                    color=TEXT_DIM, fontsize=11)
            ax.set_axis_off()
            self._importance_chart.draw()
            return

        idx    = np.argsort(fi)
        labels = [param_names[i] for i in idx]
        values = fi[idx]

        fig = self._importance_chart.figure
        fig.clear()
        fig.set_facecolor(BG0)
        ax = fig.add_subplot(111)
        self._style_ax(ax)

        colors = [GREEN if j == len(values) - 1 else BLUE for j in range(len(values))]
        ax.barh(labels, values, color=colors, height=0.5)
        ax.set_xlabel("Relative importance", color=TEXT_SUB, fontsize=10)

        fig.tight_layout(pad=1.5)
        self._importance_chart.draw()

    def _render_residuals(self, data: dict):
        y_test       = data["y_test"]
        y_pred       = data["y_pred"]
        metric_names = data["metric_names"]
        n = len(metric_names)

        fig = self._residuals_chart.figure
        fig.clear()
        fig.set_facecolor(BG0)

        for i, name in enumerate(metric_names):
            ax  = fig.add_subplot(1, n, i + 1)
            self._style_ax(ax)
            col       = _METRIC_COLORS[i % len(_METRIC_COLORS)]
            residuals = y_test[:, i] - y_pred[:, i]
            residuals = residuals[np.isfinite(residuals)]
            if len(residuals) == 0:
                ax.text(0.5, 0.5, "No finite\nresiduals", transform=ax.transAxes,
                        ha="center", va="center", color=TEXT_SUB, fontsize=9)
                continue
            ax.hist(residuals, bins=30, alpha=0.7, color=col, edgecolor="none")
            ax.axvline(0, color=TEXT_DIM, linewidth=1, linestyle="--")
            ax.set_title(name, color=TEXT, fontsize=9, pad=6)
            ax.set_xlabel("Prediction error", color=TEXT_SUB, fontsize=9)
            if i == 0:
                ax.set_ylabel("Count", color=TEXT_SUB, fontsize=9)

        fig.tight_layout(pad=1.5)
        self._residuals_chart.draw()

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_load(self):
        circuit_id = self._combo_circuit.currentData()
        if not circuit_id or not reg.model_exists(circuit_id):
            self._status.setStyleSheet(f"color: {YELLOW}; font-size: 12px;")
            self._status.setText("No trained model for this circuit.")
            return

        self._set_busy(True)
        self._status.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
        self._status.setText("Loading model and dataset...")

        self._worker = ResultsWorker(circuit_id)
        self._worker.finished.connect(self._on_loaded)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_loaded(self, data: dict):
        self._set_busy(False)
        self._last_result = data
        self._btn_export.setEnabled(True)
        self._btn_validate.setEnabled(True)

        self._render_scatter(data)
        self._render_importance(data)
        self._render_residuals(data)

        n = len(data["y_test"])
        self._status.setStyleSheet(f"color: {GREEN}; font-size: 12px;")
        self._status.setText(f"Loaded — {n} test samples.")

    def _on_error(self, msg: str):
        self._set_busy(False)
        self._status.setStyleSheet(f"color: {RED}; font-size: 12px;")
        self._status.setText(f"Error: {msg}")

    def _on_validate(self):
        circuit_id = self._combo_circuit.currentData()
        if not circuit_id:
            return

        self._val_status.setStyleSheet(f"color: {TEXT_SUB}; font-size: 11px;")
        self._val_status.setText("Running ngspice simulation...")
        self._btn_validate.setEnabled(False)

        try:
            import joblib
            import numpy as np
            import registry.circuit_registry as reg_inner
            from core.validation.spice_validator import validate
            from core.models.trainer import load_model as _load_model

            circuit    = reg_inner.get(circuit_id)
            param_defs = circuit["parameters"]

            # Prefer optimized params from the GA; fall back to circuit defaults
            if self._optimized_params and self._optimized_circuit == circuit_id:
                params    = self._optimized_params
                predicted = self._optimized_predicted
            else:
                params    = {p["name"]: p["default"] for p in param_defs}
                predicted = None

            # If no surrogate predictions yet, compute them from the model
            if predicted is None:
                model_block = circuit.get("model")
                if model_block:
                    try:
                        s_path = os.path.join(_ROOT, model_block["scaler_path"])
                        model  = _load_model(circuit_id)
                        scaler = joblib.load(s_path)
                        log_idx = [i for i, p in enumerate(param_defs)
                                   if p.get("scale") == "log"]
                        X = np.array([[params[p["name"]] for p in param_defs]], dtype=float)
                        if log_idx:
                            X = X.copy()
                            X[:, log_idx] = np.log10(np.abs(X[:, log_idx]).clip(1e-300))
                        # Compute physics-informed derived features if applicable
                        if circuit_id == "common_emitter_amplifier":
                            from core.dataset.preprocessor import compute_ce_derived_features
                            X = compute_ce_derived_features(X)
                        X_sc   = scaler.transform(X)
                        y_p    = model.predict(X_sc).ravel()
                        mnames = [m["name"] for m in circuit["metrics"]]
                        predicted = {name: float(y_p[i]) for i, name in enumerate(mnames)}
                    except Exception:
                        predicted = None

            result = validate(circuit_id, params, predicted=predicted)

            self._populate_val_table(result)
            ok = result["simulation_ok"]
            self._val_status.setStyleSheet(
                f"color: {GREEN if ok else RED}; font-size: 11px;"
            )
            self._val_status.setText(
                "Simulation complete — surrogate vs ngspice comparison."
                if ok else "Simulation failed — is ngspice on PATH?"
            )
        except Exception as exc:
            self._val_status.setStyleSheet(f"color: {RED}; font-size: 11px;")
            self._val_status.setText(f"Error: {exc}")
        finally:
            self._btn_validate.setEnabled(True)

    def _populate_val_table(self, result: dict):
        metrics = result["metrics"]
        headers = ["Metric", "Surrogate Prediction", "Ngspice Actual",
                   "Abs Error", "Rel Error (%)"]
        self._val_table.setColumnCount(len(headers))
        self._val_table.setHorizontalHeaderLabels(headers)
        self._val_table.setRowCount(len(metrics))

        for row, m in enumerate(metrics):
            self._val_table.setRowHeight(row, 34)

            def _cell(val, align=Qt.AlignmentFlag.AlignRight):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
                return item

            self._val_table.setItem(row, 0, _cell(m["name"],
                                    Qt.AlignmentFlag.AlignLeft))
            self._val_table.setItem(row, 1, _cell(
                f"{m['predicted']:.4g}" if m["predicted"] is not None else "—"
            ))
            self._val_table.setItem(row, 2, _cell(
                f"{m['actual']:.4g}" if m["actual"] is not None else "—"
            ))
            self._val_table.setItem(row, 3, _cell(
                f"{m['abs_error']:.4g}" if m["abs_error"] is not None else "—"
            ))
            rel_item = _cell(
                f"{m['rel_error']:.2f}" if m["rel_error"] is not None else "—"
            )
            if m["rel_error"] is not None:
                color = GREEN if m["rel_error"] < 5 else (YELLOW if m["rel_error"] < 15 else RED)
                rel_item.setForeground(QColor(color))
            self._val_table.setItem(row, 4, rel_item)

    def _on_export(self):
        if not self._last_result:
            return

        out_dir = os.path.join(_ROOT, "outputs")
        os.makedirs(out_dir, exist_ok=True)

        circuit_id = self._combo_circuit.currentData() or "circuit"
        for name, chart in [
            ("scatter",    self._scatter_chart),
            ("importance", self._importance_chart),
            ("residuals",  self._residuals_chart),
        ]:
            path = os.path.join(out_dir, f"{circuit_id}_{name}.png")
            chart.figure.savefig(path, dpi=150, bbox_inches="tight",
                                 facecolor=BG0)

        self._status.setStyleSheet(f"color: {GREEN}; font-size: 12px;")
        self._status.setText(f"Exported 3 PNGs to outputs/")

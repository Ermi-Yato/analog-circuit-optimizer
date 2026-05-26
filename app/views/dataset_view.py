"""
Dataset Generation View — Phase 9.

Lets the user pick a circuit, set a sample count, run generation,
and preview the resulting dataset in a table.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

import registry.circuit_registry as reg
from app.workers.simulation_worker import SimulationWorker

from app.design_system import (
    BG0, BG1, BG2, BORDER, TEXT, TEXT_SUB, TEXT_DIM,
    BLUE, BLUE_HOV, BLUE_LT, GREEN, RED, YELLOW,
    divider as _divider, eyebrow as _eyebrow,
    input_ss, table_ss,
)

_PREVIEW_ROWS = 20


def _input_ss() -> str:
    return input_ss("QComboBox, QSpinBox")


class DatasetView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {BG0};")
        self._worker: SimulationWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar())
        root.addWidget(_divider())
        root.addWidget(self._build_progress_bar())
        root.addWidget(_divider())
        root.addWidget(self._build_preview_table(), stretch=1)

        self._refresh_circuits()

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(64)
        bar.setStyleSheet(f"background: {BG1};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(16)

        # Circuit selector
        circ_label = QLabel("Circuit")
        circ_label.setStyleSheet(f"color: {TEXT_SUB}; font-size: 14px;")
        circ_label.setFont(QFont("Helvetica"))
        self._combo_circuit = QComboBox()
        self._combo_circuit.setMinimumWidth(220)
        self._combo_circuit.setStyleSheet(_input_ss())

        # Sample count
        n_label = QLabel("Samples")
        n_label.setFont(QFont("Helvetica"))
        n_label.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
        self._spin_n = QSpinBox()
        self._spin_n.setRange(10, 50000)
        self._spin_n.setValue(1000)
        self._spin_n.setSingleStep(100)
        self._spin_n.setFixedWidth(110)
        self._spin_n.setStyleSheet(_input_ss())

        # Generate button
        self._btn_generate = QPushButton("Generate Dataset")
        self._btn_generate.setFixedHeight(36)
        self._btn_generate.setMinimumWidth(150)
        self._btn_generate.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_generate.setStyleSheet(f"""
            QPushButton {{
                background: {BLUE}; color: #fff;
                border: none; border-radius: 6px;
                font-size: 13px; font-weight: 700; padding: 0 20px;
            }}
            QPushButton:hover  {{ background: {BLUE_HOV}; border: 1px solid {BLUE_LT}; }}
            QPushButton:pressed {{ background: {BG2}; }}
            QPushButton:disabled {{ background: {BG2}; color: {TEXT_DIM}; }}
        """)
        self._btn_generate.clicked.connect(self._on_generate)

        layout.addWidget(circ_label)
        layout.addWidget(self._combo_circuit)
        layout.addSpacing(8)
        layout.addWidget(n_label)
        layout.addWidget(self._spin_n)
        layout.addStretch()
        layout.addWidget(self._btn_generate)

        return bar

    # ── Progress bar ──────────────────────────────────────────────────────────

    def _build_progress_bar(self) -> QWidget:
        container = QWidget()
        container.setFixedHeight(48)
        container.setStyleSheet(f"background: {BG1};")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(12)

        self._status_label = QLabel("Ready.")
        self._status_label.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {BG2}; border: none; border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: {BLUE}; border-radius: 3px;
            }}
        """)

        layout.addWidget(self._status_label, stretch=1)
        layout.addWidget(self._progress, stretch=2)

        return container

    # ── Preview table ─────────────────────────────────────────────────────────

    def _build_preview_table(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet(f"background: {BG0};")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(10)

        layout.addWidget(_eyebrow(f"Dataset Preview — first {_PREVIEW_ROWS} rows"))

        self._table = QTableWidget(0, 0)
        self._table.setStyleSheet(table_ss())
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self._shape_label = QLabel("")
        self._shape_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")

        layout.addWidget(self._table, stretch=1)
        layout.addWidget(self._shape_label)

        return container

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_circuits()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def select_circuit(self, circuit_id: str):
        for i in range(self._combo_circuit.count()):
            if self._combo_circuit.itemData(i) == circuit_id:
                self._combo_circuit.setCurrentIndex(i)
                return

    def _refresh_circuits(self):
        self._combo_circuit.clear()
        try:
            for cid, c in reg.load_all().items():
                self._combo_circuit.addItem(c["name"], userData=cid)
        except Exception:
            pass

    def _set_busy(self, busy: bool):
        self._combo_circuit.setEnabled(not busy)
        self._spin_n.setEnabled(not busy)
        if busy:
            self._btn_generate.setText("Stop")
            self._btn_generate.setStyleSheet(f"""
                QPushButton {{
                    background: {RED}; color: #fff;
                    border: none; border-radius: 6px;
                    font-size: 13px; font-weight: 700; padding: 0 20px;
                }}
                QPushButton:hover  {{ background: #da3633; border: 1px solid {RED}; }}
                QPushButton:pressed {{ background: {BG2}; }}
            """)
            try:
                self._btn_generate.clicked.disconnect()
            except RuntimeError:
                pass
            self._btn_generate.clicked.connect(self._on_stop)
        else:
            self._progress.setValue(0)
            self._btn_generate.setText("Generate Dataset")
            self._btn_generate.setStyleSheet(f"""
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
                self._btn_generate.clicked.disconnect()
            except RuntimeError:
                pass
            self._btn_generate.clicked.connect(self._on_generate)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_generate(self):
        circuit_id = self._combo_circuit.currentData()
        if not circuit_id:
            return

        self._set_busy(True)
        self._status_label.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
        self._status_label.setText("Starting simulation...")
        self._progress.setValue(0)
        self._table.setRowCount(0)
        self._table.setColumnCount(0)

        self._worker = SimulationWorker(
            circuit_id=circuit_id,
            n_samples=self._spin_n.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._status_label.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._status_label.setStyleSheet(f"color: {YELLOW}; font-size: 12px;")
            self._status_label.setText("Stopping...")
            self._btn_generate.setEnabled(False)

    def _on_progress(self, completed: int, total: int):
        pct = int(completed / total * 100) if total > 0 else 0
        self._progress.setValue(pct)

    def _on_finished(self, df):
        self._set_busy(False)
        self._progress.setValue(100)
        self._status_label.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
        self._status_label.setText(
            f"Done — {len(df):,} rows  x  {len(df.columns)} columns"
        )
        self._populate_table(df)

    def _on_stopped(self):
        self._set_busy(False)
        self._status_label.setStyleSheet(f"color: {YELLOW}; font-size: 12px;")
        self._status_label.setText("Cancelled.")

    def _on_error(self, msg: str):
        from PySide6.QtWidgets import QMessageBox
        self._set_busy(False)
        self._status_label.setStyleSheet(f"color: {RED}; font-size: 12px;")
        # Show short version in status bar; pop a dialog for multi-line errors
        first_line = msg.splitlines()[0] if msg else "Unknown error"
        self._status_label.setText(f"Error: {first_line}")
        if "\n" in msg:
            QMessageBox.critical(self, "Dataset Generation Failed", msg)

    def _populate_table(self, df):
        preview = df.head(_PREVIEW_ROWS)
        self._table.setColumnCount(len(df.columns))
        self._table.setRowCount(len(preview))
        self._table.setHorizontalHeaderLabels(list(df.columns))

        for r, (_, row) in enumerate(preview.iterrows()):
            for c, val in enumerate(row):
                item = QTableWidgetItem(f"{val:.4g}" if isinstance(val, float) else str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(r, c, item)

        self._shape_label.setText(
            f"{len(df):,} rows total — showing first {len(preview)}"
        )

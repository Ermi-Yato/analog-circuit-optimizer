"""
Circuit Manager View — Phase 8.

Design system: GitHub-style dark theme.
  Backgrounds: 3-level depth (#0d1117 → #161b22 → #1c2128)
  Accent: blue (#388bfd)
  Semantic: green (trained), red (untrained/error), yellow (warning)
"""
from __future__ import annotations

import os

import joblib
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QPushButton,
    QTabWidget, QFormLayout, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QFrame, QAbstractItemView,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont

import registry.circuit_registry as reg


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# ── Design tokens ────────────────────────────────────────────────────────────
BG0      = "#0d1117"   # deepest background
BG1      = "#161b22"   # panel / sidebar
BG2      = "#1c2128"   # card / input surface
BORDER   = "#30363d"   # subtle borders
BORDER_F = "#388bfd"   # focused input border (blue)

TEXT     = "#e6edf3"   # primary text
TEXT_SUB = "#8b949e"   # secondary / labels
TEXT_DIM = "#484f58"   # placeholder / disabled

BLUE     = "#388bfd"   # primary action
BLUE_HOV = "#1f6feb"   # blue hover
BLUE_LT  = "#58a6ff"   # links / ghost button text

GREEN    = "#3fb950"   # success / trained
RED      = "#f85149"   # error / untrained
YELLOW   = "#d29922"   # warning

_PARAM_COLS  = ["name", "label", "unit", "min", "max", "default", "scale"]
_METRIC_COLS = ["name", "label", "unit", "optimize"]


# ── Style helpers ─────────────────────────────────────────────────────────────

def _divider() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {BORDER}; border: none;")
    return f


def _eyebrow(text: str) -> QLabel:
    """Small uppercase section label."""
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: {TEXT_DIM}; font-size: 10px; font-weight: 600; letter-spacing: 1.2px;"
    )
    return lbl


_INPUT_SS = f"""
    QLineEdit, QComboBox {{
        background: {BG2}; color: {TEXT};
        border: 1px solid {BORDER}; border-radius: 6px;
        padding: 0 10px; font-size: 12px; min-height: 32px;
    }}
    QLineEdit:focus {{ border-color: {BORDER_F}; }}
    QLineEdit:read-only {{ color: {TEXT_DIM}; }}
    QComboBox:focus {{ border-color: {BORDER_F}; }}
    QComboBox::drop-down {{ border: none; padding-right: 8px; }}
    QComboBox QAbstractItemView {{
        background: {BG2}; color: {TEXT};
        border: 1px solid {BORDER}; selection-background-color: {BLUE};
    }}
"""

_TABLE_SS = f"""
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
        font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
    }}
    QTableWidget::item {{ padding: 0 10px; border: none; }}
    QTableWidget::item:selected {{ background: {BLUE}22; color: {TEXT}; }}
    QTableCornerButton::section {{ background: {BG2}; border: none; }}
"""


def _btn_primary(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setFixedHeight(36)
    b.setMinimumWidth(130)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setStyleSheet(f"""
        QPushButton {{
            background: {BLUE}; color: #fff;
            border: none; border-radius: 6px;
            font-size: 13px; font-weight: 700; padding: 0 20px;
            letter-spacing: 0.3px;
        }}
        QPushButton:hover  {{ background: {BLUE_HOV}; border: 1px solid {BLUE_LT}; }}
        QPushButton:pressed {{ background: {BG2}; }}
        QPushButton:disabled {{ background: {BG2}; color: {TEXT_DIM}; }}
    """)
    return b


def _btn_secondary(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setFixedHeight(30)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setStyleSheet(f"""
        QPushButton {{
            background: {BG2}; color: {TEXT};
            border: 1px solid {BORDER}; border-radius: 6px;
            font-size: 11px; padding: 0 12px;
        }}
        QPushButton:hover  {{ border-color: {TEXT_SUB}; }}
        QPushButton:pressed {{ background: {BG0}; }}
    """)
    return b


def _btn_ghost(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setFixedHeight(34)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setStyleSheet(f"""
        QPushButton {{
            background: transparent; color: {BLUE_LT};
            border: 1px solid {BORDER}; border-radius: 6px;
            font-size: 12px; padding: 0 14px;
        }}
        QPushButton:hover  {{ border-color: {BLUE_LT}; background: {BLUE}18; }}
        QPushButton:pressed {{ background: {BG2}; }}
    """)
    return b


# ── Circuit list item ─────────────────────────────────────────────────────────

class _CircuitListItem(QWidget):
    def __init__(self, circuit: dict, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(8)

        name = QLabel(circuit["name"])
        name.setFont(QFont("Segoe UI", 10))
        name.setStyleSheet(f"color: {TEXT}; background: transparent;")
        layout.addWidget(name, stretch=1)

        if reg.model_exists(circuit["id"]):
            dot = QLabel("● Trained")
            dot.setStyleSheet(f"color: {GREEN}; font-size: 10px; background: transparent;")
        else:
            dot = QLabel("● Untrained")
            dot.setStyleSheet(f"color: {RED}; font-size: 10px; background: transparent;")
        layout.addWidget(dot)


# ── Editable tables ───────────────────────────────────────────────────────────

class _ParamTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, len(_PARAM_COLS), parent)
        self.setHorizontalHeaderLabels([c.capitalize() for c in _PARAM_COLS])
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setDefaultSectionSize(80)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.setStyleSheet(_TABLE_SS)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def load(self, params: list[dict]):
        self.setRowCount(0)
        for p in params:
            self._append(p)

    def _append(self, p: dict | None = None):
        row = self.rowCount()
        self.insertRow(row)
        self.setRowHeight(row, 34)
        d = p or {"name": "", "label": "", "unit": "", "min": "0",
                  "max": "1", "default": "0", "scale": "linear"}
        for col, key in enumerate(_PARAM_COLS):
            if key == "scale":
                cb = QComboBox()
                cb.addItems(["linear", "log"])
                cb.setCurrentText(str(d.get(key, "linear")))
                cb.setStyleSheet(_INPUT_SS)
                self.setCellWidget(row, col, cb)
            else:
                self.setItem(row, col, QTableWidgetItem(str(d.get(key, ""))))

    def add_row(self):    self._append()
    def delete_selected(self):
        for r in sorted({i.row() for i in self.selectedIndexes()}, reverse=True):
            self.removeRow(r)

    def to_list(self) -> list[dict]:
        out = []
        for row in range(self.rowCount()):
            entry = {}
            for col, key in enumerate(_PARAM_COLS):
                if key == "scale":
                    w = self.cellWidget(row, col)
                    entry[key] = w.currentText() if isinstance(w, QComboBox) else "linear"
                else:
                    item = self.item(row, col)
                    raw  = item.text().strip() if item else ""
                    if key in ("min", "max", "default"):
                        try:    entry[key] = float(raw)
                        except: entry[key] = 0.0
                    else:
                        entry[key] = raw
            out.append(entry)
        return out


class _MetricTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, len(_METRIC_COLS), parent)
        self.setHorizontalHeaderLabels([c.capitalize() for c in _METRIC_COLS])
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setDefaultSectionSize(100)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.setStyleSheet(_TABLE_SS)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def load(self, metrics: list[dict]):
        self.setRowCount(0)
        for m in metrics:
            self._append(m)

    def _append(self, m: dict | None = None):
        row = self.rowCount()
        self.insertRow(row)
        self.setRowHeight(row, 34)
        d = m or {"name": "", "label": "", "unit": "", "optimize": "maximize"}
        for col, key in enumerate(_METRIC_COLS):
            if key == "optimize":
                cb = QComboBox()
                cb.addItems(["maximize", "minimize"])
                cb.setCurrentText(str(d.get(key, "maximize")))
                cb.setStyleSheet(_INPUT_SS)
                self.setCellWidget(row, col, cb)
            else:
                self.setItem(row, col, QTableWidgetItem(str(d.get(key, ""))))

    def add_row(self):    self._append()
    def delete_selected(self):
        for r in sorted({i.row() for i in self.selectedIndexes()}, reverse=True):
            self.removeRow(r)

    def to_list(self) -> list[dict]:
        out = []
        for row in range(self.rowCount()):
            entry = {}
            for col, key in enumerate(_METRIC_COLS):
                if key == "optimize":
                    w = self.cellWidget(row, col)
                    entry[key] = w.currentText() if isinstance(w, QComboBox) else "maximize"
                else:
                    item = self.item(row, col)
                    entry[key] = item.text().strip() if item else ""
            out.append(entry)
        return out


# ── Editor panel ──────────────────────────────────────────────────────────────

class _EditorPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {BG0};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane  {{ border: none; background: {BG0}; }}
            QTabBar           {{ background: {BG1}; }}
            QTabBar::tab {{
                background: transparent; color: {TEXT_SUB};
                min-height: 52px; padding: 0 24px; border: none; font-size: 12px;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {TEXT}; font-weight: 600;
                border-bottom: 2px solid {BLUE};
            }}
            QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
        """)

        root.addWidget(self._tabs, stretch=1)
        root.addWidget(_divider())

        bar = QHBoxLayout()
        bar.setContentsMargins(20, 10, 20, 12)
        bar.setSpacing(10)
        self._btn_import = _btn_ghost("Import Model (.pkl)")
        self._btn_save   = _btn_primary("Save Circuit")
        self._btn_import.clicked.connect(self._on_import_model)
        self._btn_save.clicked.connect(self._on_save)
        bar.addWidget(self._btn_import)
        bar.addStretch()
        bar.addWidget(self._btn_save)
        root.addLayout(bar)

        self._build_details_tab()
        self._build_params_tab()
        self._build_metrics_tab()

        self._current_id: str | None = None
        self.setEnabled(False)

    # ── Tab builders ──────────────────────────────────────────────────────

    def _build_details_tab(self):
        tab = QWidget()
        tab.setStyleSheet(f"background: {BG0};")
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(20)

        # Identity section
        outer.addWidget(_eyebrow("Circuit Identity"))
        id_form = QFormLayout()
        id_form.setSpacing(12)
        id_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        id_form.setContentsMargins(0, 0, 0, 0)

        self._field_id   = QLineEdit(); self._field_id.setReadOnly(True)
        self._field_name = QLineEdit()
        self._field_desc = QLineEdit()
        for w in (self._field_id, self._field_name, self._field_desc):
            w.setStyleSheet(_INPUT_SS)
            w.setFixedHeight(34)

        for lbl, w in [("ID", self._field_id), ("Name", self._field_name), ("Description", self._field_desc)]:
            ql = QLabel(lbl)
            ql.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
            id_form.addRow(ql, w)
        outer.addLayout(id_form)

        outer.addWidget(_divider())

        # Simulation section
        outer.addWidget(_eyebrow("Simulation"))
        sim_form = QFormLayout()
        sim_form.setSpacing(12)
        sim_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        sim_form.setContentsMargins(0, 0, 0, 0)

        self._field_tmpl = QLineEdit()
        self._field_tmpl.setFixedHeight(34)
        self._field_tmpl.setStyleSheet(_INPUT_SS)
        btn_browse = _btn_secondary("Browse")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._browse_template)
        tmpl_row = QHBoxLayout()
        tmpl_row.setSpacing(6)
        tmpl_row.addWidget(self._field_tmpl)
        tmpl_row.addWidget(btn_browse)

        self._combo_simtype = QComboBox()
        self._combo_simtype.addItems(["ac", "transient", "dc"])
        self._combo_simtype.setFixedHeight(34)
        self._combo_simtype.setFixedWidth(160)
        self._combo_simtype.setStyleSheet(_INPUT_SS)

        for lbl, w in [("SPICE Template", tmpl_row), ("Simulation Type", self._combo_simtype)]:
            ql = QLabel(lbl)
            ql.setStyleSheet(f"color: {TEXT_SUB}; font-size: 12px;")
            if isinstance(w, QHBoxLayout):
                sim_form.addRow(ql, w)
            else:
                sim_form.addRow(ql, w)
        outer.addLayout(sim_form)
        outer.addStretch()
        self._tabs.addTab(tab, "Details")

    def _build_params_tab(self):
        tab = QWidget()
        tab.setStyleSheet(f"background: {BG0};")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        layout.addWidget(_eyebrow("Component Parameters"))
        self._param_table = _ParamTable()
        layout.addWidget(self._param_table, stretch=1)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        b_add = _btn_secondary("+ Add Row")
        b_del = _btn_secondary("Delete Selected")
        b_add.clicked.connect(self._param_table.add_row)
        b_del.clicked.connect(self._param_table.delete_selected)
        bar.addWidget(b_add)
        bar.addWidget(b_del)
        bar.addStretch()
        layout.addLayout(bar)
        self._tabs.addTab(tab, "Parameters")

    def _build_metrics_tab(self):
        tab = QWidget()
        tab.setStyleSheet(f"background: {BG0};")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        layout.addWidget(_eyebrow("Performance Metrics"))
        self._metric_table = _MetricTable()
        layout.addWidget(self._metric_table, stretch=1)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        b_add = _btn_secondary("+ Add Row")
        b_del = _btn_secondary("Delete Selected")
        b_add.clicked.connect(self._metric_table.add_row)
        b_del.clicked.connect(self._metric_table.delete_selected)
        bar.addWidget(b_add)
        bar.addWidget(b_del)
        bar.addStretch()
        layout.addLayout(bar)
        self._tabs.addTab(tab, "Metrics")

    # ── Data load / clear ─────────────────────────────────────────────────

    def load_circuit(self, circuit: dict):
        self._current_id = circuit["id"]
        self.setEnabled(True)
        self._field_id.setText(circuit["id"])
        self._field_name.setText(circuit.get("name", ""))
        self._field_desc.setText(circuit.get("description", ""))
        self._field_tmpl.setText(circuit.get("spice_template", ""))
        self._combo_simtype.setCurrentText(circuit.get("simulation_type", "ac"))
        self._param_table.load(circuit.get("parameters", []))
        self._metric_table.load(circuit.get("metrics", []))

    def clear(self):
        self._current_id = None
        self.setEnabled(False)
        for w in (self._field_id, self._field_name, self._field_desc, self._field_tmpl):
            w.clear()
        self._param_table.setRowCount(0)
        self._metric_table.setRowCount(0)

    # ── Actions ───────────────────────────────────────────────────────────

    def _browse_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SPICE Template", _PROJECT_ROOT,
            "Template files (*.template *.cir *.sp);;All files (*)"
        )
        if path:
            self._field_tmpl.setText(
                os.path.relpath(path, _PROJECT_ROOT).replace("\\", "/")
            )

    def _on_save(self):
        if not self._current_id:
            return
        circuit = {
            "id":              self._current_id,
            "name":            self._field_name.text().strip(),
            "description":     self._field_desc.text().strip(),
            "spice_template":  self._field_tmpl.text().strip(),
            "simulation_type": self._combo_simtype.currentText(),
            "parameters":      self._param_table.to_list(),
            "metrics":         self._metric_table.to_list(),
        }
        try:
            existing = reg.get(self._current_id)
            if "model" in existing:
                circuit["model"] = existing["model"]
        except KeyError:
            pass
        try:
            reg.register(circuit)
            QMessageBox.information(self, "Saved", f"'{circuit['name']}' saved.")
            self._notify_refresh()
        except ValueError as exc:
            QMessageBox.critical(self, "Validation Error", str(exc))

    def _on_import_model(self):
        if not self._current_id:
            return
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Import Surrogate Model", _PROJECT_ROOT,
            "Pickle files (*.pkl);;All files (*)"
        )
        if not model_path:
            return
        scaler_path, _ = QFileDialog.getOpenFileName(
            self, "Import Feature Scaler", _PROJECT_ROOT,
            "Pickle files (*.pkl);;All files (*)"
        )
        if not scaler_path:
            return
        try:
            joblib.load(model_path)
            joblib.load(scaler_path)
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", f"Could not load file:\n{exc}")
            return
        try:
            circuit = reg.get(self._current_id)
            circuit["model"] = {
                "surrogate_path": os.path.relpath(model_path,  _PROJECT_ROOT).replace("\\", "/"),
                "scaler_path":    os.path.relpath(scaler_path, _PROJECT_ROOT).replace("\\", "/"),
                "trained_on": None, "samples": None, "r2_scores": {},
            }
            reg.register(circuit)
            QMessageBox.information(self, "Imported", "Model imported successfully.")
            self._notify_refresh()
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _notify_refresh(self):
        p = self.parent()
        while p and not isinstance(p, CircuitManagerView):
            p = p.parent()
        if p:
            p.refresh_list()


# ── Main view ─────────────────────────────────────────────────────────────────

class CircuitManagerView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {BG0};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")
        splitter.addWidget(self._build_list_panel())
        self._editor = _EditorPanel(self)
        splitter.addWidget(self._editor)
        splitter.setSizes([230, 870])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        layout.addWidget(splitter)
        self.refresh_list()

    def _build_list_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(200)
        panel.setStyleSheet(f"background: {BG1};")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setFixedHeight(52)
        header.setStyleSheet(f"background: {BG1};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 16, 0)
        title = QLabel("Circuits")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT};")
        hl.addWidget(title)
        layout.addWidget(header)
        layout.addWidget(_divider())

        self._list = QListWidget()
        self._list.setStyleSheet(f"""
            QListWidget {{
                background: {BG1}; border: none; outline: none;
            }}
            QListWidget::item {{
                border-bottom: 1px solid {BORDER};
            }}
            QListWidget::item:selected {{ background: {BG2}; }}
            QListWidget::item:hover:!selected {{ background: {BG0}; }}
        """)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.currentItemChanged.connect(self._on_selected)
        layout.addWidget(self._list, stretch=1)

        layout.addWidget(_divider())

        btn_new = QPushButton("+ New Circuit")
        btn_new.setFixedHeight(42)
        btn_new.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_new.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {BLUE_LT};
                border: none; font-size: 12px; font-weight: 600;
            }}
            QPushButton:hover {{ background: {BG2}; }}
        """)
        btn_new.clicked.connect(self._on_new)
        layout.addWidget(btn_new)

        return panel

    def refresh_list(self):
        self._list.clear()
        try:
            circuits = reg.load_all()
        except Exception:
            return
        for cid, circuit in circuits.items():
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, cid)
            item.setSizeHint(QSize(0, 50))
            self._list.addItem(item)
            self._list.setItemWidget(item, _CircuitListItem(circuit))

    def _on_selected(self, current: QListWidgetItem, _prev):
        if current is None:
            self._editor.clear()
            return
        try:
            self._editor.load_circuit(reg.get(current.data(Qt.ItemDataRole.UserRole)))
        except KeyError:
            self._editor.clear()

    def _on_new(self):
        blank = {
            "id": "new_circuit", "name": "New Circuit", "description": "",
            "spice_template": "circuits/new_circuit/new_circuit.template",
            "simulation_type": "ac", "parameters": [], "metrics": [],
        }
        self._list.clearSelection()
        self._editor.load_circuit(blank)

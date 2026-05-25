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
from app.widgets.plot_widget import SchematicWidget
from app.widgets.circuit_drawings import get_drawing


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from app.design_system import (
    BG0, BG1, BG2, BORDER, BORDER_F, TEXT, TEXT_SUB, TEXT_DIM,
    BLUE, BLUE_HOV, BLUE_LT, GREEN, RED, YELLOW,
    divider as _ds_divider, eyebrow as _ds_eyebrow,
    input_ss as _ds_input_ss,
    table_ss, btn_primary_ss, btn_secondary_ss, btn_ghost_ss, btn_danger_ss,
    tab_ss,
)

_PARAM_COLS  = ["name", "label", "unit", "min", "max", "default", "scale"]
_METRIC_COLS = ["name", "label", "unit", "optimize"]


class _FlowLayout(QHBoxLayout):
    """Simple horizontal wrapping layout for suggestion chips."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 4, 0, 4)
        self.setSpacing(6)


# ── Style helpers ─────────────────────────────────────────────────────────────

def _divider() -> QFrame:
    return _ds_divider()


def _eyebrow(text: str) -> QLabel:
    return _ds_eyebrow(text)


_INPUT_SS = _ds_input_ss("QLineEdit, QComboBox") + f"""
    QLineEdit:read-only {{ color: {TEXT_DIM}; }}
"""

_TABLE_SS = table_ss()


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
        self._build_schematic_tab()

        self._current_id: str | None = None
        self._is_new: bool = False
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
        outer.addWidget(_divider())

        # Dynamic template reference panel
        outer.addWidget(_eyebrow("Template Reference"))
        self._template_guide = QLabel("Define parameters and metrics first.")
        self._template_guide.setStyleSheet(
            f"color: {TEXT_SUB}; font-size: 11px; background: {BG2}; "
            f"border: 1px solid {BORDER}; border-radius: 6px; padding: 12px 14px;"
        )
        self._template_guide.setWordWrap(True)
        self._template_guide.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        outer.addWidget(self._template_guide)

        validate_row = QHBoxLayout()
        self._btn_validate = _btn_secondary("Validate Template")
        self._btn_validate.clicked.connect(self._on_validate_template)
        self._validate_result = QLabel("")
        self._validate_result.setStyleSheet(f"font-size: 11px; color: {TEXT_SUB};")
        validate_row.addWidget(self._btn_validate)
        validate_row.addWidget(self._validate_result, stretch=1)
        outer.addLayout(validate_row)

        outer.addStretch()
        self._tabs.addTab(tab, "Details")
        self._tabs.currentChanged.connect(self._on_tab_changed)

    def _build_params_tab(self):
        tab = QWidget()
        tab.setStyleSheet(f"background: {BG0};")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        layout.addWidget(_eyebrow("Component Parameters"))
        self._param_table = _ParamTable()
        self._param_table.itemChanged.connect(self._refresh_placeholder_bar)
        self._param_table.model().rowsInserted.connect(self._refresh_placeholder_bar)
        self._param_table.model().rowsRemoved.connect(self._refresh_placeholder_bar)
        self._param_table.model().rowsInserted.connect(self._refresh_param_suggestions)
        self._param_table.model().rowsRemoved.connect(self._refresh_param_suggestions)
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

        self._placeholder_bar = QLabel("Placeholders will appear here as you add parameters.")
        self._placeholder_bar.setStyleSheet(
            f"color: {TEXT_SUB}; font-size: 11px; background: {BG2}; "
            f"border: 1px solid {BORDER}; border-radius: 5px; padding: 7px 12px;"
        )
        self._placeholder_bar.setWordWrap(True)
        self._placeholder_bar.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._placeholder_bar)

        # Parameter suggestions from template
        layout.addWidget(_divider())
        layout.addWidget(_eyebrow("Suggestions from template  (click to add)"))
        self._param_suggestion_wrap = QWidget()
        self._param_suggestion_wrap.setStyleSheet("background: transparent;")
        self._param_suggestion_layout = _FlowLayout(self._param_suggestion_wrap)
        layout.addWidget(self._param_suggestion_wrap)

        # Wire template field changes to refresh suggestions
        self._field_tmpl.textChanged.connect(self._refresh_param_suggestions)

        self._tabs.addTab(tab, "Parameters")

    def _refresh_placeholder_bar(self, *_args):
        params = self._param_table.to_list()
        names  = [p["name"].strip() for p in params if p.get("name", "").strip()]
        if not names:
            self._placeholder_bar.setText(
                "Placeholders will appear here as you add parameters."
            )
            return
        placeholders = ["{" + n.replace("_", "").upper() + "_VAL}" for n in names]
        pairs = "     ".join(
            f"{n}  ->  {ph}" for n, ph in zip(names, placeholders)
        )
        self._placeholder_bar.setText(f"Placeholders:   {pairs}")

    def _refresh_param_suggestions(self, *_args):
        import re
        # Clear existing chips
        while self._param_suggestion_layout.count():
            item = self._param_suggestion_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        tmpl_rel = self._field_tmpl.text().strip()
        if not tmpl_rel:
            self._param_suggestion_layout.addStretch()
            return

        tmpl_path = os.path.join(_PROJECT_ROOT, tmpl_rel)
        if not os.path.isfile(tmpl_path):
            self._param_suggestion_layout.addStretch()
            return

        try:
            with open(tmpl_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            self._param_suggestion_layout.addStretch()
            return

        # Extract placeholder names: {SOMENAME_VAL} -> "SOMENAME"
        found = re.findall(r'\{([A-Z0-9]+)_VAL\}', content)
        if not found:
            self._param_suggestion_layout.addStretch()
            return

        # Build set of already-defined param names (normalised: strip _ uppercase)
        existing_params = self._param_table.to_list()
        existing_keys = {
            p.get("name", "").replace("_", "").upper().strip()
            for p in existing_params
        }

        seen = set()
        for placeholder_name in found:
            if placeholder_name in seen:
                continue
            seen.add(placeholder_name)
            if placeholder_name in existing_keys:
                continue
            chip = QPushButton(f"+ {placeholder_name}")
            chip.setFixedHeight(26)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setStyleSheet(
                f"QPushButton {{ background: {BG2}; color: {TEXT_SUB}; "
                f"border: 1px solid {BORDER}; border-radius: 5px; "
                f"font-size: 10px; padding: 0 10px; }}"
                f"QPushButton:hover {{ color: {TEXT}; border-color: {BLUE}; background: {BG1}; }}"
            )
            chip.clicked.connect(lambda _=False, n=placeholder_name: self._add_preset_param(n))
            self._param_suggestion_layout.addWidget(chip)

        self._param_suggestion_layout.addStretch()

    def _add_preset_param(self, placeholder_name: str):
        """Add a param row pre-filled from a template placeholder chip."""
        # Don't add if already in table (normalised comparison)
        existing = {
            p.get("name", "").replace("_", "").upper().strip()
            for p in self._param_table.to_list()
        }
        if placeholder_name in existing:
            return
        preset = {
            "name": placeholder_name,
            "label": placeholder_name.replace("_", " ").title(),
            "unit": "",
            "min": "0",
            "max": "1",
            "default": "0",
            "scale": "linear",
        }
        self._param_table._append(preset)
        self._refresh_param_suggestions()

    def _build_metrics_tab(self):
        _PRESETS = {
            "ac": [
                {"name": "Peak_Gain_dB",      "label": "Voltage Gain",    "unit": "dB",  "optimize": "maximize"},
                {"name": "Bandwidth_Hz",       "label": "Bandwidth",       "unit": "Hz",  "optimize": "maximize"},
                {"name": "Phase_Margin_deg",   "label": "Phase Margin",    "unit": "deg", "optimize": "maximize"},
                {"name": "Cutoff_Freq_Hz",     "label": "Cutoff Frequency","unit": "Hz",  "optimize": "maximize"},
                {"name": "Q_factor",           "label": "Q Factor",        "unit": "",    "optimize": "maximize"},
                {"name": "CMRR_dB",            "label": "CMRR",            "unit": "dB",  "optimize": "maximize"},
                {"name": "Transimpedance_dBOhm","label": "Transimpedance", "unit": "dBΩ", "optimize": "maximize"},
            ],
            "transient": [
                {"name": "Output_Swing_V",     "label": "Output Swing",    "unit": "V",   "optimize": "maximize"},
                {"name": "THD_percent",        "label": "Total Harmonic Distortion", "unit": "%", "optimize": "minimize"},
                {"name": "Efficiency_percent", "label": "Efficiency",      "unit": "%",   "optimize": "maximize"},
                {"name": "Rise_Time_s",        "label": "Rise Time",       "unit": "s",   "optimize": "minimize"},
                {"name": "Settling_Time_s",    "label": "Settling Time",   "unit": "s",   "optimize": "minimize"},
            ],
            "dc": [
                {"name": "Operating_Point_V",  "label": "Operating Point", "unit": "V",   "optimize": "maximize"},
                {"name": "Current_Draw_A",     "label": "Quiescent Current","unit": "A",  "optimize": "minimize"},
            ],
        }

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

        # Suggestions
        layout.addWidget(_divider())
        layout.addWidget(_eyebrow("Suggestions  (click to add)"))
        self._suggestion_wrap = QWidget()
        self._suggestion_wrap.setStyleSheet("background: transparent;")
        self._suggestion_layout = _FlowLayout(self._suggestion_wrap)
        layout.addWidget(self._suggestion_wrap)

        self._metric_presets = _PRESETS
        self._combo_simtype.currentTextChanged.connect(self._refresh_metric_suggestions)

        self._tabs.addTab(tab, "Metrics")

    def _build_schematic_tab(self):
        tab = QWidget()
        tab.setStyleSheet(f"background: {BG0};")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.addWidget(_eyebrow("Circuit Schematic"))
        hdr.addStretch()
        self._btn_import_schematic = _btn_secondary("Import Image")
        self._btn_import_schematic.clicked.connect(self._on_import_schematic)
        self._btn_clear_schematic = _btn_secondary("Clear")
        self._btn_clear_schematic.clicked.connect(self._on_clear_schematic)
        hdr.addWidget(self._btn_import_schematic)
        hdr.addWidget(self._btn_clear_schematic)
        layout.addLayout(hdr)

        # Schemdraw canvas (built-in circuits)
        self._schematic_plot = SchematicWidget()
        layout.addWidget(self._schematic_plot, stretch=1)

        # Image label (user-imported)
        self._schematic_image_lbl = QLabel()
        self._schematic_image_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._schematic_image_lbl.setStyleSheet(f"background: {BG1}; border-radius: 6px;")
        layout.addWidget(self._schematic_image_lbl, stretch=1)

        # Placeholder
        self._schematic_placeholder = QLabel(
            "No schematic available.\nImport a PNG / JPG / SVG image using the button above."
        )
        self._schematic_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._schematic_placeholder.setStyleSheet(f"color: {TEXT_DIM}; font-size: 13px;")
        layout.addWidget(self._schematic_placeholder)

        self._schematic_image_path: str | None = None
        self._tabs.addTab(tab, "Schematic")

    # ── Data load / clear ─────────────────────────────────────────────────

    def load_circuit(self, circuit: dict, is_new: bool = False):
        self._current_id = circuit["id"]
        self._is_new = is_new
        self.setEnabled(True)
        self._field_id.setText(circuit["id"])
        self._field_id.setReadOnly(not is_new)
        self._field_id.setStyleSheet(
            _INPUT_SS if is_new
            else _INPUT_SS + f"QLineEdit {{ color: {TEXT_DIM}; }}"
        )
        self._field_id.setPlaceholderText("e.g. my_filter (no spaces)" if is_new else "")
        self._field_name.setText(circuit.get("name", ""))
        self._field_desc.setText(circuit.get("description", ""))
        self._field_tmpl.setText(circuit.get("spice_template", ""))
        self._combo_simtype.setCurrentText(circuit.get("simulation_type", "ac"))
        self._param_table.load(circuit.get("parameters", []))
        self._metric_table.load(circuit.get("metrics", []))
        img_rel = circuit.get("schematic_image", "")
        self._schematic_image_path = (
            os.path.join(_PROJECT_ROOT, img_rel) if img_rel else None
        )
        self._refresh_schematic(circuit["id"])
        self._update_template_guide()
        self._refresh_placeholder_bar()
        self._refresh_param_suggestions()
        self._refresh_metric_suggestions()

    def _refresh_schematic(self, circuit_id: str):
        # Priority: built-in schemdraw > user image > placeholder
        fig = get_drawing(circuit_id)
        if fig is not None:
            self._schematic_plot.set_figure(fig)
            self._schematic_plot.show()
            self._schematic_image_lbl.hide()
            self._schematic_placeholder.hide()
            return

        if self._schematic_image_path:
            self._show_image(self._schematic_image_path)
            return

        self._schematic_plot.hide()
        self._schematic_image_lbl.hide()
        self._schematic_placeholder.show()

    def _show_image(self, path: str):
        from PySide6.QtGui import QPixmap
        pix = QPixmap(path)
        if pix.isNull():
            self._schematic_plot.hide()
            self._schematic_image_lbl.hide()
            self._schematic_placeholder.setText(f"Could not load image:\n{path}")
            self._schematic_placeholder.show()
            return
        self._schematic_image_lbl.setPixmap(
            pix.scaled(self._schematic_image_lbl.size(),
                       Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        )
        self._schematic_plot.hide()
        self._schematic_image_lbl.show()
        self._schematic_placeholder.hide()

    def _on_import_schematic(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Schematic Image", _PROJECT_ROOT,
            "Images (*.png *.jpg *.jpeg *.svg *.bmp);;All files (*)"
        )
        if not path:
            return
        self._schematic_image_path = path
        self._show_image(path)

    def _on_clear_schematic(self):
        self._schematic_image_path = None
        circuit_id = self._current_id or ""
        self._refresh_schematic(circuit_id)

    def clear(self):
        self._current_id = None
        self.setEnabled(False)
        for w in (self._field_id, self._field_name, self._field_desc, self._field_tmpl):
            w.clear()
        self._param_table.setRowCount(0)
        self._metric_table.setRowCount(0)
        self._schematic_image_path = None
        self._schematic_plot.hide()
        self._schematic_image_lbl.hide()
        self._schematic_placeholder.show()

    # ── Actions ───────────────────────────────────────────────────────────

    def _refresh_metric_suggestions(self, sim_type: str = ""):
        sim = sim_type or self._combo_simtype.currentText()
        presets = self._metric_presets.get(sim, [])

        # Clear existing chips
        while self._suggestion_layout.count():
            item = self._suggestion_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        existing_names = {
            m.get("name", "").strip()
            for m in self._metric_table.to_list()
        }

        for preset in presets:
            name = preset["name"]
            if name in existing_names:
                continue
            chip = QPushButton(f"+ {name}")
            chip.setFixedHeight(26)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setStyleSheet(
                f"QPushButton {{ background: {BG2}; color: {TEXT_SUB}; "
                f"border: 1px solid {BORDER}; border-radius: 5px; "
                f"font-size: 10px; padding: 0 10px; }}"
                f"QPushButton:hover {{ color: {TEXT}; border-color: {BLUE}; background: {BG1}; }}"
            )
            chip.clicked.connect(lambda _=False, p=preset: self._add_preset_metric(p))
            self._suggestion_layout.addWidget(chip)

        self._suggestion_layout.addStretch()

    def _add_preset_metric(self, preset: dict):
        # Don't add duplicates
        existing = {m.get("name", "").strip() for m in self._metric_table.to_list()}
        if preset["name"] in existing:
            return
        self._metric_table._append(preset)
        # Remove the chip for this metric
        self._refresh_metric_suggestions()

    def _on_tab_changed(self, index: int):
        if index == 0:   # Details tab
            self._update_template_guide()

    def _update_template_guide(self):
        params  = self._param_table.to_list()
        metrics = self._metric_table.to_list()
        sim     = self._combo_simtype.currentText()

        lines = []

        # Parameter placeholders
        if params:
            lines.append("PARAMETER PLACEHOLDERS  (use these in your netlist)")
            for p in params:
                name = p.get("name", "").strip()
                if not name:
                    continue
                placeholder = "{" + name.replace("_", "").upper() + "_VAL}"
                lines.append(f"  {name:<18} ->  {placeholder}")
        else:
            lines.append("No parameters defined yet.")

        lines.append("")

        # Required wrdata line
        if sim == "ac":
            lines.append("REQUIRED IN .control BLOCK")
            lines.append("  wrdata ngspice_simulation_output.txt frequency v(<out_node>)")
            lines.append("  (replace <out_node> with your output node name)")
        elif sim == "transient":
            lines.append("REQUIRED IN .control BLOCK")
            lines.append("  wrdata ngspice_simulation_output.txt time v(<out_node>)")
            lines.append("  (replace <out_node> with your output node name)")

        lines.append("")

        # Defined metrics
        if metrics:
            lines.append("DEFINED OUTPUT METRICS")
            for m in metrics:
                name = m.get("name", "").strip()
                opt  = m.get("optimize", "maximize")
                unit = m.get("unit", "")
                if name:
                    lines.append(f"  {name}  [{unit}]  ({opt})")
        else:
            lines.append("No metrics defined yet.")

        self._template_guide.setText("\n".join(lines))

    def _on_validate_template(self):
        tmpl_rel = self._field_tmpl.text().strip()
        if not tmpl_rel:
            self._validate_result.setStyleSheet(f"font-size: 11px; color: {YELLOW};")
            self._validate_result.setText("No template path set.")
            return

        tmpl_path = os.path.join(_PROJECT_ROOT, tmpl_rel)
        if not os.path.isfile(tmpl_path):
            self._validate_result.setStyleSheet(f"font-size: 11px; color: {RED};")
            self._validate_result.setText(f"File not found: {tmpl_rel}")
            return

        with open(tmpl_path, "r", encoding="utf-8") as f:
            content = f.read()

        params = self._param_table.to_list()
        if not params:
            self._validate_result.setStyleSheet(f"font-size: 11px; color: {YELLOW};")
            self._validate_result.setText("No parameters defined yet.")
            return

        missing = []
        for p in params:
            name = p.get("name", "").strip()
            if not name:
                continue
            placeholder = "{" + name.replace("_", "").upper() + "_VAL}"
            if placeholder not in content:
                missing.append(placeholder)

        if missing:
            self._validate_result.setStyleSheet(f"font-size: 11px; color: {RED};")
            self._validate_result.setText("Missing: " + "  ".join(missing))
        else:
            self._validate_result.setStyleSheet(f"font-size: 11px; color: {GREEN};")
            self._validate_result.setText("All placeholders found.")

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
        # Collect and validate all fields before touching the registry
        errors = []

        # ID
        if self._is_new:
            circuit_id = self._field_id.text().strip()
            if not circuit_id:
                errors.append("Circuit ID is required.")
            elif " " in circuit_id:
                errors.append("Circuit ID must not contain spaces.")
        else:
            circuit_id = self._current_id
            if not circuit_id:
                return   # editor not loaded

        name = self._field_name.text().strip()
        if not name:
            errors.append("Name is required.")

        tmpl = self._field_tmpl.text().strip()
        if not tmpl:
            errors.append("SPICE template path is required.")
        elif not os.path.isfile(os.path.join(_PROJECT_ROOT, tmpl)):
            errors.append(f"Template file not found: {tmpl}")

        params = self._param_table.to_list()
        if not params or all(not p.get("name", "").strip() for p in params):
            errors.append("At least one parameter is required.")

        metrics = self._metric_table.to_list()
        if not metrics or all(not m.get("name", "").strip() for m in metrics):
            errors.append("At least one metric is required.")

        if errors:
            QMessageBox.warning(
                self, "Cannot Save",
                "Please fix the following before saving:\n\n" +
                "\n".join(f"  • {e}" for e in errors)
            )
            return

        self._current_id = circuit_id
        circuit = {
            "id":              circuit_id,
            "name":            name,
            "description":     self._field_desc.text().strip(),
            "spice_template":  tmpl,
            "simulation_type": self._combo_simtype.currentText(),
            "parameters":      params,
            "metrics":         metrics,
        }
        if self._schematic_image_path:
            circuit["schematic_image"] = os.path.relpath(
                self._schematic_image_path, _PROJECT_ROOT
            ).replace("\\", "/")
        try:
            existing = reg.get(circuit_id)
            if "model" in existing:
                circuit["model"] = existing["model"]
        except KeyError:
            pass
        try:
            reg.register(circuit)
            self._is_new = False
            self._field_id.setReadOnly(True)
            self._field_id.setStyleSheet(_INPUT_SS + f"QLineEdit {{ color: {TEXT_DIM}; }}")
            QMessageBox.information(self, "Saved", f"'{name}' saved.")
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
            "id": "", "name": "", "description": "",
            "spice_template": "",
            "simulation_type": "ac", "parameters": [], "metrics": [],
        }
        self._list.clearSelection()
        self._editor.load_circuit(blank, is_new=True)
        self._editor._field_id.setFocus()

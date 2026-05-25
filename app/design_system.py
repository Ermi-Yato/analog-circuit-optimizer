"""
Centralized design tokens and reusable stylesheet builders.

All views import from here — single source of truth for colors, spacing,
and component styling. GitHub-dark inspired palette.
"""
from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel
from PySide6.QtGui import QFont  # noqa: F401 — re-exported for views

# ── Backgrounds (3-level depth) ──────────────────────────────────────────────
BG0 = "#0d1117"   # deepest — main canvas / window
BG1 = "#161b22"   # mid — panels, sidebars, toolbars
BG2 = "#21262d"   # surface — cards, inputs, buttons  (was #1c2128, bumped for contrast)

# ── Borders ──────────────────────────────────────────────────────────────────
BORDER   = "#3d444d"   # subtle borders / dividers  (was #30363d, lightened)
BORDER_F = "#388bfd"   # focused input border (= BLUE)

# ── Text (3-tier hierarchy) ──────────────────────────────────────────────────
TEXT     = "#f0f3f6"   # primary text                 (was #e6edf3, brighter)
TEXT_SUB = "#b1bac4"   # secondary / labels           (was #9198a1, noticeably brighter)
TEXT_DIM = "#8b949e"   # placeholder / disabled        (was #6e7681, brighter)

# Aliases — some views used shorter names
TEXT_S = TEXT_SUB
TEXT_D = TEXT_DIM

# ── Accent colors ────────────────────────────────────────────────────────────
BLUE     = "#388bfd"   # primary action
BLUE_HOV = "#1f6feb"   # blue hover
BLUE_LT  = "#58a6ff"   # links / ghost-button text

GREEN  = "#3fb950"   # success / trained
RED    = "#f85149"   # error / untrained
YELLOW = "#d29922"   # warning / in-progress

# ── Chart accent palette (per-metric) ────────────────────────────────────────
METRIC_COLORS = ["#388bfd", "#3fb950", "#f9e2af", "#f38ba8", "#cba6f7"]


# ── Reusable widget builders ─────────────────────────────────────────────────

def divider(vertical: bool = False) -> QFrame:
    """1px line separator."""
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine if vertical else QFrame.Shape.HLine)
    if vertical:
        f.setFixedWidth(1)
    else:
        f.setFixedHeight(1)
    f.setStyleSheet(f"background: {BORDER}; border: none;")
    return f


def eyebrow(text: str) -> QLabel:
    """Uppercase category label in dim text."""
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: {TEXT_DIM}; font-size: 10px; font-weight: 600; letter-spacing: 1.2px;"
    )
    return lbl


# ── Stylesheet snippets ─────────────────────────────────────────────────────

def input_ss(extra_widgets: str = "QComboBox, QDoubleSpinBox, QSpinBox") -> str:
    """Standard input-field stylesheet."""
    return f"""
        {extra_widgets} {{
            background: {BG2}; color: {TEXT};
            border: 1px solid {BORDER}; border-radius: 6px;
            padding: 0 10px; font-size: 12px; min-height: 34px;
        }}
        {extra_widgets}:focus {{ border-color: {BORDER_F}; }}
        QComboBox QAbstractItemView {{
            background: {BG2}; color: {TEXT};
            selection-background-color: {BLUE}; selection-color: #fff;
            border: 1px solid {BORDER}; outline: none;
        }}
        QComboBox::drop-down {{
            border: none; width: 24px;
        }}
        QComboBox::down-arrow {{
            image: none; border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid {TEXT_SUB};
        }}
    """


def btn_primary_ss() -> str:
    return f"""
        QPushButton {{
            background: {BLUE}; color: #ffffff;
            border: none; border-radius: 6px;
            font-size: 13px; font-weight: 700;
            padding: 0 20px; min-height: 34px;
        }}
        QPushButton:hover  {{ background: {BLUE_HOV}; border: 1px solid {BLUE_LT}; }}
        QPushButton:pressed {{ background: {BG2}; }}
        QPushButton:disabled {{ background: {BG2}; color: {TEXT_DIM}; }}
    """


def btn_secondary_ss() -> str:
    return f"""
        QPushButton {{
            background: {BG2}; color: {TEXT};
            border: 1px solid {BORDER}; border-radius: 6px;
            font-size: 12px; font-weight: 500;
            padding: 0 14px; min-height: 34px;
        }}
        QPushButton:hover  {{ border-color: {TEXT_SUB}; }}
        QPushButton:pressed {{ background: {BG0}; }}
        QPushButton:disabled {{ color: {TEXT_DIM}; }}
    """


def btn_ghost_ss() -> str:
    return f"""
        QPushButton {{
            background: transparent; color: {BLUE_LT};
            border: 1px solid {BORDER}; border-radius: 6px;
            font-size: 12px; padding: 0 14px; min-height: 34px;
        }}
        QPushButton:hover  {{ border-color: {BLUE_LT}; background: {BLUE}18; }}
        QPushButton:pressed {{ background: {BG2}; }}
        QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {BG2}; }}
    """


def btn_danger_ss() -> str:
    return f"""
        QPushButton {{
            background: transparent; color: {RED};
            border: 1px solid {BORDER}; border-radius: 6px;
            font-size: 12px; padding: 0 14px; min-height: 34px;
        }}
        QPushButton:hover  {{ border-color: {RED}; background: {RED}18; }}
        QPushButton:pressed {{ background: {BG2}; }}
    """


def table_ss() -> str:
    return f"""
        QTableWidget {{
            background: {BG1}; color: {TEXT};
            border: 1px solid {BORDER}; border-radius: 6px;
            gridline-color: {BORDER};
            alternate-background-color: {BG0};
            font-size: 12px; selection-color: {TEXT};
            selection-background-color: {BLUE}22;
        }}
        QHeaderView::section {{
            background: {BG2}; color: {TEXT_SUB};
            border: none; border-bottom: 1px solid {BORDER};
            padding: 0 10px; min-height: 32px;
            font-size: 11px; font-weight: 600;
        }}
        QScrollBar:vertical {{
            background: {BG1}; width: 8px; border: none;
        }}
        QScrollBar::handle:vertical {{
            background: {BORDER}; border-radius: 4px; min-height: 24px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
    """


def progress_bar_ss() -> str:
    return f"""
        QProgressBar {{
            background: {BG2}; border: 1px solid {BORDER};
            border-radius: 4px; text-align: center;
            color: {TEXT}; font-size: 11px; min-height: 22px;
        }}
        QProgressBar::chunk {{
            background: {BLUE}; border-radius: 3px;
        }}
    """


def tab_ss() -> str:
    return f"""
        QTabWidget::pane {{
            border: none; background: {BG0};
        }}
        QTabBar {{
            background: {BG1};
        }}
        QTabBar::tab {{
            background: transparent; color: {TEXT_SUB};
            padding: 10px 18px; font-size: 12px;
            border: none; border-bottom: 2px solid transparent;
        }}
        QTabBar::tab:selected {{
            color: {TEXT}; border-bottom: 2px solid {BLUE};
            font-weight: 600;
        }}
        QTabBar::tab:hover:!selected {{
            color: {TEXT}; border-bottom: 2px solid {BORDER};
        }}
    """


def toolbar_ss() -> str:
    return f"""
        background: {BG1};
        border-bottom: 1px solid {BORDER};
    """

import os

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QStatusBar, QLabel, QPushButton, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon, QPixmap

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGO_PATH    = os.path.join(_PROJECT_ROOT, "logo.jpg")

from app.views.dashboard_view       import DashboardView
from app.views.circuit_manager_view import CircuitManagerView
from app.views.dataset_view         import DatasetView
from app.views.training_view        import TrainingView
from app.views.optimizer_view       import OptimizerView
from app.views.results_view         import ResultsView
from app.design_system import (
    BG0, BG1, BG2, BORDER, TEXT, TEXT_SUB as TEXT_S, TEXT_DIM as TEXT_D,
    BLUE, GREEN, RED,
)

_SIDEBAR_W = 196
_ITEM_H    = 40

_NAV_ITEMS = [
    ("Dashboard",       0),
    ("Circuit Manager", 1),
    ("Dataset",         2),
    ("Training",        3),
    ("Optimizer",       4),
    ("Results",         5),
]


class _NavButton(QPushButton):
    _BASE = (
        f"QPushButton {{ background: transparent; color: {TEXT_S}; border: none; "
        f"text-align: left; padding: 0 20px; font-size: 12px; border-radius: 0; }}"
    )
    _HOV  = (
        f"QPushButton {{ background: {BG2}; color: {TEXT}; border: none; "
        f"text-align: left; padding: 0 20px; font-size: 12px; border-radius: 0; }}"
    )
    _SEL  = (
        f"QPushButton {{ background: {BG2}; color: {TEXT}; border: none; "
        f"border-left: 2px solid {BLUE}; text-align: left; padding: 0 18px; "
        f"font-size: 12px; font-weight: 600; border-radius: 0; }}"
    )

    def __init__(self, label: str, parent=None):
        super().__init__(label, parent)
        self.setFixedHeight(_ITEM_H)
        self.setFont(QFont("Segoe UI", 10))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sel = False
        self.setStyleSheet(self._BASE)

    def setSelected(self, v: bool):
        self._sel = v
        self.setStyleSheet(self._SEL if v else self._BASE)

    def enterEvent(self, e):
        if not self._sel:
            self.setStyleSheet(self._HOV)
        super().enterEvent(e)

    def leaveEvent(self, e):
        if not self._sel:
            self.setStyleSheet(self._BASE)
        super().leaveEvent(e)


class _ToggleButton(QPushButton):
    """Small hamburger button, always accessible."""
    def __init__(self, parent=None):
        super().__init__("☰", parent)
        self.setFixedSize(32, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Toggle sidebar")
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {TEXT_S};
                border: none; border-radius: 6px;
                font-size: 15px;
            }}
            QPushButton:hover {{ background: {BG2}; color: {TEXT}; }}
            QPushButton:pressed {{ background: {BORDER}; }}
        """)


class MainWindow(QMainWindow):
    def __init__(self, ngspice_available: bool = False):
        super().__init__()
        self.setWindowTitle("Xtal")
        self.setWindowIcon(QIcon(_LOGO_PATH))
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(f"background: {BG0};")

        self._ngspice_available = ngspice_available
        self._nav_buttons: list[_NavButton] = []
        self._sidebar_visible = False   # starts hidden

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._sidebar = self._build_sidebar()
        root.addWidget(self._sidebar)

        self._sep = QFrame()
        self._sep.setFrameShape(QFrame.Shape.VLine)
        self._sep.setFixedWidth(1)
        self._sep.setStyleSheet(f"background: {BORDER}; border: none;")
        root.addWidget(self._sep)

        self._stack = self._build_stack()
        root.addWidget(self._stack)

        # Floating toggle button — always on top when sidebar is hidden
        self._float_toggle = _ToggleButton(central)
        self._float_toggle.clicked.connect(self._toggle_sidebar)
        self._float_toggle.raise_()
        self._float_toggle.move(6, 12)

        self._build_status_bar()
        self._select(0)

        # Apply initial hidden state without animation
        self._sidebar.setVisible(False)
        self._sep.setVisible(False)
        self._float_toggle.show()

    # ── Sidebar build ─────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(_SIDEBAR_W)
        sidebar.setStyleSheet(f"background: {BG1};")

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title row with logo, name, and collapse button
        title_row = QWidget()
        title_row.setFixedHeight(56)
        title_row.setStyleSheet(f"background: {BG1};")
        tr = QHBoxLayout(title_row)
        tr.setContentsMargins(12, 0, 8, 0)
        tr.setSpacing(8)

        logo_lbl = QLabel()
        pix = QPixmap(_LOGO_PATH)
        if not pix.isNull():
            logo_lbl.setPixmap(pix.scaled(26, 26,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        logo_lbl.setStyleSheet("background: transparent;")
        tr.addWidget(logo_lbl)

        name_lbl = QLabel("Xtal")
        name_lbl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        name_lbl.setStyleSheet(f"color: {TEXT}; background: transparent;")
        tr.addWidget(name_lbl, 1)

        # Collapse button inside sidebar header
        collapse_btn = _ToggleButton()
        collapse_btn.clicked.connect(self._toggle_sidebar)
        tr.addWidget(collapse_btn)

        layout.addWidget(title_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {BORDER}; border: none;")
        layout.addWidget(sep)
        layout.addSpacing(8)

        for label, idx in _NAV_ITEMS:
            btn = _NavButton(label)
            btn.clicked.connect(lambda _c, i=idx: self._select(i))
            layout.addWidget(btn)
            self._nav_buttons.append(btn)

        layout.addStretch()

        ver = QLabel("  v0.1-dev")
        ver.setStyleSheet(f"color: {TEXT_D}; font-size: 11px; padding: 10px 0;")
        layout.addWidget(ver)

        return sidebar

    # ── Stack build ───────────────────────────────────────────────────────────

    def _build_stack(self) -> QStackedWidget:
        stack = QStackedWidget()
        self._dashboard = DashboardView(
            navigate_cb=self.navigate_to,
            ngspice_available=self._ngspice_available,
        )
        stack.addWidget(self._dashboard)
        stack.addWidget(CircuitManagerView())
        stack.addWidget(DatasetView())
        stack.addWidget(TrainingView())

        optimizer_view = OptimizerView()
        results_view   = ResultsView()
        optimizer_view.optimization_complete.connect(results_view.set_optimized_result)

        stack.addWidget(optimizer_view)
        stack.addWidget(results_view)
        return stack

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self):
        bar = QStatusBar()
        bar.setStyleSheet(f"background: {BG1}; border-top: 1px solid {BORDER};")
        self.setStatusBar(bar)

        if self._ngspice_available:
            lbl = QLabel("ngspice: available")
            lbl.setStyleSheet(f"color: {GREEN}; padding: 0 10px; font-size: 11px;")
        else:
            lbl = QLabel("ngspice: NOT FOUND  simulation disabled")
            lbl.setStyleSheet(f"color: {RED}; padding: 0 10px; font-size: 11px;")

        bar.addPermanentWidget(lbl)

    # ── Toggle ────────────────────────────────────────────────────────────────

    def _toggle_sidebar(self):
        self._sidebar_visible = not self._sidebar_visible
        self._sidebar.setVisible(self._sidebar_visible)
        self._sep.setVisible(self._sidebar_visible)
        # Float button: visible only when sidebar is hidden
        self._float_toggle.setVisible(not self._sidebar_visible)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _select(self, index: int):
        self._stack.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_buttons):
            btn.setSelected(i == index)
        if index == 0:
            self._dashboard.refresh()

    def navigate_to(self, index: int, circuit_id: str | None = None):
        self._select(index)
        if circuit_id:
            view = self._stack.widget(index)
            if hasattr(view, "select_circuit"):
                view.select_circuit(circuit_id)

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QStatusBar, QLabel, QPushButton, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from app.views.dashboard_view       import DashboardView
from app.views.circuit_manager_view import CircuitManagerView
from app.views.dataset_view         import DatasetView
from app.views.training_view        import TrainingView
from app.views.optimizer_view       import OptimizerView
from app.views.results_view         import ResultsView

# Design tokens (shared with views)
BG0    = "#0d1117"
BG1    = "#161b22"
BG2    = "#1c2128"
BORDER = "#30363d"
TEXT   = "#e6edf3"
TEXT_S = "#8b949e"
TEXT_D = "#484f58"
BLUE   = "#388bfd"
GREEN  = "#3fb950"
RED    = "#f85149"

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


class MainWindow(QMainWindow):
    def __init__(self, ngspice_available: bool = False):
        super().__init__()
        self.setWindowTitle("Xtal")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(f"background: {BG0};")

        self._ngspice_available = ngspice_available
        self._nav_buttons: list[_NavButton] = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(1)
        sep.setStyleSheet(f"background: {BORDER}; border: none;")
        root.addWidget(sep)

        self._stack = self._build_stack()
        root.addWidget(self._stack)

        self._build_status_bar()
        self._select(0)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(_SIDEBAR_W)
        sidebar.setStyleSheet(f"background: {BG1};")

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title = QLabel("  Xtal")
        title.setFixedHeight(56)
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT}; background: {BG1}; padding-left: 4px;")
        layout.addWidget(title)

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

    def _build_stack(self) -> QStackedWidget:
        stack = QStackedWidget()
        stack.addWidget(DashboardView())
        stack.addWidget(CircuitManagerView())
        stack.addWidget(DatasetView())
        stack.addWidget(TrainingView())

        optimizer_view = OptimizerView()
        results_view   = ResultsView()
        optimizer_view.optimization_complete.connect(results_view.set_optimized_result)

        stack.addWidget(optimizer_view)
        stack.addWidget(results_view)
        return stack

    def _build_status_bar(self):
        bar = QStatusBar()
        bar.setStyleSheet(f"background: {BG1}; border-top: 1px solid {BORDER};")
        self.setStatusBar(bar)

        if self._ngspice_available:
            lbl = QLabel("ngspice: available")
            lbl.setStyleSheet(f"color: {GREEN}; padding: 0 10px; font-size: 11px;")
        else:
            lbl = QLabel("ngspice: NOT FOUND — simulation disabled")
            lbl.setStyleSheet(f"color: {RED}; padding: 0 10px; font-size: 11px;")

        bar.addPermanentWidget(lbl)

    def _select(self, index: int):
        self._stack.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_buttons):
            btn.setSelected(i == index)

    def navigate_to(self, index: int):
        self._select(index)

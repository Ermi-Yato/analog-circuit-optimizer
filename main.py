import sys
import shutil
import argparse


def check_ngspice() -> bool:
    return shutil.which("ngspice") is not None


def run_gui(ngspice_ok: bool):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from app.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Xtal")
    app.setOrganizationName("thesis")

    if not ngspice_ok:
        dlg = QMessageBox()
        dlg.setWindowTitle("ngspice Not Found")
        dlg.setIcon(QMessageBox.Icon.Warning)
        dlg.setText("ngspice was not found on your PATH.")
        dlg.setInformativeText(
            "Dataset generation and SPICE verification will be unavailable.\n\n"
            "To install ngspice:\n"
            "  Windows:  winget install ngspice\n"
            "  macOS:    brew install ngspice\n"
            "  Linux:    sudo apt install ngspice\n\n"
            "The optimizer will still work using pre-trained models."
        )
        dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
        dlg.exec()

    window = MainWindow(ngspice_available=ngspice_ok)
    window.show()
    sys.exit(app.exec())


def run_headless(args):
    # Placeholder for future CLI mode
    print("[headless mode — not yet implemented]")
    print(f"  circuit : {args.circuit}")
    print(f"  targets : {args.target}")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Analog Circuit Optimizer")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (future CLI mode)",
    )
    parser.add_argument("--circuit", type=str, help="Circuit ID (headless mode)")
    parser.add_argument(
        "--target",
        nargs="*",
        metavar="metric=value",
        help="Target specs e.g. gain=20 bw=100000 (headless mode)",
    )
    args = parser.parse_args()

    ngspice_ok = check_ngspice()

    if not ngspice_ok:
        print(
            "WARNING: ngspice not found on PATH.\n"
            "  Dataset generation and SPICE verification will be unavailable.\n"
            "  Install ngspice: https://ngspice.sourceforge.io/download.html\n"
            "  On Windows: winget install ngspice\n"
            "  The optimizer can still run using pre-trained models.\n"
        )

    if args.headless:
        run_headless(args)
    else:
        run_gui(ngspice_ok)


if __name__ == "__main__":
    main()

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


def _safe(text: str) -> str:
    """Encode text safely for the current terminal (replaces unrepresentable chars)."""
    return text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")


def run_headless(args):
    if not args.circuit:
        print("ERROR: --circuit is required in headless mode.")
        print("  Example: python main.py --headless --circuit sallen_key_filter --target Cutoff_Freq_Hz=10000 Q_factor=0.707")
        sys.exit(1)

    # Parse targets
    targets: dict[str, float] = {}
    for item in (args.target or []):
        if "=" not in item:
            print(f"ERROR: invalid target '{item}' — expected metric=value format.")
            sys.exit(1)
        k, v = item.split("=", 1)
        try:
            targets[k.strip()] = float(v.strip())
        except ValueError:
            print(f"ERROR: non-numeric value for target '{k}': {v}")
            sys.exit(1)

    import registry.circuit_registry as reg
    from core.optimization.genetic_algorithm import optimize

    try:
        circuit = reg.get(args.circuit)
    except KeyError:
        all_ids = list(reg.load_all().keys())
        print(f"ERROR: unknown circuit '{args.circuit}'.")
        print(f"  Available circuits: {', '.join(all_ids)}")
        sys.exit(1)

    if not reg.model_exists(args.circuit):
        print(f"ERROR: no trained model found for '{args.circuit}'.")
        print("  Train a model first via the GUI (Training view) or import a .pkl.")
        sys.exit(1)

    metric_names = [m["name"] for m in circuit["metrics"]]
    if not targets:
        # Default targets from circuit JSON
        targets = {
            m["name"]: m.get("target_default", 0.0)
            for m in circuit["metrics"]
        }
        print("No targets specified — using defaults:")
        for k, v in targets.items():
            print(f"  {k} = {v}")

    # Validate supplied target names
    for k in targets:
        if k not in metric_names:
            print(f"ERROR: unknown metric '{k}' for circuit '{args.circuit}'.")
            print(f"  Valid metrics: {', '.join(metric_names)}")
            sys.exit(1)

    n_gen   = args.generations
    pop     = args.population
    model_block = circuit.get("model", {})
    model_type  = model_block.get("model_type", "random_forest")

    # Adaptive defaults
    if n_gen is None:
        n_gen = 200 if model_type == "mlp" else 100
    if pop is None:
        pop = 100 if model_type == "mlp" else 200

    print(f"\nOptimizing '{circuit['name']}' ({args.circuit})")
    print(f"  model type   : {model_type}")
    print(f"  generations  : {n_gen}")
    print(f"  population   : {pop}")
    print(f"  targets      :", " | ".join(f"{k}={v}" for k, v in targets.items()))
    print()

    def _progress(gen: int, best: float):
        bar_width = 30
        filled = int(bar_width * gen / n_gen)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"\r  gen {gen:>{len(str(n_gen))}}/{n_gen}  [{bar}]  score={best:.6f}", end="", flush=True)

    result = optimize(
        args.circuit,
        targets,
        n_generations=n_gen,
        pop_size=pop,
        progress_callback=_progress,
    )
    print()  # newline after progress bar

    print("\n--- Best Result ---")
    print(f"  Score (lower = better): {result['best_score']:.6f}\n")

    print("  Component Values:")
    param_defs = circuit["parameters"]
    for p in param_defs:
        name = p["name"]
        val  = result["best_params"][name]
        unit = p.get("unit", "")
        print(f"    {name:<12} = {val:.4g} {_safe(unit)}")

    print("\n  Predicted Performance:")
    for m in circuit["metrics"]:
        name = m["name"]
        pred = result["best_predicted"].get(name, float("nan"))
        tgt  = targets.get(name)
        unit = m.get("unit", "")
        if tgt is not None and abs(tgt) > 1e-12:
            err_pct = abs(pred - tgt) / abs(tgt) * 100
            print(f"    {name:<25} = {pred:.4g} {_safe(unit)}  (target {tgt:.4g}, err {err_pct:.1f}%)")
        else:
            print(f"    {name:<25} = {pred:.4g} {_safe(unit)}")

    print()
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
        help="Target specs e.g. Cutoff_Freq_Hz=10000 Q_factor=0.707",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        metavar="N",
        help="GA generations (headless mode, default: 200 MLP / 100 RF)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=None,
        metavar="N",
        help="GA population size (headless mode, default: 100 MLP / 200 RF)",
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

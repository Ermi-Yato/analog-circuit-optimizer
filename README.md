# Xtal — Analog Circuit Optimizer

A machine-learning-assisted analog circuit optimization framework.
Define target performance specs (gain, bandwidth, THD, etc.) and the system finds
the optimal component values using a surrogate ML model + Genetic Algorithm, with
optional SPICE verification.

Built as part of a master's thesis on ML-based analog design automation.

---

## How It Works

Three core blocks drive the system:

```
Component Values
      |
      v
[Block 1: Surrogate Model]  <-- trained on SPICE-generated data
      |  predicts performance metrics
      v
[Block 2: Genetic Algorithm]  <-- proposes + evaluates candidates
      |  finds optimal component values
      v
[Block 3: SPICE Verification]  <-- ground-truth validation
      |  compares predicted vs real
      v
Results
```

The GA never calls SPICE directly — it queries the fast surrogate model (~1ms vs ~1s per eval),
enabling population sizes of 100-200 in seconds instead of hours.

---

## Circuit Library

| Circuit | Sim Type | Parameters | Metrics | Model |
|---------|----------|-----------|---------|-------|
| Common-Emitter Amplifier | AC | R1, R2, Rc, Re | Peak_Gain_dB, Bandwidth_Hz | RF (R2>0.98) |
| Differential Amplifier | AC | R_L, R_E, R_tail | Diff_Gain_dB | RF (R2=0.998) |
| Sallen-Key Active Filter | AC | R1, R2, C1, C2 | Cutoff_Freq_Hz, Q_factor | MLP (R2>0.999) |
| Transimpedance Amplifier | AC | R_f, C_f | Transimpedance_dBOhm, Bandwidth_Hz, Phase_Margin_deg | RF |
| Class-A Amplifier | Transient | R_bias1, R_bias2, R_load, R_emitter | Output_Swing_V, THD_percent, Efficiency_percent | MLP (R2>0.95) |

Pre-trained models ship with the repo. The optimizer works out of the box without ngspice.

---

## Requirements

- Python 3.12+
- ngspice (for dataset generation and SPICE verification; optional for optimizer)

### Install Python dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Install ngspice (optional)

```bash
# Windows
winget install ngspice

# macOS
brew install ngspice

# Linux
sudo apt install ngspice
```

---

## Running the GUI

```bash
python main.py
```

The GUI opens with a sidebar for navigating between views:

| View | Purpose |
|------|---------|
| Dashboard | Select circuit, check model status |
| Circuit Manager | Add/edit circuits and parameters |
| Dataset | Generate SPICE simulation data |
| Training | Train the surrogate model (RF or MLP) |
| Optimizer | Set target specs, run GA, view candidates |
| Results | Predicted vs actual charts, SPICE validation |

---

## Headless CLI Mode

Run the optimizer without opening the GUI:

```bash
python main.py --headless --circuit sallen_key_filter \
    --target Cutoff_Freq_Hz=10000 Q_factor=0.707

python main.py --headless --circuit common_emitter_amplifier \
    --target Peak_Gain_dB=25 Bandwidth_Hz=50000000

python main.py --headless --circuit class_a_amplifier \
    --target Output_Swing_V=10 THD_percent=3 Efficiency_percent=40
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--generations N` | 200 (MLP) / 100 (RF) | GA generations |
| `--population N` | 100 (MLP) / 200 (RF) | Population size |

If `--target` is omitted, sensible defaults from the circuit JSON are used.

Example output:

```
Optimizing 'Sallen-Key Active Filter' (sallen_key_filter)
  model type   : mlp
  generations  : 200
  population   : 100
  targets      : Cutoff_Freq_Hz=10000.0 | Q_factor=0.707

  gen 200/200  [##############################]  score=0.000012

--- Best Result ---
  Score (lower = better): 0.000012

  Component Values:
    R1           = 15920 ohm
    R2           = 15920 ohm
    C1           = 1e-09 F
    C2           = 1e-09 F

  Predicted Performance:
    Cutoff_Freq_Hz            = 10001 Hz   (target 10000, err 0.0%)
    Q_factor                  = 0.7071     (target 0.707,  err 0.0%)
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests do not require ngspice or a GPU. The simulator is mocked where needed.

---

## Project Structure

```
analog-circuit-optimizer/
├── main.py                    # Entry point (GUI or --headless CLI)
├── requirements.txt
├── app/                       # GUI layer (PySide6)
│   ├── main_window.py
│   ├── views/                 # One view per workflow step
│   └── workers/               # QThread workers (non-blocking)
├── core/                      # Pure Python business logic (no GUI imports)
│   ├── dataset/               # Dataset generation + preprocessing
│   ├── models/                # RandomForest, MLP, trainer
│   ├── optimization/          # Genetic algorithm + objective function
│   ├── simulation/            # Ngspice wrapper
│   └── validation/            # SPICE verification
├── registry/
│   ├── circuit_registry.py    # Load/validate/register circuit JSONs
│   └── circuits/              # One JSON per topology
├── circuits/                  # SPICE netlists + .template files
├── data/                      # Generated CSVs (gitignored)
├── trained_models/            # Pre-trained .pkl files (ships with repo)
└── docs/
    └── TASKS.md               # Task tracker
```

---

## Adding a New Circuit

1. **Circuit Manager GUI** -> click "+ New Circuit"
2. Fill in name, description, SPICE template path, sim type
3. Add parameters (name, unit, min/max/default, scale: linear/log)
4. Add metrics (name, unit, optimize: maximize/minimize)
5. Click Save -> JSON written to `registry/circuits/<id>.json`
6. **Dataset view** -> Generate N samples (calls ngspice)
7. **Training view** -> Train model (select RF or MLP or Auto)
8. **Optimizer view** -> Set targets -> Run

---

## Model Types

| Type | Best For | Speed |
|------|---------|-------|
| Random Forest | Linear/piecewise metrics, feature importance | Fast |
| MLP Neural Network | Log-scale or complex nonlinear metrics | Moderate |
| Auto | Trains both, selects best mean R2 | Slowest |

The Sallen-Key filter uses MLP because cutoff frequency and Q factor are log-linear in component
values. MLP achieves R2 > 0.999 vs ~0.68 for RF on this circuit.

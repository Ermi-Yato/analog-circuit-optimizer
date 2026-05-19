# Analog Circuit Optimizer

A machine-learning-assisted framework for analog circuit parameter optimization. The system learns the relationship between component values and circuit performance from SPICE simulation data, then uses that learned model inside an optimization loop to find component values that satisfy user-defined performance targets — without running SPICE on every candidate.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Three Core Blocks](#3-three-core-blocks)
4. [Tech Stack](#4-tech-stack)
5. [Project Structure](#5-project-structure)
6. [Installation and Setup](#6-installation-and-setup)
7. [End-to-End Workflow](#7-end-to-end-workflow)
8. [Circuit Registry and Adding Custom Circuits](#8-circuit-registry-and-adding-custom-circuits)
9. [Dataset Generation](#9-dataset-generation)
10. [Surrogate Model Training](#10-surrogate-model-training)
11. [Optimization Engine](#11-optimization-engine)
12. [Fitness Function Configuration](#12-fitness-function-configuration)
13. [Results and Validation](#13-results-and-validation)
14. [Built-in Circuit Library](#14-built-in-circuit-library)
15. [Headless / CLI Usage](#15-headless--cli-usage)
16. [Testing](#16-testing)

---

## 1. Project Overview

Analog circuit design is an inherently iterative process. A designer specifies a target — for example, a voltage gain of 20 dB and a bandwidth of 100 kHz — and then manually adjusts component values (resistors, capacitors, transistor biasing) and re-runs SPICE simulations until the targets are met. This trial-and-error loop is slow, computationally expensive, and requires deep domain expertise.

This framework replaces the inner simulation loop with a **surrogate machine learning model** trained on a dataset of SPICE simulation results. Once trained, the surrogate can predict circuit performance for any parameter combination in microseconds rather than seconds. A global optimizer then uses this fast surrogate to search the parameter space efficiently, finding combinations that meet or exceed the specified targets. The best candidates are then verified with a real SPICE simulation to confirm correctness — closing the loop between ML and physics.

The framework is built around three non-negotiable pillars:

```
Input: Target specs (Gain, Bandwidth, CMRR, ...)
         ↓
Block 1: Surrogate ML Model      — predicts performance from parameters
         ↓ (evaluated inside Block 2's loop)
Block 2: Optimizer               — searches parameter space using Block 1
         ↓ (best candidate passed to Block 3)
Block 3: SPICE Verification      — confirms result with real simulation
         ↓
Output: Verified component values + predicted vs actual comparison table
```

---

## 2. System Architecture

```
analog-circuit-optimizer/
├── main.py                          # Entry point — launches GUI or headless mode
├── requirements.txt
│
├── app/                             # GUI layer — PySide6 ONLY
│   ├── main_window.py               # Sidebar navigation + stacked view container
│   ├── views/
│   │   ├── dashboard_view.py        # Home: circuit cards + pipeline status
│   │   ├── circuit_manager_view.py  # Add / edit circuits, parameters, metrics
│   │   ├── dataset_view.py          # Trigger dataset generation, preview results
│   │   ├── training_view.py         # Train surrogate model, view R² / MAE
│   │   ├── optimizer_view.py        # Set targets, choose algorithm, run optimizer
│   │   └── results_view.py          # Scatter plots, feature importance, SPICE validation
│   ├── widgets/
│   │   ├── plot_widget.py           # Embedded matplotlib canvas
│   │   └── circuit_drawings/        # Schemdraw diagrams for built-in circuits
│   └── workers/                     # QThread subclasses — keep UI non-blocking
│       ├── simulation_worker.py
│       ├── training_worker.py
│       ├── optimizer_worker.py
│       └── results_worker.py
│
├── core/                            # Pure Python — ZERO GUI imports
│   ├── simulation/
│   │   ├── base_simulator.py
│   │   └── ngspice.py               # Ngspice subprocess wrapper + parallel runner
│   ├── dataset/
│   │   ├── generator.py             # Parameter sweep + simulation + extraction
│   │   └── preprocessor.py          # StandardScaler, validation, CSV I/O
│   ├── models/
│   │   ├── base_model.py
│   │   ├── random_forest.py
│   │   ├── mlp.py
│   │   └── trainer.py               # Full training pipeline
│   ├── optimization/
│   │   ├── genetic_algorithm.py
│   │   ├── scipy_optimizer.py       # Differential Evolution + Dual Annealing
│   │   ├── bayesian_optimizer.py    # Optuna TPE + CMA-ES
│   │   └── objective.py             # Fitness function builder
│   └── validation/
│       └── spice_validator.py
│
├── registry/
│   ├── circuit_registry.py
│   └── circuits/                    # One JSON per circuit topology
│
├── circuits/                        # SPICE .template files
├── data/                            # Generated CSV datasets (gitignored)
├── trained_models/                  # Saved .pkl model files
└── outputs/                         # Exported PNG charts
```

The `core/` layer enforces a strict rule: **zero PySide6 imports**. It is fully importable in headless scripts and unit tests. All GUI code lives in `app/`, and all long-running tasks run inside `workers/` QThreads so the interface never freezes.

---

## 3. Three Core Blocks

### Block 1 — Surrogate ML Model

The surrogate is a supervised regression model trained to approximate the SPICE simulator. Given component values as inputs (e.g. R1=100 kΩ, R2=22 kΩ, Rc=3.3 kΩ, Re=470 Ω), it predicts performance metrics as outputs (e.g. Gain=22.1 dB, Bandwidth=87 kHz). Two architectures are supported: Random Forest and MLP (see [Section 10](#10-surrogate-model-training) for full details). Inference time is 0.1–1 ms per evaluation — orders of magnitude faster than SPICE.

### Block 2 — Optimizer

The optimizer treats the surrogate as a black-box function and searches the parameter space for component values that minimize a fitness score representing how far the predictions are from the user's targets. Five algorithms are available. The critical enabler: evaluating the surrogate takes ~0.1 ms vs ~0.5–5 seconds for real SPICE, making it feasible to evaluate hundreds of thousands of candidates.

### Block 3 — SPICE Verification

After optimization, a single real Ngspice simulation is run on the best candidate. Ground-truth metrics are compared against the surrogate's predictions in a table with absolute and relative errors. This closes the loop between ML and physics and provides the final, trustworthy performance figures.

---

## 4. Tech Stack

| Component | Library | Role |
|---|---|---|
| Language | Python 3.12 | — |
| GUI | PySide6 (Qt6) | Full desktop application |
| ML — Random Forest | scikit-learn `RandomForestRegressor` | Ensemble surrogate model |
| ML — Neural Network | scikit-learn `MLPRegressor` | MLP surrogate model |
| Feature Scaling | scikit-learn `StandardScaler` | Input normalization |
| GA Optimizer | DEAP | Genetic Algorithm |
| Scipy Optimizers | scipy | Differential Evolution, Dual Annealing |
| Bayesian Optimizer | optuna | TPE and CMA-ES samplers |
| SPICE Simulation | Ngspice (external) | Circuit simulation |
| Circuit Diagrams | schemdraw | Schematic rendering |
| Data | pandas, numpy | Dataset handling |
| Serialization | joblib | Model persistence (.pkl) |
| Plotting | matplotlib | Embedded charts |
| Testing | pytest | Unit tests |

---

## 5. Project Structure

See [Section 2](#2-system-architecture) for the annotated directory tree.

Key design rules enforced throughout the codebase:

- `core/` has zero PySide6 imports — importable headlessly for testing and scripting
- Long-running tasks always run in `workers/` QThreads — the main thread is never blocked
- The circuit registry JSON is the single source of truth — no hardcoded circuit definitions in Python
- Users never edit JSON directly — the Circuit Manager GUI owns all circuit config writes
- Pre-trained models ship with the repository for all five built-in circuits

---

## 6. Installation and Setup

### Prerequisites

- Python 3.12
- Ngspice installed and on `PATH`

### Install Ngspice

**Windows**: Download the installer from [ngspice.sourceforge.io](http://ngspice.sourceforge.io/download.html). After installation, add the `bin\` subdirectory to your `PATH` environment variable.

**macOS**: `brew install ngspice`

**Linux**: `sudo apt install ngspice`

Verify: `ngspice --version`

Ngspice is required for dataset generation and SPICE validation. The optimizer itself only needs the trained model and works without Ngspice.

### Install Python dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# Install
pip install -r requirements.txt
```

### Launch

```bash
python main.py
```

A warning banner appears on the dashboard if Ngspice is not found on PATH. Pre-trained models for all five circuits ship with the repository, so the optimizer works immediately without any setup.

---

## 7. End-to-End Workflow

```
Dashboard
    │  Select or add a circuit
    ▼
Circuit Manager
    │  Define parameters, metrics, SPICE template
    ▼
Dataset View
    │  Generate N simulation samples
    ▼
Training View
    │  Train surrogate model (RF, MLP, or Auto)
    ▼
Optimizer View
    │  Set target specs, choose algorithm and fitness settings, run
    ▼
Results View
    │  Inspect charts, run SPICE verification, export
    ▼
Verified component values + performance comparison table
```

For the five built-in circuits, pre-trained models are included. You can skip directly to **Optimizer View** without generating data or training.

### Dashboard

Each circuit appears as a card showing:
- Name and simulation type badge (AC / Transient / DC)
- **Pipeline status bar**: colored dots for Data, Train, and Optimize — green = complete, blue = current step, gray = locked
- R² scores from the last training run
- A smart CTA button routing to the next needed step: "Generate Data" → "Train Model" → "Optimize"

Clicking anywhere on a card navigates to the appropriate view and auto-selects that circuit in the dropdown.

---

## 8. Circuit Registry and Adding Custom Circuits

### The Registry

All circuit metadata lives in `registry/circuits/<id>.json`. The Python registry module validates, loads, and caches these files. The GUI's Circuit Manager is the only writer.

### Circuit JSON Schema

```json
{
  "id": "my_amplifier",
  "name": "My Amplifier",
  "description": "Common-emitter with emitter degeneration",
  "spice_template": "circuits/my_amplifier/template.cir",
  "simulation_type": "ac",
  "parameters": [
    {
      "name": "R1",
      "label": "Base Bias R1",
      "unit": "Ω",
      "min": 50000,
      "max": 150000,
      "default": 100000,
      "scale": "linear"
    }
  ],
  "metrics": [
    {
      "name": "Peak_Gain_dB",
      "label": "Voltage Gain",
      "unit": "dB",
      "optimize": "maximize"
    }
  ],
  "model": {
    "surrogate_path": "trained_models/my_amplifier/circuit_model.pkl",
    "scaler_path":    "trained_models/my_amplifier/feature_scaler.pkl",
    "trained_on": "2025-05-19",
    "samples": 1000,
    "r2_scores": {"Peak_Gain_dB": 0.987},
    "model_type": "random_forest",
    "log_metrics": []
  }
}
```

The `"model"` block is optional — if absent the dashboard shows "Not trained yet".

### Parameter Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Internal identifier, used to generate the SPICE placeholder |
| `label` | string | Human-readable name shown in UI |
| `unit` | string | Display unit (Ω, Hz, F, H, V, etc.) |
| `min` | float | Minimum value sampled during dataset generation |
| `max` | float | Maximum value sampled during dataset generation |
| `default` | float | Used for SPICE validation when no optimized params exist |
| `scale` | `"linear"` or `"log"` | Sampling distribution. Use `"log"` for capacitors, inductors, and any component spanning multiple decades |

### Metric Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Identifier used in CSV header and extractor lookup |
| `label` | string | Human-readable name |
| `unit` | string | Display unit |
| `optimize` | `"maximize"` or `"minimize"` | Direction for fitness scoring |

### Adding a New Circuit — Step by Step

#### Step 1: Write the SPICE template

Create `circuits/<your_id>/template.cir`. Replace every component value with a placeholder.

**Placeholder naming rule**: parameter name → strip underscores → uppercase → append `_VAL` in `{}`.
Examples: `R1` → `{R1_VAL}`, `R_bias` → `{RBIAS_VAL}`, `C_filter` → `{CFILTER_VAL}`

```spice
* My Amplifier Template
.model BC547B NPN (...)

Vcc  1 0 DC 12
R1   1 2 {R1_VAL}
R2   2 0 {R2_VAL}
Rc   1 3 {RC_VAL}
Re   4 0 {RE_VAL}
Q1   3 2 4 BC547B
Cin  5 2 10uF
Cout 3 6 10uF
Vin  5 0 AC 1

.control
  ac dec 100 100 10Meg
  wrdata ngspice_simulation_output.txt frequency v(6)
  exit
.endc
.end
```

**Required**: the `.control` block must write to `ngspice_simulation_output.txt` using `wrdata`:
- AC analysis: `wrdata ngspice_simulation_output.txt frequency v(<output_node>)`
- Transient analysis: `wrdata ngspice_simulation_output.txt time v(<output_node>)`

#### Step 2: Open Circuit Manager → New Circuit

Click **+ New Circuit** in the left panel. Fill in the Details tab:
- **Circuit ID**: lowercase, no spaces (becomes the JSON filename)
- **Name**: display name
- **Description**: brief topology description
- **SPICE Template**: path to your `.cir` file (use Browse)
- **Simulation Type**: `ac`, `transient`, or `dc`

The **Template Reference** panel auto-generates the exact `{NAME_VAL}` placeholders from your defined parameters, updating live as you add parameters.

#### Step 3: Define Parameters

Go to the **Parameters** tab. The **Suggestions from template** section scans your template file for `{NAME_VAL}` patterns and shows clickable chips for each one — click a chip to pre-fill a row.

Set ranges carefully: too wide → slow generation with unphysical operating points; too narrow → optimal solution may be outside the range.

The **Placeholders** bar updates live showing the exact placeholder string for each defined parameter.

#### Step 4: Define Metrics

Go to the **Metrics** tab. Use the preset chips for common metrics or add custom rows. Metric names must match the built-in extractor patterns:

| Pattern in metric name | Extracted quantity |
|---|---|
| `gain` or `db` | Peak gain from AC magnitude response |
| `bandwidth`, `bw`, `hz`, or `freq` | -3 dB bandwidth |
| `phase` | Phase at peak-gain frequency |
| `cmrr` | Common-mode rejection ratio |
| `q_factor` or `q` | Quality factor from resonance peak |
| `transimpedance` | `|V(out)| / |I(in)|` in dBΩ |
| `swing` | Peak-to-peak output voltage (transient) |
| `thd` | Total harmonic distortion (transient) |
| `efficiency` | Power efficiency percentage (transient) |

If a metric name does not match any pattern, dataset generation raises a descriptive error listing the unresolved names and suggesting naming alternatives.

#### Step 5: Add Schematic (optional)

Go to the **Schematic** tab and click **Import Image** to attach a PNG, JPG, or SVG diagram of the circuit.

#### Step 6: Save

Click **Save Circuit**. The registry validates all fields and writes the JSON. The circuit immediately appears on the dashboard.

#### Step 7: Validate template (optional but recommended)

In the **Details** tab, click **Validate Template** to check that every defined parameter's placeholder appears in the template file. Missing placeholders are listed in red.

---

## 9. Dataset Generation

### What it does

The generator runs a parameter sweep: it samples N combinations of component values from the defined ranges, runs each through Ngspice, extracts the specified metrics from simulation output, and assembles a CSV at `data/<circuit_id>_dataset.csv`.

### Sampling strategy

- **Linear-scale parameters**: uniform sampling — `value = uniform(min, max)`
- **Log-scale parameters**: uniform sampling in log-space — `value = 10 ^ uniform(log10(min), log10(max))`

Log-space sampling is critical for capacitors and inductors. For a 1 nF–100 nF capacitor, linear sampling puts 90% of samples in the 10–100 nF decade and barely covers 1–10 nF. Log-space sampling places equal emphasis on every decade.

### Parameter injection

Placeholders in the template are substituted with formatted float values before each simulation run. Each run executes in its own temporary directory to prevent filesystem race conditions during parallel execution.

### Parallel simulation

Simulations run concurrently via `ThreadPoolExecutor`. The worker count defaults to CPU cores − 1 and is configurable in the UI. Each Ngspice process is completely independent; they do not share state.

### Metric extraction

After each simulation, the output file is parsed and metrics are extracted using pattern-matched extractor functions:

- For **AC analysis**: the output contains frequency and complex voltage columns. Gain is computed as `20 * log10(|V|)`, bandwidth as the -3 dB frequency range, phase at the peak frequency.
- For **transient analysis**: the output contains time and voltage columns. Swing is the peak-to-peak amplitude, THD is computed via FFT of the steady-state waveform.

### Failure handling

Rows where:
- Ngspice fails to converge (common near saturation or cutoff boundaries)
- Any extracted metric is NaN or infinite
- Metric extraction fails for any registered reason

...are dropped automatically with a failure type breakdown printed in the console. If > 20% of rows are dropped, the dataset view warns the user. Common causes:
- Parameter ranges too extreme (transistor driven into saturation or cutoff)
- Template missing a `wrdata` statement
- Metric names not matching any extractor pattern (pre-validated and reported before simulations run)

### Dataset View controls

| Control | Description |
|---|---|
| Circuit | Dropdown showing all registered circuits |
| Number of samples | How many parameter combinations to simulate. 500 = quick test; 1000–2000 = good accuracy; 5000+ = production quality |
| Max workers | Parallel simulation threads. Default: CPU count − 1 |
| Generate | Starts generation; progress bar updates per completed simulation |
| Preview | Shows first rows of the generated CSV after completion |

---

## 10. Surrogate Model Training

### The training pipeline (step by step)

**1. Load** the CSV dataset.

**2. Validate** — the preprocessor checks for: excessive NaN values, near-zero variance metrics (circuit is insensitive to a parameter), columns with unexpected data types. Raises `ValueError` with a descriptive message on failure.

**3. Log-transform log-scale parameters** — for parameters marked `"scale": "log"`, the CSV values are transformed as `log10(value)` before training. This linearizes relationships like `fc = 1 / (2π√(R₁R₂C₁C₂))` which is linear in log-space, dramatically improving model accuracy for filter circuits.

**4. Log-transform log-scale metrics** — for metrics where the raw values span many decades (e.g. `Cutoff_Freq_Hz` ranging from 100 Hz to 500 kHz), training in linear space produces poor accuracy at the low end: an MAE of ±1000 Hz is fine at 100 kHz but represents 1000% error at 1 kHz. Training in log-space ensures uniform relative accuracy across the full range.

**5. StandardScaler normalization** — scales all features to zero mean and unit variance. This ensures that a parameter in the range 1 kΩ–100 kΩ and a parameter in the range 10 nF–100 nF receive equal treatment by the model. The fitted scaler is saved alongside the model and applied at inference time.

**6. 80/20 train/test split** — 80% of data for fitting, 20% held out for unbiased evaluation. The test split is never used during training.

**7. Train** the chosen model type.

**8. Evaluate** on the held-out test set. Computes R² and MAE for each output metric.

**9. Save** model and scaler to `trained_models/<circuit_id>/circuit_model.pkl` and `feature_scaler.pkl`.

**10. Update** the circuit JSON `"model"` block with training date, sample count, R² scores, model type, and log-metric names.

### Model Type: Random Forest

Trains an ensemble of 200 decision trees (`sklearn.ensemble.RandomForestRegressor`). Each tree is trained on a bootstrap sample of the training data, and at each split only a random subset of features is considered. The final prediction is the mean of all 200 trees.

**Strengths:**
- Handles non-linear, discontinuous, and interaction-heavy parameter-performance relationships without any configuration
- Robust to outliers in simulation data (bad SPICE runs that slipped through)
- Provides feature importances showing which parameters most affect each metric
- Generalizes well even from relatively small datasets (500–1000 samples)
- No hyperparameter tuning required for typical analog circuits

**Weaknesses:**
- Predictions are piecewise constant — the model cannot smoothly extrapolate outside the training distribution; predictions near the boundary of parameter ranges may be unreliable
- High memory usage for large ensembles on high-dimensional inputs
- Feature importance does not capture parameter interaction effects directly

**Best for:** Transistor amplifier circuits with non-smooth gain-bandwidth characteristics, circuits with operating-point discontinuities (saturation/cutoff), any circuit where you prioritize robustness over smoothness.

### Model Type: MLP (Multi-Layer Perceptron)

Trains a fully connected neural network with architecture `input → 256 → 128 → 64 → output`, ReLU activations, Adam optimizer, up to 500 epochs, scikit-learn's `MLPRegressor`.

**Strengths:**
- Produces smooth, continuous predictions — can interpolate meaningfully between training points
- Captures complex non-linear interactions through learned representations
- Supports efficient batch inference (entire GA population evaluated in a single matrix multiply)
- Often superior for filter circuits where the underlying physics follows smooth analytical relationships (e.g. Sallen-Key)

**Weaknesses:**
- Requires more training data to generalize — typically 2000+ samples for reliable results
- Training is slower and less deterministic than Random Forest
- No built-in feature importance
- More sensitive to hyperparameter choices; the fixed architecture may not be optimal for all circuits

**Best for:** Filter circuits (Sallen-Key, active bandpass, notch), circuits with smooth transfer functions, situations where very large datasets (5000+) have been generated.

### Model Type: Auto

Trains both Random Forest and MLP, evaluates both on the held-out test set, and keeps whichever achieves higher **mean R² across all metrics**. The chosen model type is stored in the circuit JSON so subsequent optimizer and results runs load the correct class automatically.

**Recommendation**: Use Auto when you are unsure which model type is better for a new circuit. The overhead of training both is small (seconds to minutes) compared to the dataset generation cost.

### Interpreting Training Metrics

| Metric | Meaning |
|---|---|
| **R²** | Fraction of variance explained by the model. R²=1.0 is perfect; R²=0 is no better than predicting the mean; R²<0 is worse than the mean. Computed per output metric. |
| **MAE** | Mean Absolute Error on the test set, in the metric's native units. For a gain metric, MAE=0.4 dB means predictions are off by ±0.4 dB on average. |

**R² quality guide:**

| R² | Interpretation | Action |
|---|---|---|
| ≥ 0.99 | Excellent | Suitable for precision optimization |
| 0.95–0.99 | Good | Optimizer will find reliable solutions |
| 0.90–0.95 | Acceptable | Small verification errors expected; results are still usable |
| 0.80–0.90 | Marginal | Generate more data or try the other model type |
| < 0.80 | Poor | Do not use. Generate more data, verify template, check metric extraction |

**Troubleshooting low R²:**
1. Generate more samples (double the count)
2. Check that parameter ranges are physically meaningful — ranges spanning unphysical operating points (transistor fully cut off or saturated for most samples) produce noisy data
3. Check the dataset for excessive failure rows — if > 30% of rows were dropped during generation, the data quality is too low
4. Switch model type — try Auto to compare RF vs MLP
5. Check log-scale settings — if a metric spans multiple decades, add `"scale": "log"` to its metric definition in Circuit Manager to enable log-space training

---

## 11. Optimization Engine

The optimizer takes a trained surrogate model and user-defined target specifications, then searches the parameter space for the component values that best satisfy those targets. All algorithms minimize a scalar fitness score (lower = better) built from the fitness function described in [Section 12](#12-fitness-function-configuration).

**Shared result format:**

| Key | Type | Description |
|---|---|---|
| `best_params` | dict | Best component value set found: `{param_name: value}` |
| `best_score` | float | Fitness score for the best solution (lower = better match to targets) |
| `best_predicted` | dict | Surrogate predictions at best params: `{metric_name: value}` |
| `history` | list | `[(iteration, best_score), ...]` — convergence trace for the chart |
| `population` | list | Top 50 candidates sorted by score: `[(params_dict, score, predicted_dict), ...]` |

---

### Algorithm 1: Genetic Algorithm (GA)

**Concept:** Mimics biological evolution. Maintains a population of candidate solutions. Each generation applies:
1. **Tournament selection** — random groups compete; fitter individuals are more likely to breed
2. **Blend crossover** — two parent vectors combine as `α·parent₁ + (1−α)·parent₂`
3. **Gaussian mutation** with a linearly decaying sigma (simulated annealing schedule) — aggressive early exploration, fine-grained late exploitation
4. **Elitism** — top N individuals copied unchanged, preventing loss of best solutions

**Parameters:**

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Generations** | 100 | 10–2000 | Total evolution cycles. Each evaluates the full population. More = better exploration at cost of runtime. 50–200 covers most circuits. |
| **Population** | 200 | 20–2000 | Candidates per generation. Minimum recommended: 10 × number of parameters. Larger = broader exploration but proportionally slower. |
| **Mutation rate** | 0.2 | 0–1 | Probability a child mutates. Too low (< 0.05) → premature convergence. Too high (> 0.5) → random walk. Best range: 0.1–0.3. |
| **Crossover rate** | 0.7 | 0–1 | Probability two parents produce a blended child. Standard range: 0.6–0.8. |
| **Elite size** | 4 | 1–50 | Top N preserved each generation unchanged. 2–6 is typical. Large values reduce diversity. |
| **Tournament size** | 3 | 2–20 | Competitors per breeding slot. Larger = stronger selection pressure = faster but riskier convergence. |

**Best for:** General-purpose baseline. Works well with Random Forest surrogates (piecewise predictions tolerate non-smooth fitness landscapes). Predictable runtime.

---

### Algorithm 2: Differential Evolution (DE)

**Concept:** Population-based with vector arithmetic mutation. For each candidate `x`, the trial vector is `v = a + F·(b−c)` where `a`, `b`, `c` are randomly chosen population members and `F` is the mutation factor. A child is formed by elementwise crossover between `x` and `v`. If the child is fitter, it replaces `x`. After convergence, a local polishing step refines the best solution with L-BFGS-B.

**Parameters:**

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Max iterations** | 300 | 10–5000 | Total DE generations. Converges in fewer iterations than GA — 100–500 is typical. |
| **Pop size / dim** | 15 | 5–50 | Population = this value × number of parameters. For 4 parameters: 60 candidates. Standard range: 5–20. |
| **Mutation F** | 0.7 | 0–2 | Scale of the difference vector. Low (0.4) = fine search. High (0.9) = aggressive jumps. 0.5–0.9 for most circuits. Values > 1.2 rarely help. |
| **Recombination** | 0.7 | 0–1 | Per-dimension crossover probability. High (0.9) = trial vector dominates. Low (0.3) = conservative updates. |

**Best for:** Smooth continuous objective landscapes. Often the best all-around algorithm — few hyperparameters, fast convergence, and the polishing step adds final precision that evolutionary algorithms miss.

---

### Algorithm 3: Dual Annealing (DA)

**Concept:** Combines Generalized Simulated Annealing (stochastic trajectory with probabilistic acceptance of worse solutions guided by a temperature schedule) with a fast local minimizer. Unlike GA and DE, maintains only a single solution — lower memory, simpler, but higher risk of local trapping. Temperature starts high (wide exploration) and decreases according to the visiting parameter `ν`, eventually focusing on a narrow region.

**Parameters:**

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Max iterations** | 1000 | 100–10000 | Total function evaluations. Needs more than DE to compensate for single-trajectory search. 500–3000 recommended. |
| **Initial temp** | 5230 | 0.1–50000 | Starting temperature. The default (5230) is derived from the mathematical optimum for the GSA visiting distribution. Higher = more exploratory. Rarely needs changing unless the objective surface is known to be very flat or very sharp. |

**Best for:** Highly multimodal landscapes where GA and DE get trapped in local optima. The probabilistic uphill acceptance allows it to escape such traps.

---

### Algorithm 4: Bayesian Optimization — TPE (Tree-structured Parzen Estimator)

**Concept:** Builds a probabilistic model of the objective surface from previous observations and uses it to select the next trial point — balancing exploration (sampling uncertain regions) and exploitation (sampling promising regions). TPE models the distribution of good configurations `l(x)` and bad configurations `g(x)` separately using kernel density estimation, then maximizes `l(x)/g(x)` (the Expected Improvement) to select the next point.

After `n_startup_trials` random explorations, every subsequent trial is informed by all previous results.

**Parameters:**

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Trials** | 300 | 50–5000 | Total surrogate model evaluations. TPE is significantly more sample-efficient than evolutionary algorithms — 100–300 trials often matches thousands of GA evaluations. |
| **Startup (random)** | 20 | 5–200 | Random trials before the TPE model activates. Too few (< 10) = poor initial model. 15–30 for circuits with 2–5 parameters; 40–60 for 6+ parameters. |

**Best for:** When sample efficiency matters — when you want the best result from the fewest evaluations. Excellent for circuits where you want to explore multiple parameter configurations interactively, trying different target specs without long waits between runs.

---

### Algorithm 5: Bayesian Optimization — CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

**Concept:** Maintains a multivariate Gaussian distribution over the parameter space. After each batch of evaluations, updates both the mean (toward successful candidates) and the full covariance matrix (to learn correlations between parameters). If gain depends on the ratio R1/R2 and both must increase together, CMA-ES discovers this correlation and generates future samples aligned with that direction. Also adapts the step size to maintain a target acceptance rate.

**Parameters:**

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Trials** | 300 | 50–5000 | Total evaluations. CMA-ES needs more trials than TPE to learn the covariance structure. 200–500 typical. |
| **Startup (random)** | 20 | 5–200 | Random trials before CMA-ES initializes. More critical than for TPE — the covariance estimate requires adequate initial data. 30–50 recommended. |

**Best for:** Circuits where parameters are **correlated** — matched pairs in differential amplifiers, resistor dividers, filter component ratios, bias networks. Outperforms TPE on higher-dimensional problems (6+ parameters) with structural correlations.

---

### Algorithm Comparison

| Algorithm | Style | Best For | Sample Efficiency | Runtime |
|---|---|---|---|---|
| **GA** | Population, evolutionary | General baseline, RF surrogates | Low | Gen × Pop |
| **Differential Evolution** | Population, vector arithmetic | Smooth continuous landscapes | Medium | Iter × Pop/dim |
| **Dual Annealing** | Single trajectory, stochastic | Multimodal, many local optima | Medium | Max iterations |
| **Bayesian TPE** | Probabilistic model | Fewest evaluations needed | High | n_trials |
| **Bayesian CMA-ES** | Covariance-guided | Correlated parameters | High | n_trials |

**General recommendation:**
- Start with **Differential Evolution** for a reliable baseline
- Use **Bayesian TPE** when you want fast interactive exploration of different target specs
- Use **Bayesian CMA-ES** for circuits with correlated parameters (differential amplifiers, matched-component filters)
- Use **GA** when the surrogate is a Random Forest and the landscape is non-smooth
- Use **Dual Annealing** when DE and GA both get stuck in the same local optimum

---

## 12. Fitness Function Configuration

The fitness function translates the multi-objective targeting problem into a single scalar that all algorithms minimize. The score for a candidate set of component values is:

```
total_score = Σ over all metrics:
    weight_m × direction_factor_m × loss( (predicted_m - target_m) / scale_m )
```

where `scale_m = |target_m|` (or 1.0 if the target is near zero). Three controls configure this function.

---

### Loss Type

Controls the mathematical shape of the penalty for normalized error `e = (predicted − target) / target`.

#### MAE — Mean Absolute Error (default)
```
loss(e) = |e|
```
Linear in the error. A 10% miss costs exactly 10× less than a 100% miss. Predictable, balanced, and easy to interpret. The optimizer treats all deviations proportionally regardless of magnitude. **Use this for most circuits.**

#### MSE — Mean Squared Error
```
loss(e) = e²
```
Quadratic in the error. A miss twice as large costs four times as much. The optimizer becomes disproportionately motivated to eliminate large deviations at the cost of tolerating many small ones. **Use when large misses are disproportionately harmful** — for example, a maximum output swing spec where exceeding the limit by 2× is catastrophic but missing by 5% is acceptable.

#### Huber — L1+L2 Hybrid
```
loss(e) = e²/2           if |e| ≤ 1 (quadratic zone)
loss(e) = |e| − 0.5      if |e| > 1 (linear zone)
```
Quadratic near the target (like MSE — sensitive to small improvements) but linear for large errors (like MAE — resistant to outlier predictions dominating the score). Particularly useful when the surrogate model occasionally produces outlier predictions for candidates near the edge of the training distribution — these outliers do not catastrophically inflate the fitness score. **Use when the surrogate is a Random Forest** operating on parameter combinations far from the training centroid.

#### Log — Logarithmic Compression
```
loss(e) = log(1 + |e|)
```
Grows very slowly for large errors — a 10× miss costs only `log(11) ≈ 2.4` instead of 10. This prevents a single metric with a large absolute deviation from monopolizing the optimizer's attention. **Use when one metric has a much wider natural range than others**, or when you want the optimizer to make roughly equal progress on all metrics simultaneously rather than fixing the worst-miss metric first.

---

### Direction Penalty

**Default: 1.0 (symmetric)**

In engineering, missing a spec in the wrong direction is categorically different from exceeding it. A gain spec of 20 dB met with a prediction of 22 dB (overshoot) is a success; met with 18 dB (undershoot) is a failure that requires redesign. Symmetric loss treats both equally.

The direction penalty multiplier applies specifically to **wrong-direction misses**:
- For `maximize` metrics: when `predicted < target` (undershooting the spec)
- For `minimize` metrics: when `predicted > target` (overshooting the spec)

When a wrong-direction miss occurs: `loss_effective = direction_penalty × loss(e)`

**Practical guide:**

| Value | Behavior | Use Case |
|---|---|---|
| **1.0** | Symmetric — minimize distance to target in all directions equally | When you want precise on-target matching, not just spec satisfaction |
| **1.5–2.0** | Mild spec preference — slightly prefers meeting or exceeding over symmetric matching | Good default for most analog design targets |
| **2.5–4.0** | Strong spec satisfaction — optimizer strongly prefers meeting or exceeding targets over matching precisely | Standard engineering specs: "must achieve at least X" |
| **5.0–8.0** | Near-hard constraint — optimizer almost never allows wrong-direction misses | Critical specs: noise figure, supply voltage limit, maximum current |
| **10.0+** | Hard constraint approximation — wrong-direction misses are treated as near-unacceptable | Use with caution: very high values can cause the optimizer to fixate on direction at the expense of precision |

**Example with direction penalty = 3.0, target gain = 20 dB:**
- Predicted 21.5 dB: error = +1.5 dB (overshoot), score contribution = `w × loss(1.5/20)`
- Predicted 18.5 dB: error = −1.5 dB (undershoot), score contribution = `3 × w × loss(1.5/20)`

The optimizer strongly prefers 21.5 dB even though it is numerically further from the target, because undershooting the gain spec would be a design failure.

---

### Tolerance Band (%)

**Default: 0% (disabled)**

Creates a dead-zone around each target: predictions within `± tolerance_pct%` of the target contribute zero penalty regardless of the exact value.

```
penalty = 0                   if |predicted − target| ≤ (tolerance_pct/100) × |target|
penalty = normal loss         otherwise
```

**Practical guide:**

| Value | Behavior | Use Case |
|---|---|---|
| **0%** | All deviations penalized | When precise target matching is required |
| **1–3%** | Tight tolerance | High-precision circuits where component precision matters |
| **3–7%** | Practical engineering tolerance | Standard design: component manufacturing tolerances (E96: ±1%, E24: ±5%) make sub-1% optimization meaningless |
| **10–15%** | Broad feasibility search | First-pass exploration of a new circuit topology — find any valid operating point |
| **20%+** | Very loose — approximate region only | When you need a rough starting point for manual refinement |

The tolerance band is most powerful when combined with direction penalty: set a moderate tolerance (5%) with a moderate direction penalty (2.0). The optimizer will find solutions that meet all specs approximately, rather than precisely optimizing one metric while letting others drift.

---

### Combining Options

These three settings compose independently. Some recommended configurations:

**Standard design exploration** (default):
- Loss: MAE · Direction penalty: 1.5 · Tolerance: 0%

**Spec-satisfying mode** (the component values must meet or beat all targets):
- Loss: MAE · Direction penalty: 3.0 · Tolerance: 3%

**Precision tuning** (a previously satisfying solution needs micro-optimization):
- Loss: MSE · Direction penalty: 1.0 · Tolerance: 0%

**Robust search** (surrogate has high uncertainty, want a feasible region not a precise point):
- Loss: Huber · Direction penalty: 2.0 · Tolerance: 8%

**Critical spec** (one metric must never miss in the bad direction):
- Loss: MAE · Direction penalty: 8.0 · Tolerance: 0%

---

## 13. Results and Validation

The Results view presents four analysis panels after loading a trained model.

### Tab 1: Predicted vs Actual (Scatter Plot)

One subplot per output metric. The x-axis shows actual simulation values from the held-out test set; the y-axis shows the surrogate model's predictions for those same inputs. The dashed diagonal represents perfect prediction.

**Interpreting the plot:**

| Pattern | Meaning | Action |
|---|---|---|
| Points tightly along the diagonal across the full range | Excellent model, R² ≥ 0.98 | None needed |
| Systematic S-curve or parabolic deviation | Model misses a non-linearity | Try the other model type or generate more data in the problematic region |
| Scatter increasing at the extremes | Model less confident at edge of training distribution | Normal for RF; avoid setting optimization targets at the extreme parameter range edges |
| Vertical or horizontal banding | RF piecewise-constant artifacts | Qualitatively correct but lower precision; try MLP if this causes optimizer issues |
| Points clustered in a blob with no relationship to diagonal | R² is near zero — the model learned nothing | Check parameter ranges, metric extraction, and dataset quality |

### Tab 2: Feature Importance (Random Forest only)

Horizontal bar chart of each input parameter's relative contribution to the model's predictions for each metric. Values sum to 1.0 per metric.

**Interpreting importances:**

| Pattern | Meaning | Action |
|---|---|---|
| One parameter at > 0.7 importance | The metric is primarily controlled by a single component | Physically meaningful (e.g. gain dominated by Rc in a CE amplifier). Narrow other params' ranges for faster optimization. |
| Uniform distribution across all parameters | All parameters contribute equally | Broader parameter sweep may be needed; consider adding more samples |
| A parameter with near-zero importance | This component barely affects the metric within the defined range | Check whether the range is physically meaningful; the parameter may be operating in a region where it has no effect |

Feature importance is not available for MLP models — the tab displays "Not available for MLP" in that case.

### Tab 3: Error Residuals (Histogram)

Distribution of prediction errors `(actual − predicted)` for each metric. A well-calibrated model produces a distribution centered at zero with a narrow Gaussian-like spread.

**Interpreting the histogram:**

| Pattern | Meaning | Action |
|---|---|---|
| Narrow symmetric bell curve centered at 0 | Excellent calibration, no systematic bias | None needed |
| Distribution shifted left (mean < 0) | Model systematically over-predicts this metric | Check for data skew; the training set may over-represent high-performance operating points |
| Distribution shifted right (mean > 0) | Model systematically under-predicts | Same as above, opposite direction |
| Very wide spread (high variance) | Model has high uncertainty | Generate more data; this metric may need more samples to learn |
| Bimodal distribution | Two distinct circuit operating regions (e.g. linear vs. saturation) | Consider splitting the parameter space into two ranges and training separate models |

### Tab 4: SPICE Validation

The ground truth check — Block 3 of the framework. After an optimization run completes, the best candidate parameters are automatically passed to this tab. Click **Run SPICE Validation** to execute a real Ngspice simulation on those exact values.

**Validation table:**

| Column | Description |
|---|---|
| **Metric** | Performance metric name |
| **Surrogate Prediction** | ML model's predicted value at the optimized component values |
| **Ngspice Actual** | Real SPICE simulation result — the ground truth |
| **Abs Error** | `|Surrogate − Actual|` in the metric's natural units |
| **Rel Error (%)** | `|Surrogate − Actual| / |Actual| × 100` |

**Relative error color coding:**
- Green (< 5%): Excellent agreement — the surrogate accurately represents the circuit in this region
- Yellow (5–15%): Acceptable for design exploration — the actual SPICE values are the authoritative figures
- Red (> 15%): Poor agreement — the surrogate is inaccurate near the optimized point. Generate more data in this parameter region and retrain.

**Using validation without the optimizer:** The validation tab can also be used with default parameter values (no optimizer run needed) as a quick sanity check on model calibration. Load a circuit, go to the SPICE Validation tab, and click Run.

### Export

**Export PNGs** saves all three chart tabs as high-resolution PNG files to `outputs/<circuit_id>_scatter.png`, `outputs/<circuit_id>_importance.png`, and `outputs/<circuit_id>_residuals.png`. Suitable for direct inclusion in papers and reports.

---

## 14. Built-in Circuit Library

Five circuits ship with pre-trained Random Forest and/or MLP models. The optimizer can be used immediately without any setup.

| Circuit | ID | Sim Type | Parameters | Metrics |
|---|---|---|---|---|
| Common-Emitter Amplifier | `common_emitter_amplifier` | AC | R1, R2, Rc, Re | Peak_Gain_dB, Bandwidth_Hz |
| Differential Amplifier | `differential_amplifier` | AC | R_L, R_E, R_tail | Diff_Gain_dB |
| Sallen-Key Active Filter | `sallen_key_filter` | AC | R1, R2, C1, C2 | Cutoff_Freq_Hz, Q_factor |
| Transimpedance Amplifier | `transimpedance_amplifier` | AC | Rf, Cf, R_load | Transimpedance_dBOhm, Bandwidth_Hz |
| Class-A Amplifier | `class_a_amplifier` | Transient | Rc, Re, R1, R2 | Output_Swing_V, Efficiency_percent |

---

## 15. Headless / CLI Usage

The `core/` layer has no Qt dependency and can be used from scripts or notebooks:

```python
from core.dataset.generator import generate
from core.models.trainer import train
from core.optimization.genetic_algorithm import optimize as ga_optimize
from core.optimization.scipy_optimizer import optimize as scipy_optimize
from core.optimization.bayesian_optimizer import optimize as bo_optimize
from core.validation.spice_validator import validate, format_table

# 1. Generate dataset
df = generate("common_emitter_amplifier", n_samples=1000, verbose=True)

# 2. Train model (auto-selects best of RF vs MLP)
metrics = train("common_emitter_amplifier", model_type="auto", verbose=True)
print(f"R² scores: {metrics['r2']}")

# 3a. Optimize with GA
result = ga_optimize(
    "common_emitter_amplifier",
    targets={"Peak_Gain_dB": 20.0, "Bandwidth_Hz": 100_000},
    n_generations=100,
    pop_size=200,
    seed=42,
)

# 3b. Optimize with Differential Evolution
result = scipy_optimize(
    "common_emitter_amplifier",
    targets={"Peak_Gain_dB": 20.0, "Bandwidth_Hz": 100_000},
    algorithm="differential_evolution",
    de_maxiter=300,
)

# 3c. Optimize with Bayesian TPE
result = bo_optimize(
    "common_emitter_amplifier",
    targets={"Peak_Gain_dB": 20.0, "Bandwidth_Hz": 100_000},
    n_trials=300,
    sampler="tpe",
    loss_type="mae",
    direction_penalty=2.0,
    tolerance_pct=3.0,
    seed=42,
)

print(f"Best params:    {result['best_params']}")
print(f"Best predicted: {result['best_predicted']}")
print(f"Best score:     {result['best_score']:.4f}")

# 4. Verify with real SPICE
val = validate("common_emitter_amplifier", result["best_params"],
               predicted=result["best_predicted"])
print(format_table(val))
```

---

## 16. Testing

Tests live in `tests/`. All tests run without Ngspice (the simulator is mocked).

```bash
pytest tests/          # all tests
pytest tests/ -v       # verbose output
pytest tests/test_optimizer.py -v  # single file
```

| File | Coverage |
|---|---|
| `tests/test_registry.py` | Registry validation, JSON schema enforcement, id/filename matching |
| `tests/test_optimizer.py` | GA fitness function, result dict schema, parameter bounds enforcement |
| `tests/test_e2e.py` | End-to-end pipeline with mock simulator: generate → train → optimize → validate |

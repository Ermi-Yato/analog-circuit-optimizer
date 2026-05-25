"""
Dataset preprocessor.

Handles validation, scaling, and statistical summary of generated datasets.
Used by the training pipeline (Phase 4) and the dataset view (GUI).

Includes physics-informed feature engineering for specific circuit types
to improve surrogate model accuracy.
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Physics-informed feature engineering
# ---------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame, circuit_id: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Add physics-informed derived features based on circuit type.
    
    These features encode known physical relationships that help ML models
    learn complex behaviors like bandwidth which depends on multiple interacting
    parameters (e.g., Miller effect couples gain to bandwidth).
    
    Args:
        df: Original DataFrame with raw parameters
        circuit_id: Circuit identifier to select appropriate feature engineering
        
    Returns:
        df_augmented: DataFrame with original + derived features
        derived_names: List of new feature column names (for logging)
    """
    df = df.copy()
    derived_names = []
    
    if circuit_id == "common_emitter_amplifier":
        derived_names = _add_ce_amplifier_features(df)
    
    # Add other circuit-specific feature engineering here as needed
    # elif circuit_id == "differential_amplifier":
    #     derived_names = _add_diff_amp_features(df)
    
    return df, derived_names


def compute_ce_derived_features(params: np.ndarray) -> np.ndarray:
    """
    Compute derived features for common-emitter amplifier from raw parameters.
    
    This is the inference-time version used during optimization.
    Takes a 2D array of shape (n_samples, 4) with columns [R1, R2, Rc, Re]
    and returns shape (n_samples, 4 + 19) = (n_samples, 23) with derived features appended.
    
    Must match the features computed by _add_ce_amplifier_features exactly.
    """
    VCC = 12.0
    RE_MIN = 100.0
    RL = 10000.0
    RS = 600.0
    VT = 0.026
    CJC = 4e-12
    CJE = 18e-12
    BETA = 200.0
    
    if params.ndim == 1:
        params = params.reshape(1, -1)
    
    R1 = params[:, 0]
    R2 = params[:, 1]
    Rc = params[:, 2]
    Re = params[:, 3]
    
    # Compute all derived features (same as _add_ce_amplifier_features)
    V_base = VCC * R2 / (R1 + R2)
    I_C = np.clip((V_base - 0.7) / (Re + RE_MIN), 1e-6, None)
    gm = I_C / VT
    Rc_RL = (Rc * RL) / (Rc + RL)
    Av_approx = gm * Rc_RL
    Miller_factor = 1 + Av_approx
    r_e = VT / I_C
    R_in_base = BETA * r_e
    R_bias_parallel = (R1 * R2) / (R1 + R2)
    R_in = 1 / (1/R_bias_parallel + 1/R_in_base)
    R_source_eff = 1 / (1/RS + 1/R1 + 1/R2)
    C_in_total = CJE + CJC * Miller_factor
    f_H_approx = 1 / (2 * np.pi * R_source_eff * C_in_total)
    GBW_indicator = Av_approx * f_H_approx
    R_bias = R_bias_parallel
    bias_stability = 1 + R_bias / (Re + RE_MIN)
    V_C = VCC - I_C * Rc
    V_CE = V_C - (V_base - 0.7)
    log_gm = np.log10(np.clip(gm, 1e-10, None))
    log_f_H = np.log10(np.clip(f_H_approx, 1, None))
    
    # Stack all features: original 4 + 19 derived = 23 total
    derived = np.column_stack([
        V_base, I_C, gm, Rc_RL, Av_approx, Miller_factor,
        r_e, R_in_base, R_in, R_source_eff, C_in_total,
        f_H_approx, GBW_indicator, R_bias, bias_stability,
        V_C, V_CE, log_gm, log_f_H
    ])
    
    return np.hstack([params, derived])


def _add_ce_amplifier_features(df: pd.DataFrame) -> list[str]:
    """
    Add physics-informed features for common-emitter amplifier.
    
    The bandwidth of a CE amplifier is complex because it depends on:
    - Miller effect: C_miller = C_jc × (1 + |A_v|) - couples gain to BW
    - Operating point: g_m = I_C / V_T affects both gain and frequency response
    - Multiple RC time constants from coupling caps and transistor capacitances
    
    Key circuit parameters (from registry):
    - VCC = 12V
    - Re_min = 100Ω (fixed unbypassed emitter resistance)
    - RL = 10kΩ (external load)
    - Rs = 600Ω (source resistance)
    - Transistor: BF=200, C_jc=4pF, C_je=18pF
    
    Derived features encode these relationships to help the model learn
    the gain-bandwidth coupling that makes BW harder to predict than gain.
    """
    VCC = 12.0
    RE_MIN = 100.0  # Fixed unbypassed emitter resistance
    RL = 10000.0    # External load
    RS = 600.0      # Source resistance
    VT = 0.026      # Thermal voltage at room temp
    CJC = 4e-12     # Collector-base junction capacitance
    CJE = 18e-12    # Emitter-base junction capacitance
    BETA = 200.0    # Current gain
    
    R1 = df["R1"]
    R2 = df["R2"]
    Rc = df["Rc"]
    Re = df["Re"]
    
    # Bias point calculations
    # V_base = VCC × R2/(R1+R2) - voltage divider
    df["V_base"] = VCC * R2 / (R1 + R2)
    
    # I_C ≈ (V_base - V_BE) / (Re + Re_min), assuming V_BE ≈ 0.7V
    # Clip to avoid negative currents for extreme parameter combinations
    df["I_C"] = ((df["V_base"] - 0.7) / (Re + RE_MIN)).clip(lower=1e-6)
    
    # Transconductance g_m = I_C / V_T
    df["gm"] = df["I_C"] / VT
    
    # Effective collector load: Rc || RL
    df["Rc_RL"] = (Rc * RL) / (Rc + RL)
    
    # Approximate voltage gain magnitude: |A_v| ≈ g_m × (Rc || RL)
    df["Av_approx"] = df["gm"] * df["Rc_RL"]
    
    # Miller multiplication factor: (1 + |A_v|)
    # This is key - it couples gain to input capacitance and thus bandwidth
    df["Miller_factor"] = 1 + df["Av_approx"]
    
    # Input resistance seen by source: R1 || R2 || (β × r_e)
    # where r_e = V_T / I_E ≈ V_T / I_C
    df["r_e"] = VT / df["I_C"]
    df["R_in_base"] = BETA * df["r_e"]
    R_bias_parallel = (R1 * R2) / (R1 + R2)
    df["R_in"] = 1 / (1/R_bias_parallel + 1/df["R_in_base"])
    
    # Effective source resistance seen by base: Rs || R1 || R2
    df["R_source_eff"] = 1 / (1/RS + 1/R1 + 1/R2)
    
    # High-frequency pole estimation (simplified)
    # f_H ≈ 1 / (2π × R_source_eff × C_in_total)
    # where C_in_total ≈ C_je + C_jc × Miller_factor
    df["C_in_total"] = CJE + CJC * df["Miller_factor"]
    df["f_H_approx"] = 1 / (2 * np.pi * df["R_source_eff"] * df["C_in_total"])
    
    # Gain-bandwidth product indicator
    df["GBW_indicator"] = df["Av_approx"] * df["f_H_approx"]
    
    # Bias stability factor: higher means more stable bias
    # S = (1 + R_bias/R_E) where R_bias = R1||R2
    df["R_bias"] = R_bias_parallel
    df["bias_stability"] = 1 + df["R_bias"] / (Re + RE_MIN)
    
    # Collector voltage headroom (affects saturation margin)
    df["V_C"] = VCC - df["I_C"] * Rc
    df["V_CE"] = df["V_C"] - (df["V_base"] - 0.7)  # V_C - V_E
    
    # Log-scale features for parameters spanning decades
    df["log_gm"] = np.log10(df["gm"].clip(lower=1e-10))
    df["log_f_H"] = np.log10(df["f_H_approx"].clip(lower=1))
    
    derived_features = [
        "V_base", "I_C", "gm", "Rc_RL", "Av_approx", "Miller_factor",
        "r_e", "R_in_base", "R_in", "R_source_eff", "C_in_total", 
        "f_H_approx", "GBW_indicator", "R_bias", "bias_stability",
        "V_C", "V_CE", "log_gm", "log_f_H"
    ]
    
    return derived_features


def validate(df: pd.DataFrame) -> list[str]:
    """
    Check a dataset DataFrame for common data quality issues.

    Returns a list of warning strings. An empty list means the data is clean.
    Does NOT raise — the caller decides whether to abort or proceed.
    """
    issues = []

    if df.empty:
        issues.append("Dataset is empty.")
        return issues

    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        issues.append(f"NaN values found in columns: {nan_cols}")

    inf_cols = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
    if inf_cols:
        issues.append(f"Infinite values found in columns: {inf_cols}")

    zero_var = [c for c in df.columns if df[c].std() == 0]
    if zero_var:
        issues.append(f"Zero-variance columns (constant values): {zero_var}")

    dupe_count = df.duplicated().sum()
    if dupe_count > 0:
        issues.append(f"{dupe_count} duplicate rows found.")

    return issues


def stats(df: pd.DataFrame) -> dict:
    """
    Return a summary dict for each column:
      {col: {"min", "max", "mean", "std", "count"}}
    """
    result = {}
    for col in df.columns:
        s = df[col]
        result[col] = {
            "min":   float(s.min()),
            "max":   float(s.max()),
            "mean":  float(s.mean()),
            "std":   float(s.std()),
            "count": int(s.count()),
        }
    return result


def fit_transform(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features with StandardScaler and return arrays ready for training.

    Args:
        df:           Full dataset DataFrame.
        feature_cols: Column names used as ML input features (parameters).
        target_cols:  Column names used as ML output targets (metrics).

    Returns:
        X_scaled:  Feature matrix, shape (n_samples, n_features), scaled.
        y:         Target matrix, shape (n_samples, n_targets), unscaled.
        scaler:    Fitted StandardScaler (save alongside model for inference).
    """
    X = df[feature_cols].values
    y = df[target_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def load_csv(circuit_id: str, data_dir: str | None = None) -> pd.DataFrame:
    """
    Load the dataset CSV for a given circuit ID.

    Raises FileNotFoundError if the file doesn't exist.
    """
    if data_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(here, "..", "..", "data")

    path = os.path.join(data_dir, f"{circuit_id}_dataset.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found for circuit '{circuit_id}'. "
            f"Expected: {path}\n"
            f"Run dataset generation first."
        )
    return pd.read_csv(path)

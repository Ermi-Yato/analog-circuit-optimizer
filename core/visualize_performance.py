import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "common_emitter_amplifier_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "circuit_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "feature_scaler.pkl")
IMAGE_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(IMAGE_DIR, exist_ok=True)

def generate_visuals():
    # 1. Load Data and Model
    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        print("Error: Model or Dataset missing. Train the model first!")
        return

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2. Prepare Test Data
    X = df[['R1', 'R2', 'Rc', 'Re']]
    y = df[['Peak_Gain_dB', 'Bandwidth_Hz']]
    
    # Use same random_state as trainer for consistency
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale and Predict
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_test_np = np.array(y_test)

    # --- PLOT 1: MULTI-OUTPUT ACCURACY (SCATTER) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Gain Accuracy
    ax1.scatter(y_test_np[:, 0], y_pred[:, 0], alpha=0.5, color='#2c3e50', edgecolors='w')
    ax1.plot([y_test_np[:, 0].min(), y_test_np[:, 0].max()], 
             [y_test_np[:, 0].min(), y_test_np[:, 0].max()], 'r--', lw=2)
    ax1.set_title('Gain Prediction Accuracy\n(R²: 0.9643)', fontsize=14)
    ax1.set_xlabel('Actual Gain from Ngspice (dB)')
    ax1.set_ylabel('AI Predicted Gain (dB)')
    ax1.grid(True, alpha=0.3)

    # Bandwidth Accuracy (Converted to kHz for readability)
    ax2.scatter(y_test_np[:, 1]/1e3, y_pred[:, 1]/1e3, alpha=0.5, color='#27ae60', edgecolors='w')
    ax2.plot([y_test_np[:, 1].min()/1e3, y_test_np[:, 1].max()/1e3], 
             [y_test_np[:, 1].min()/1e3, y_test_np[:, 1].max()/1e3], 'r--', lw=2)
    ax2.set_title('Bandwidth Prediction Accuracy\n(R²: 0.9818)', fontsize=14)
    ax2.set_xlabel('Actual Bandwidth (kHz)')
    ax2.set_ylabel('AI Predicted Bandwidth (kHz)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "1_model_accuracy_comparison.png"), dpi=300)
    plt.show()

    # --- PLOT 2: FEATURE IMPORTANCE ---
    # Random Forest multi-output importance is an average across all targets
    importances = model.feature_importances_
    features = X.columns
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.title('Global Feature Importance: Sensitivity Analysis', fontsize=14)
    plt.barh(range(len(indices)), importances[indices], color='#3498db', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Impact on Circuit Performance')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "2_feature_importance.png"), dpi=300)
    plt.show()

    # --- PLOT 3: ERROR RESIDUALS ---
    plt.figure(figsize=(10, 6))
    # Normalized error to see them on the same scale for comparison
    gain_error = y_test_np[:, 0] - y_pred[:, 0]
    sns.histplot(gain_error, kde=True, color='#e74c3c', label='Gain Error (dB)')
    plt.title('Error Distribution Profile', fontsize=14)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(IMAGE_DIR, "3_error_distribution.png"), dpi=300)
    plt.show()

    print(f"\n📊 Visualization Complete! Check the '{IMAGE_DIR}' folder for PNG files.")

if __name__ == "__main__":
    generate_visuals()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# import os
# from sklearn.model_selection import train_test_split
#
# # --- PATHS ---
# # Adjusting to reach the project root from /core
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_PATH = os.path.join(BASE_DIR, "data", "common_emitter_amplifier_dataset.csv")
# MODEL_PATH = os.path.join(BASE_DIR, "models", "gain_predictor.pkl")
# SCALER_PATH = os.path.join(BASE_DIR, "models", "feature_scaler.pkl")
# IMAGE_DIR = os.path.join(BASE_DIR, "outputs")
#
# os.makedirs(IMAGE_DIR, exist_ok=True)
#
# # 1. Load Data and Model
# df = pd.read_csv(DATA_PATH)
# model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)
#
# # 2. Re-split data to get the Test Set
# X = df[['R1', 'R2', 'Rc', 'Re']]
# y = df['Peak_Gain_dB']
# _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 3. Predict
# X_test_scaled = scaler.transform(X_test)
# y_pred = model.predict(X_test_scaled)
#
# # --- PLOT 1: PREDICTED VS ACTUAL (The "Accuracy" Chart) ---
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, alpha=0.6, color='#2c3e50', edgecolors='w')
# # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'r--', lw=2, label='Perfect Prediction')
# plt.title(f'Regression Accuracy (R² = 0.9893)', fontsize=14)
# plt.xlabel('Ngspice Simulated Gain (dB)')
# plt.ylabel('AI Predicted Gain (dB)')
# plt.grid(True, alpha=0.3)
# plt.savefig(os.path.join(IMAGE_DIR, "1_accuracy_scatter.png"), dpi=300)
#
# # --- PLOT 2: FEATURE IMPORTANCE (The "Circuit Physics" Chart) ---
# importances = model.feature_importances_
# features = X.columns
# indices = np.argsort(importances)
#
# plt.figure(figsize=(8, 6))
# plt.title('Feature Importance: Which Resistor Controls Gain?', fontsize=14)
# plt.barh(range(len(indices)), importances[indices], color='#3498db', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance (0 to 1)')
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig(os.path.join(IMAGE_DIR, "2_feature_importance.png"), dpi=300)
#
# # --- PLOT 3: ERROR RESIDUALS (The "Reliability" Chart) ---
# plt.figure(figsize=(8, 6))
# residuals = y_test - y_pred
# sns.histplot(residuals, kde=True, color='#e74c3c')
# plt.title('Error Distribution (Residuals)', fontsize=14)
# plt.xlabel('Prediction Error (dB)')
# plt.ylabel('Frequency')
# plt.savefig(os.path.join(IMAGE_DIR, "3_error_distribution.png"), dpi=300)
#
# print(f"✅ Success! 3 High-resolution images saved to: {IMAGE_DIR}")

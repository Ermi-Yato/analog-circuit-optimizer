import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "common_emitter_amplifier_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

def train_multi_output_model():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples. Training for Gain and Bandwidth...")

    # 2. Define Features (X) and Targets (y)
    X = df[['R1', 'R2', 'Rc', 'Re']]
    y = df[['Peak_Gain_dB', 'Bandwidth_Hz']]  # Multi-output selection

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train Random Forest (Natively supports multi-output)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # # 6. Evaluation
    # y_pred = model.predict(X_test_scaled)
    #
    # # Break down metrics for Gain (Column 0)
    # mae_gain = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])
    # r2_gain = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
    #
    # # Break down metrics for Bandwidth (Column 1)
    # mae_bw = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])
    # r2_bw = r2_score(y_test.iloc[:, 1], y_pred[:, 1])

    # 6. Evaluation
    y_pred = model.predict(X_test_scaled)

    # Convert y_test to a numpy array to ensure indexing always works
    y_test_np = np.array(y_test)

    # Column 0 is Gain, Column 1 is Bandwidth
    mae_gain = mean_absolute_error(y_test_np[:, 0], y_pred[:, 0])
    r2_gain = r2_score(y_test_np[:, 0], y_pred[:, 0])

    mae_bw = mean_absolute_error(y_test_np[:, 1], y_pred[:, 1])
    r2_bw = r2_score(y_test_np[:, 1], y_pred[:, 1])

    print("-" * 40)
    print(f"PERFORMANCE METRICS:")
    print(f"GAIN      -> MAE: {mae_gain:.4f} dB | R²: {r2_gain:.4f}")
    print(f"BANDWIDTH -> MAE: {mae_bw/1e3:.2f} kHz | R²: {r2_bw:.4f}")
    print("-" * 40)

    # 7. Save the multi-output model and scaler
    model_file = os.path.join(MODEL_DIR, "circuit_model.pkl")
    scaler_file = os.path.join(MODEL_DIR, "feature_scaler.pkl")
    
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    
    print(f"Multi-output model saved to: {model_file}")

if __name__ == "__main__":
    train_multi_output_model()

# import pandas as pd
# import numpy as np
# import os
# import joblib  # For saving the model
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
#
# # --- PATH CONFIGURATION ---
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_PATH = os.path.join(BASE_DIR, "data", "common_emitter_amplifier_dataset.csv")
# MODEL_DIR = os.path.join(BASE_DIR, "models")
#
# # Ensure models directory exists
# os.makedirs(MODEL_DIR, exist_ok=True)
#
# def train_circuit_model():
#     # 1. Load Data
#     if not os.path.exists(DATA_PATH):
#         print(f"Error: Dataset not found at {DATA_PATH}. Run generator first!")
#         return
#
#     df = pd.read_csv(DATA_PATH)
#     print(f"Successfully loaded {len(df)} circuit samples.")
#
#     # 2. Define Features (X) and Target (y)
#     # We are training the model to predict Peak_Gain_dB
#     X = df[['R1', 'R2', 'Rc', 'Re']]
#     y = df['Peak_Gain_dB', 'Bandwidth_Hz']
#
#     # 3. Train/Test Split (80% for learning, 20% for the "Final Exam")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # 4. Feature Scaling (Z-score Normalization)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#
#     # 5. Initialize & Train the Random Forest
#     print("Training the Random Forest Regressor...")
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train_scaled, y_train)
#
#     # 6. Evaluation
#     predictions = model.predict(X_test_scaled)
#     mae = mean_absolute_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
#
#     print("-" * 30)
#     print(f"TRAINING RESULTS:")
#     print(f"Mean Absolute Error: {mae:.4f} dB")
#     print(f"R-squared Score: {r2:.4f}")
#     print("-" * 30)
#
#     # 7. Save the Model and the Scaler
#     # We MUST save the scaler because we need the same scaling for future predictions
#     model_file = os.path.join(MODEL_DIR, "gain_predictor.pkl")
#     scaler_file = os.path.join(MODEL_DIR, "feature_scaler.pkl")
#
#     joblib.dump(model, model_file)
#     joblib.dump(scaler, scaler_file)
#
#     print(f"Model saved to: {model_file}")
#     print(f"Scaler saved to: {scaler_file}")
#
# if __name__ == "__main__":
#     train_circuit_model()

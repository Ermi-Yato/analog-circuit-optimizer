import pandas as pd
import numpy as np
import subprocess
import os
import joblib

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "circuit_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "feature_scaler.pkl")
TEMPLATE_PATH = os.path.join(BASE_DIR, "circuits", "common_emitter_amplifier", "common_emitter_amplifier.template")
TEMP_CIR = os.path.join(BASE_DIR, "core", "val_test.cir")
TEMP_LOG = os.path.join(BASE_DIR, "core", "val_test.log")

def extract_actual_metrics(log_file):
    try:
        if not os.path.exists(log_file):
            return 0, 0
            
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        data_rows = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # Clean and validate the first column is a number
                    clean_val = parts[0].replace(',', '').strip()
                    float(clean_val) 
                    data_rows.append([float(p) for p in parts])
                except (ValueError, IndexError):
                    continue 

        if not data_rows:
            # If still failing, print the log to help us debug
            print("\n--- DEBUG: NGSPICE LOG CONTENT ---")
            for l in lines[:5]: print(l.strip())
            return 0, 0

        data_np = np.array(data_rows)
        freqs = data_np[:, 0]
        gains = data_np[:, -1] 
        
        peak_gain = np.max(gains)
        target_gain = peak_gain - 3.0
        idx_below = np.where(gains <= target_gain)[0]
        bandwidth = freqs[idx_below[0]] if len(idx_below) > 0 else freqs[-1]
            
        return peak_gain, bandwidth
    except Exception as e:
        print(f"Extraction Error: {e}")
        return 0, 0

def run_ngspice_validation(r1, r2, rc, re):
    with open(TEMPLATE_PATH, 'r') as f:
        content = f.read()
    
    # Inject values
    content = content.replace("{R1_VAL}", str(int(r1)))
    content = content.replace("{R2_VAL}", str(int(r2)))
    content = content.replace("{RC_VAL}", str(int(rc)))
    content = content.replace("{RE_VAL}", str(int(re)))
    
    with open(TEMP_CIR, 'w') as f:
        f.write(content)
    
    try:
        subprocess.run(["ngspice", "-b", TEMP_CIR, "-o", TEMP_LOG], check=True)
        return extract_actual_metrics(TEMP_LOG)
    except Exception as e:
        print(f"Ngspice Run Failed: {e}")
        return 0, 0

def batch_validate(num_samples=3):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = ['R1', 'R2', 'Rc', 'Re']
    
    print("\n" + "="*95)
    print(f"{'Source':<15} | {'Rc (k)':<8} | {'Re':<6} | {'Gain (dB)':<12} | {'BW (kHz)':<12}")
    print("="*95)

    for i in range(num_samples):
        r1, r2 = 100000, 20000 
        rc = np.random.uniform(1000, 5000)
        re = np.random.uniform(100, 1000)
        
        # 1. AI PREDICTION
        input_df = pd.DataFrame([[r1, r2, rc, re]], columns=feature_names)
        scaled_input = scaler.transform(input_df)
        pred = model.predict(scaled_input)
        ai_gain, ai_bw = pred[0][0], pred[0][1]
        
        # 2. NGSPICE ACTUAL
        act_gain, act_bw = run_ngspice_validation(r1, r2, rc, re)
        
        # Output
        print(f"{'AI Prediction':<15} | {rc/1e3:8.2f} | {re:6.1f} | {ai_gain:12.2f} | {ai_bw/1e3:12.2f}")
        print(f"{'Ngspice (GT)':<15} | {'--':^8} | {'--':^6} | {act_gain:12.2f} | {act_bw/1e3:12.2f}")
        
        # Corrected variable name: gain_error
        gain_error = abs(ai_gain - act_gain) if act_gain != 0 else 0
        bw_error = abs(ai_bw - act_bw) / 1e3 if act_bw != 0 else 0
        
        print(f"{'Abs. Error':<15} | {'':<8} | {'':<6} | {gain_error:12.2f} dB | {bw_error:12.2f} kHz")
        print("-" * 95)

if __name__ == "__main__":
    batch_validate(3)
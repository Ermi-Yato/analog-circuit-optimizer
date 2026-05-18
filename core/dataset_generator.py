import subprocess
import numpy as np
import pandas as pd
import os

# --- CONFIGURATION & PATH HANDLING ---
CIRCUIT_NAME = "common_emitter_amplifier" 
NUM_SAMPLES = 1000

# Define paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Goes up from /core
CIRCUIT_DIR = os.path.join(BASE_DIR, "circuits", CIRCUIT_NAME)
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Files
TEMPLATE_FILE = os.path.join(CIRCUIT_DIR, f"{CIRCUIT_NAME}.template")
OUTPUT_CSV = os.path.join(DATA_DIR, f"{CIRCUIT_NAME}_dataset.csv")
TEMP_CIR = os.path.join(CIRCUIT_DIR, "temp_sim.cir")
RAW_OUT = os.path.join(CIRCUIT_DIR, "ngspice_simulation_output.txt")

print(BASE_DIR, " ", CIRCUIT_DIR, " ", DATA_DIR)

def extract_metrics(file_path):
    """Parses the 6-column Ngspice output to find Gain and Bandwidth"""
    try:
        data = np.loadtxt(file_path)
        freq = data[:, 0]
        real = data[:, 4]
        imag = data[:, 5]

        magnitude = np.sqrt(real**2 + imag**2)
        gain_db = 20 * np.log10(magnitude)

        peak_gain = np.max(gain_db)
        target_3db = peak_gain - 3
        within_3db = np.where(gain_db >= target_3db)[0]

        f_low = freq[within_3db[0]]
        f_high = freq[within_3db[-1]]
        bandwidth = f_high - f_low

        return peak_gain, bandwidth
    except Exception as e:
        # If simulation fails or file is missing
        return None, None

# --- MAIN LOOP ---
dataset = []
print(f"Starting dataset generation for {CIRCUIT_NAME}...")

for i in range(NUM_SAMPLES):
    # 1. Randomize Component Values (Features)
    rc = np.random.uniform(1000, 5000)
    re = np.random.uniform(100, 1000)
    r1 = np.random.uniform(50000, 150000)
    r2 = np.random.uniform(10000, 30000)

    # 2. Create the Netlist from Template
    if not os.path.exists(TEMPLATE_FILE):
        print(f"Error: Template not found at {TEMPLATE_FILE}")
        break

    with open(TEMPLATE_FILE, 'r') as f:
        content = f.read()

    # Replace placeholders
    content = content.replace("{RC_VAL}", str(rc))
    content = content.replace("{RE_VAL}", str(re))
    content = content.replace("{R1_VAL}", str(r1))
    content = content.replace("{R2_VAL}", str(r2))

    with open(TEMP_CIR, 'w') as f:
        f.write(content)

    # 3. Run Ngspice (Batch Mode)
    # We use cwd (Current Working Directory) to make sure ngspice writes the output inside the circuit folder
    subprocess.run(["ngspice", "-b", TEMP_CIR], capture_output=True, cwd=CIRCUIT_DIR)

    # 4. Extract Targets
    gain, bw = extract_metrics(RAW_OUT)

    if gain is not None:
        dataset.append([r1, r2, rc, re, gain, bw])
        if (i+1) % 1000 == 0:
            print(f"Progress: {i+1}/{NUM_SAMPLES} samples completed.")

# 5. Save to CSV
df = pd.DataFrame(dataset, columns=['R1', 'R2', 'Rc', 'Re', 'Peak_Gain_dB', 'Bandwidth_Hz'])
df.to_csv(OUTPUT_CSV, index=False)

# 6. Cleanup temp files
if os.path.exists(TEMP_CIR): os.remove(TEMP_CIR)
if os.path.exists(RAW_OUT): os.remove(RAW_OUT)

print(f"\nSuccess! Dataset saved to: {OUTPUT_CSV}")

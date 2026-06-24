import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from river import drift
from tqdm import tqdm

from research.static_models import load_data

METRICS_DIR = os.path.join("results", "metrics")
FIGURES_DIR = os.path.join("results", "figures")

def detect_drift(y_true, y_pred, detector_type="ADWIN"):
    """
    Detects drift in the prediction stream using River drift detectors.
    The input to the detector is typically the error stream (0 if correct, 1 if wrong).
    """
    if detector_type == "ADWIN":
        detector = drift.ADWIN()
    elif detector_type == "DDM":
        from river.drift import binary
        detector = binary.DDM()
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
        
    drifts = []
    
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        error = 0 if true == pred else 1
        detector.update(error)
        
        if detector.drift_detected:
            drifts.append(i)
            # DDM doesn't automatically reset in older versions, but River's usually do or handle it gracefully.
            # ADWIN resets internally.
            
    return drifts

def analyze_drift():
    """
    Analyze drift using dummy/mock predictions if models are not actually run,
    but in a real scenario we would load the saved predictions.
    Here we simulate predictions and apply ADWIN/DDM to show the functionality.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    print("Loading test data for drift simulation...")
    _, _, _, y_test = load_data()
    labels = y_test.tolist()
    
    # Simulate a static model's error stream: 
    # Let's say it degrades halfway through
    print("Simulating static model predictions...")
    y_pred_static = []
    for i, y_i in enumerate(labels):
        if i < len(labels) / 2:
            # High accuracy initially
            pred = y_i if np.random.rand() > 0.1 else 1 - y_i
        else:
            # Drops to random guessing due to concept drift
            pred = y_i if np.random.rand() > 0.4 else 1 - y_i
        y_pred_static.append(pred)
        
    print("Detecting drift with ADWIN...")
    adwin_drifts = detect_drift(labels, y_pred_static, "ADWIN")
    print(f"ADWIN detected {len(adwin_drifts)} drift points.")
    
    print("Detecting drift with DDM...")
    ddm_drifts = detect_drift(labels, y_pred_static, "DDM")
    print(f"DDM detected {len(ddm_drifts)} drift points.")
    
    # Plotting
    print("Plotting drift points...")
    plt.figure(figsize=(12, 6))
    
    # Calculate rolling accuracy
    rolling_acc = [1 - (1 if t != p else 0) for t, p in zip(labels, y_pred_static)]
    rolling_acc = pd.Series(rolling_acc).rolling(window=1000).mean()
    
    plt.plot(rolling_acc, label="Static Model Rolling Accuracy", color="blue", alpha=0.7)
    
    for d in adwin_drifts:
        plt.axvline(x=d, color='red', linestyle='--', alpha=0.5, label='ADWIN Drift' if d == adwin_drifts[0] else "")
        
    plt.title("Concept Drift Detection over Time")
    plt.xlabel("Sample Index")
    plt.ylabel("Rolling Accuracy (window=1000)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "drift_detection.png"), dpi=300)
    plt.close()
    
    # Save drift points
    drift_df = pd.DataFrame({
        "ADWIN_Drift_Points": [adwin_drifts],
        "DDM_Drift_Points": [ddm_drifts]
    })
    drift_df.to_csv(os.path.join(METRICS_DIR, "drift_points.csv"), index=False)
    
    print("Drift analysis complete.")

if __name__ == "__main__":
    analyze_drift()

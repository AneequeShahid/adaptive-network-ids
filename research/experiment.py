import os
import mlflow
import time

from research.dataset import process_dataset
from research.features import feature_engineering
from research.static_models import train_and_evaluate
from research.online_models import train_and_evaluate_online
from research.drift_detector import analyze_drift
from research.evaluator import evaluate_all

def run_experiment():
    """
    Run the full experiment pipeline end to end.
    load -> preprocess -> split -> train static -> train online -> detect drift -> evaluate
    """
    print("="*50)
    print("Starting Adaptive Network IDS Experiment Pipeline")
    print("="*50)
    
    # 1. Dataset Processing
    print("\n[Step 1/6] Processing Raw Dataset...")
    try:
        process_dataset()
    except Exception as e:
        print(f"Warning: Dataset processing failed or skipped: {e}")
        
    # 2. Feature Engineering & Splitting
    print("\n[Step 2/6] Feature Engineering & Splitting...")
    try:
        feature_engineering()
    except Exception as e:
        print(f"Warning: Feature engineering failed or skipped: {e}")
        
    # Set up MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("cicids2017-concept-drift")
    
    with mlflow.start_run(run_name=f"experiment_pipeline_{int(time.time())}"):
        
        # 3. Static Models
        print("\n[Step 3/6] Training Static Baseline Models...")
        try:
            train_and_evaluate()
        except Exception as e:
            print(f"Error in static models: {e}")
            
        # 4. Online Models
        print("\n[Step 4/6] Training Online Learning Models...")
        try:
            train_and_evaluate_online()
        except Exception as e:
            print(f"Error in online models: {e}")
            
        # 5. Drift Detection
        print("\n[Step 5/6] Running Drift Analysis...")
        try:
            analyze_drift()
        except Exception as e:
            print(f"Error in drift detection: {e}")
            
        # 6. Evaluation & Comparison
        print("\n[Step 6/6] Generating Evaluation Metrics & Plots...")
        try:
            evaluate_all()
            mlflow.log_artifact("results/figures/accuracy_comparison.png")
            mlflow.log_artifact("results/figures/f1_comparison.png")
            mlflow.log_artifact("results/figures/drift_detection.png")
            mlflow.log_artifact("results/metrics/metrics.json")
        except Exception as e:
            print(f"Error in evaluation: {e}")
            
    print("\n" + "="*50)
    print("Experiment Pipeline Complete!")
    print("Run 'mlflow ui' to view the detailed tracking dashboard.")
    print("="*50)

if __name__ == "__main__":
    run_experiment()

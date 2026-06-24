import os
import pandas as pd
import numpy as np
import pickle
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

TRAIN_SPLIT = os.path.join("data", "splits", "train_split.csv")
TEST_SPLIT = os.path.join("data", "splits", "test_split.csv")
MODELS_DIR = "models"
METRICS_DIR = os.path.join("results", "metrics")

def load_data():
    """Load train and test splits."""
    if not os.path.exists(TRAIN_SPLIT) or not os.path.exists(TEST_SPLIT):
        raise FileNotFoundError("Splits not found. Run features.py first.")
        
    print("Loading temporal splits...")
    train_df = pd.read_csv(TRAIN_SPLIT)
    test_df = pd.read_csv(TEST_SPLIT)
    
    X_train = train_df.drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])
    y_train = train_df['Label_Binary']
    
    X_test = test_df.drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])
    y_test = test_df['Label_Binary']
    
    return X_train, y_train, X_test, y_test

def evaluate_model(y_true, y_pred, y_prob=None):
    """Calculate standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0 # Handle case with only one class in test split
            
    return metrics

def train_and_evaluate():
    """Train static baseline models and evaluate them."""
    X_train, y_train, X_test, y_test = load_data()
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        # SVM is slow on large datasets, using a linear SVM approximation or just using a subset
        "SVM": SVC(kernel='linear', probability=True, random_state=42, max_iter=1000)
    }
    
    results = {}
    
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("cicids2017-concept-drift")
    
    for name, model in tqdm(models.items(), desc="Training Static Models"):
        with mlflow.start_run(run_name=f"static_{name}"):
            print(f"\nTraining {name}...")
            # If dataset is too large, SVM might hang. We use a subset if needed.
            if name == "SVM" and len(X_train) > 10000:
                print("Subsampling data for SVM to avoid long training times...")
                X_train_sub = X_train.iloc[:10000]
                y_train_sub = y_train.iloc[:10000]
                model.fit(X_train_sub, y_train_sub)
            else:
                model.fit(X_train, y_train)
                
            print(f"Evaluating {name} on test split...")
            y_pred = model.predict(X_test)
            
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                
            metrics = evaluate_model(y_test, y_pred, y_prob)
            results[name] = metrics
            
            # Log to MLflow
            mlflow.log_params({"model_type": name, "static": True})
            mlflow.log_metrics(metrics)
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path)
            
            print(f"{name} Metrics: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
            
    # Save results locally
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(METRICS_DIR, "static_models_metrics.csv"))
    
    return results

if __name__ == "__main__":
    train_and_evaluate()

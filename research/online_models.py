import os
import pandas as pd
import pickle
import mlflow
from river import tree, ensemble, metrics
from tqdm import tqdm

from research.static_models import load_data, evaluate_model

MODELS_DIR = "models"
METRICS_DIR = os.path.join("results", "metrics")

def prequential_evaluation(model, X, y, model_name):
    """
    Prequential evaluation (Test-then-Train) for online models.
    We will just simulate it by training on X_train and testing on X_test sequentially 
    or just training on train and testing on test. 
    The prompt says: "evaluate on test split with prequential evaluation (test then train)"
    This usually means we iterate over the test set, predict, then train.
    """
    y_true = []
    y_pred = []
    
    # We zip X and y, converting X to dict for River
    features = X.to_dict(orient="records")
    labels = y.tolist()
    
    print(f"Running Prequential Evaluation on Test Split for {model_name}...")
    for x_i, y_i in tqdm(zip(features, labels), total=len(labels), desc=f"Eval {model_name}"):
        # Predict
        pred = model.predict_one(x_i)
        if pred is None:
            pred = 0 # Default if model hasn't learned enough
            
        y_pred.append(pred)
        y_true.append(y_i)
        
        # Train
        model.learn_one(x_i, y_i)
        
    return evaluate_model(y_true, y_pred)

def train_and_evaluate_online():
    """Train and evaluate River online models."""
    X_train, y_train, X_test, y_test = load_data()
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    models = {
        "HoeffdingTree": tree.HoeffdingTreeClassifier(),
        "AdaptiveRandomForest": ensemble.AdaptiveRandomForestClassifier(n_models=10, seed=42)
    }
    
    results = {}
    
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("cicids2017-concept-drift")
    
    features_train = X_train.to_dict(orient="records")
    labels_train = y_train.tolist()
    
    for name, model in models.items():
        with mlflow.start_run(run_name=f"online_{name}"):
            print(f"\nTraining {name} on Train Split...")
            # First, pre-train on the training split
            for x_i, y_i in tqdm(zip(features_train, labels_train), total=len(labels_train), desc=f"Train {name}"):
                model.learn_one(x_i, y_i)
                
            # Then evaluate on test split using prequential
            model_metrics = prequential_evaluation(model, X_test, y_test, name)
            results[name] = model_metrics
            
            # Log to MLflow
            mlflow.log_params({"model_type": name, "static": False})
            mlflow.log_metrics(model_metrics)
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path)
            
            print(f"{name} Metrics: Accuracy={model_metrics['accuracy']:.4f}, F1={model_metrics['f1']:.4f}")
            
    # Save results locally
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(METRICS_DIR, "online_models_metrics.csv"))
    
    return results

if __name__ == "__main__":
    train_and_evaluate_online()

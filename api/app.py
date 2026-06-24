import sys
import os
import json
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Adaptive Network IDS API", description="API for Network Intrusion Detection Predictions")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "metrics")

class FlowFeatures(BaseModel):
    features: List[float]

def load_best_model():
    # Try to load ARF as default best online model, or fallback to Random Forest
    try:
        path = os.path.join(MODELS_DIR, "AdaptiveRandomForest.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f), "online"
                
        path = os.path.join(MODELS_DIR, "RandomForest.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f), "static"
    except Exception as e:
        print(f"Failed to load model: {e}")
        
    return None, None

@app.post("/predict")
def predict_flow(request: FlowFeatures):
    """Predict if a network flow is benign or an attack."""
    model, model_type = load_best_model()
    
    if not model:
        raise HTTPException(status_code=500, detail="No trained models found. Run experiment pipeline first.")
        
    try:
        if model_type == "online":
            # River model
            # Convert list of features to dict mapping
            feat_dict = {f"Feature_{i}": v for i, v in enumerate(request.features)}
            pred = model.predict_one(feat_dict)
            if pred is None:
                pred = 0
            
            # Optionally update the model in real-time if we knew true label, but here we just predict
            return {"prediction": int(pred), "model_type": "River AdaptiveRandomForest"}
        else:
            # Scikit-learn model
            pred = model.predict([request.features])[0]
            confidence = 1.0
            if hasattr(model, "predict_proba"):
                confidence = float(max(model.predict_proba([request.features])[0]))
            return {"prediction": int(pred), "confidence": confidence, "model_type": "Scikit-Learn RandomForest"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """Retrieve the latest experiment metrics."""
    metrics_path = os.path.join(METRICS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return {"message": "Metrics not found. Run experiment pipeline first."}
        
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift")
def get_drift_events():
    """Returns detected drift events."""
    drift_path = os.path.join(METRICS_DIR, "drift_points.csv")
    if not os.path.exists(drift_path):
        return {"message": "Drift data not found."}
        
    try:
        import pandas as pd
        df = pd.read_csv(drift_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

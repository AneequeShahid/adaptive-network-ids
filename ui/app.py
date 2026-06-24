import streamlit as st
import pandas as pd
import json
import os
import requests
from PIL import Image

METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "metrics")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Adaptive Network IDS Dashboard", layout="wide", page_icon="🛡️")

st.title("🛡️ Adaptive Network IDS Research Dashboard")
st.markdown("""
This dashboard visualizes the findings of the **Adaptive Network IDS** research project.
It compares static baseline models against online learning models (River ML) under simulated concept drift conditions using the CICIDS2017 dataset.
""")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🏆 Model Comparison", "⚠️ Drift Analysis", "⚡ Live Prediction"])

with tab1:
    st.header("Dataset Overview (CICIDS2017)")
    st.markdown("""
    The dataset was split temporally to simulate concept drift:
    - **Train Split**: Monday - Wednesday (Benign + Standard Attacks)
    - **Test Split**: Thursday - Friday (Introduction of Web Attacks, Infiltration, DDoS)
    """)
    st.info("Run the '01_eda.ipynb' notebook for deep dive visualizations on class imbalance.")

with tab2:
    st.header("Static vs Online Model Comparison")
    
    metrics_path = os.path.join(METRICS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        df = pd.DataFrame.from_dict(metrics, orient='index')
        st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))
    else:
        st.warning("Metrics file not found. Please run the experiment pipeline.")
        
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accuracy Comparison")
        acc_path = os.path.join(FIGURES_DIR, "accuracy_comparison.png")
        if os.path.exists(acc_path):
            st.image(Image.open(acc_path), use_column_width=True)
            
    with col2:
        st.subheader("F1 Score Comparison")
        f1_path = os.path.join(FIGURES_DIR, "f1_comparison.png")
        if os.path.exists(f1_path):
            st.image(Image.open(f1_path), use_column_width=True)
            
    st.subheader("Feature Importance (Random Forest)")
    feat_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    if os.path.exists(feat_path):
        st.image(Image.open(feat_path), use_column_width=True)

with tab3:
    st.header("Concept Drift Detection")
    st.markdown("This timeline shows where explicit drift detectors (ADWIN, DDM) triggered alerts as error rates spiked.")
    
    drift_img = os.path.join(FIGURES_DIR, "drift_detection.png")
    if os.path.exists(drift_img):
        st.image(Image.open(drift_img), use_column_width=True)
    else:
        st.warning("Drift plot not found. Run the experiment pipeline.")

with tab4:
    st.header("Live Network Flow Prediction")
    st.write("Submit normalized features of a network flow to the FastAPI backend for prediction.")
    
    # Generate 20 dummy inputs
    inputs = []
    cols = st.columns(4)
    for i in range(20):
        with cols[i % 4]:
            inputs.append(st.number_input(f"Feature_{i}", value=0.0, step=0.1, format="%.2f"))
            
    if st.button("Predict Flow Type"):
        with st.spinner("Calling API..."):
            try:
                response = requests.post(f"{API_URL}/predict", json={"features": inputs})
                if response.status_code == 200:
                    data = response.json()
                    pred = data.get("prediction")
                    
                    if pred == 0:
                        st.success(f"**Prediction: BENIGN** (Model: {data.get('model_type')})")
                    else:
                        st.error(f"**Prediction: ATTACK** (Model: {data.get('model_type')})")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Could not reach API at {API_URL}. Is it running? Error: {e}")

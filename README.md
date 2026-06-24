# Adaptive Network IDS: Addressing Concept Drift with Online Learning

This repository contains the codebase for a research paper submission to **IEEE Access**. It presents a research-grade Network Intrusion Detection System (NIDS) designed to solve the concept drift problem in ML-based intrusion detection using online learning algorithms.

## 🔬 Research Question & Hypothesis
**Research Question:** Can online learning models maintain higher detection accuracy than static models when network traffic patterns shift over time (concept drift)?

**Hypothesis:** Online learning models (River ML) will degrade less than static models (scikit-learn) when evaluated on temporally split CICIDS2017 data, simulating real-world concept drift.

## 🏗 Architecture
```text
  CICIDS2017 ──▶ Preprocessing ──▶ Temporal Split
                                         │
                                         ▼
                         Static Models (RF, SVM, LR)
                                         │
                                         ▼
                     Online Models (HT, ARF) + Drift Detection
                                         │
                                         ▼
                          Comparison ──▶ Paper Results
```

## 📊 Dataset Setup (CICIDS2017)
1. Go to kaggle.com and search "CICIDS2017".
2. Download the **UNB CICIDS 2017** dataset.
3. Extract all CSV files into `data/raw/`.
   *Required files:*
   - Monday-WorkingHours.pcap_ISCX.csv
   - Tuesday-WorkingHours.pcap_ISCX.csv  
   - Wednesday-WorkingHours.pcap_ISCX.csv
   - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   - Friday-WorkingHours-Morning.pcap_ISCX.csv
   - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
4. Run the experiment pipeline to automatically preprocess the data.

## 🚀 Installation & Usage
1. Clone the repository and navigate to the directory:
```bash
git clone https://github.com/AneequeShahid/adaptive-network-ids.git
cd adaptive-network-ids
```
2. Set up the virtual environment and install dependencies:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
# source venv/bin/activate
pip install -r requirements.txt
```
3. Run the full experiment pipeline (ensure data is in `data/raw/`):
```bash
python main.py
```
4. Start the Streamlit Dashboard:
```bash
streamlit run ui/app.py
```
5. Start the FastAPI Prediction Server:
```bash
uvicorn api.app:app --reload
```
6. View MLflow Experiment Tracking:
```bash
mlflow ui
```

## 📈 Results Summary
| Model | Type | Accuracy | F1 Score | Adaptability |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | Static | *TBD* | *TBD* | *TBD* |
| Random Forest | Static | *TBD* | *TBD* | *TBD* |
| SVM | Static | *TBD* | *TBD* | *TBD* |
| Hoeffding Tree | Online | *TBD* | *TBD* | *TBD* |
| Adaptive Random Forest | Online | *TBD* | *TBD* | *TBD* |

## 📝 Citation
*(Placeholder for IEEE Access citation once published)*
```bibtex
@article{shahid2026adaptive,
  title={Adaptive Network Intrusion Detection Systems using Online Learning to Mitigate Concept Drift},
  author={Shahid, Aneeque},
  journal={IEEE Access},
  year={2026}
}
```

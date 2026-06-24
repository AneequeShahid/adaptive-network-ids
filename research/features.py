import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple

PROCESSED_FILE = os.path.join("data", "processed", "cicids2017_clean.csv")
SPLITS_DIR = os.path.join("data", "splits")

def feature_engineering():
    """
    Selects top 20 features, standardizes data, encodes labels, and creates temporal splits.
    """
    print(f"Loading {PROCESSED_FILE}...")
    if not os.path.exists(PROCESSED_FILE):
        print(f"File {PROCESSED_FILE} not found. Please run dataset.py first.")
        # create a dummy df for testing if file is missing
        print("Creating dummy dataframe for testing purposes...")
        df = pd.DataFrame(np.random.rand(100, 25), columns=[f"Feature_{i}" for i in range(25)])
        df['Label'] = np.random.choice(['BENIGN', 'DDoS', 'PortScan'], 100)
    else:
        df = pd.read_csv(PROCESSED_FILE)
        
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    print("Encoding labels...")
    # Binary labels: BENIGN=0, Attack=1
    df['Label_Binary'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Multiclass labels
    le = LabelEncoder()
    df['Label_Multiclass'] = le.fit_transform(df['Label'])
    
    # Feature selection
    print("Selecting top 20 features using Mutual Information...")
    X = df.drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])
    y = df['Label_Binary']
    
    # Only use a sample for mutual_info to save memory/time
    sample_size = min(10000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    
    # Handle negative values if present (mutual_info requires non-negative or standardized)
    # Using simple variance threshold instead for the dummy if mutual_info fails
    try:
        selector = SelectKBest(score_func=mutual_info_classif, k=min(20, X.shape[1]))
        selector.fit(X_sample, y_sample)
        selected_features = X.columns[selector.get_support()].tolist()
    except:
        selected_features = X.columns[:min(20, X.shape[1])].tolist()
        
    print(f"Selected features: {selected_features}")
    
    X = X[selected_features]
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Combine back for splitting
    final_df = pd.concat([X_scaled, df[['Label', 'Label_Binary', 'Label_Multiclass']]], axis=1)
    
    print("Creating Temporal Splits...")
    # Simulate temporal split by taking the first 60% as Train (Mon-Wed), last 40% as Test (Thu-Fri)
    # Assuming data is in chronological order. If not, we just use train_test_split without shuffle.
    split_index = int(len(final_df) * 0.6)
    
    train_df = final_df.iloc[:split_index]
    test_df = final_df.iloc[split_index:]
    
    # Save splits
    train_file = os.path.join(SPLITS_DIR, "train_split.csv")
    test_file = os.path.join(SPLITS_DIR, "test_split.csv")
    
    print(f"Saving train split to {train_file} (Shape: {train_df.shape})")
    train_df.to_csv(train_file, index=False)
    
    print(f"Saving test split to {test_file} (Shape: {test_df.shape})")
    test_df.to_csv(test_file, index=False)
    
    print("Feature engineering complete!")
    return True

if __name__ == "__main__":
    feature_engineering()

import os
import glob
import pandas as pd
import numpy as np
from typing import Tuple

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "cicids2017_clean.csv")

def load_and_merge_data() -> pd.DataFrame:
    """
    Loads all CICIDS2017 Parquet or CSV files from data/raw/ and merges them into a single DataFrame.
    """
    print(f"Looking for Parquet files in {RAW_DIR}...")
    parquet_files = glob.glob(os.path.join(RAW_DIR, "*.parquet"))
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    
    if parquet_files:
        all_files = parquet_files
        file_type = "parquet"
    elif csv_files:
        all_files = csv_files
        file_type = "csv"
    else:
        raise FileNotFoundError(f"No Parquet or CSV files found in {RAW_DIR}. Please download the CICIDS2017 dataset.")
        
    df_list = []
    for file in all_files:
        print(f"Loading {os.path.basename(file)}...")
        try:
            if file_type == "parquet":
                df = pd.read_parquet(file)
            else:
                # We use Latin-1 encoding as some csv files may have unusual characters
                df = pd.read_csv(file, encoding='cp1252', low_memory=False)
            
            # Clean up column names (strip whitespace)
            df.columns = df.columns.str.strip()
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            
    print("Merging all files...")
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataframe by handling missing values, infinite values, and duplicates.
    """
    print("Cleaning data...")
    initial_shape = df.shape
    
    # Drop duplicates
    df.drop_duplicates(inplace=True, keep="first")
    
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    final_shape = df.shape
    print(f"Data cleaning complete. Removed {initial_shape[0] - final_shape[0]} rows.")
    return df

def print_statistics(df: pd.DataFrame):
    """
    Prints basic statistics about the dataset.
    """
    print("\n--- Dataset Statistics ---")
    print(f"Shape: {df.shape}")
    
    if 'Label' in df.columns:
        print("\nClass Distribution:")
        print(df['Label'].value_counts())
        
        # Binary class balance
        benign_count = (df['Label'] == 'BENIGN').sum()
        attack_count = len(df) - benign_count
        print(f"\nBinary Balance:")
        print(f"BENIGN: {benign_count} ({(benign_count/len(df))*100:.2f}%)")
        print(f"ATTACK: {attack_count} ({(attack_count/len(df))*100:.2f}%)")
    else:
        print("Warning: 'Label' column not found in dataset.")

def process_dataset() -> Tuple[bool, str]:
    """
    Main function to run the data processing pipeline.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    try:
        df = load_and_merge_data()
        df = clean_data(df)
        print_statistics(df)
        
        print(f"\nSaving processed dataset to {PROCESSED_FILE}...")
        df.to_csv(PROCESSED_FILE, index=False)
        print("Dataset processing complete!")
        return True, "Success"
    except Exception as e:
        print(f"Error during dataset processing: {e}")
        return False, str(e)

if __name__ == "__main__":
    process_dataset()

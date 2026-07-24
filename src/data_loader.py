import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Define data paths relative to root
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "credit_risk.csv"

def load_and_prep_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_PATH}. Please place your CSV inside data/raw/")
    
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Standard feature schema
    feature_cols = ['Credit_Score', 'Annual_Income', 'Debt_Ratio', 'Age', 'Open_Credit_Lines', 'Late_Payments']
    X = df[feature_cols]
    y = df['Default']
    
    # Keep as DataFrames to preserve column names and raw scale for TreeExplainer
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = load_and_prep_data()
    print("[SUCCESS] Data loaded and preprocessed successfully without stripping feature metadata!")
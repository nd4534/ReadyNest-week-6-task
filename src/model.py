import sys
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier

# Fix path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data_loader import load_and_prep_data

MODEL_PATH = ROOT_DIR / "models" / "credit_risk_rf.joblib"

def train_and_save_model():
    X_train, X_test, y_train, y_test = load_and_prep_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model to the models/ directory
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[SUCCESS] Model saved to {MODEL_PATH}")
    
    return model, X_train, X_test, y_train, y_test
train_model = train_and_save_model

def load_saved_model():
    """Fast load for Streamlit app"""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    else:
        model, *_ = train_and_save_model()
        return model

if __name__ == "__main__":
    train_and_save_model()
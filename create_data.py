import pandas as pd
import numpy as np
from pathlib import Path

# Define output path
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)
csv_path = data_dir / "credit_risk.csv"

# Generate synthetic dataset matching our schema
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'Credit_Score': np.random.randint(300, 850, n_samples),
    'Annual_Income': np.random.randint(20000, 150000, n_samples),
    'Debt_Ratio': np.round(np.random.uniform(0.1, 0.9, n_samples), 2),
    'Age': np.random.randint(18, 70, n_samples),
    'Open_Credit_Lines': np.random.randint(1, 15, n_samples),
    'Late_Payments': np.random.randint(0, 10, n_samples),
    'Default': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
})

df.to_csv(csv_path, index=False)
print(f"[SUCCESS] Synthetic dataset created at {csv_path.resolve()}")
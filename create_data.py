import pandas as pd
import numpy as np
from pathlib import Path

# Define output path
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)
csv_path = data_dir / "credit_risk.csv"

# Set seed for reproducibility
np.random.seed(42)
n_samples = 100000

# 1. Generate realistic input features
credit_scores = np.random.randint(300, 850, n_samples)
annual_incomes = np.random.randint(20000, 150000, n_samples)
debt_ratios = np.round(np.random.uniform(0.1, 0.9, n_samples), 2)
ages = np.random.randint(18, 70, n_samples)
open_lines = np.random.randint(1, 15, n_samples)
late_payments = np.random.randint(0, 10, n_samples)

# 2. Create a realistic Logit Score (High credit/income REDUCES risk; High debt/delinquency INCREASES risk)
# Scale normalized features:
score_norm = (credit_scores - 300) / 550.0       # 0 to 1 (Higher = Better)
income_norm = (annual_incomes - 20000) / 130000.0 # 0 to 1 (Higher = Better)
debt_norm = debt_ratios                           # 0 to 1 (Higher = Worse)
late_norm = late_payments / 10.0                  # 0 to 1 (Higher = Worse)

# Log-odds risk calculation
log_odds = (
    1.5 * debt_norm 
    + 3.0 * late_norm 
    - 2.5 * score_norm 
    - 1.8 * income_norm 
    - 0.2
)

# Convert log-odds to risk probability using Sigmoid
risk_probs = 1 / (1 + np.exp(-log_odds))

# Assign target variable: 1 = Default Risk, 0 = Non-Default
defaults = (np.random.rand(n_samples) < risk_probs).astype(int)

df = pd.DataFrame({
    'Credit_Score': credit_scores,
    'Annual_Income': annual_incomes,
    'Debt_Ratio': debt_ratios,
    'Age': ages,
    'Open_Credit_Lines': open_lines,
    'Late_Payments': late_payments,
    'Default': defaults
})

df.to_csv(csv_path, index=False)
print(f"[SUCCESS] Domain-aligned synthetic dataset created at {csv_path.resolve()}")
print(f"Overall Default Rate: {df['Default'].mean()*100:.1f}%")
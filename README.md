# Explainable AI (XAI) Credit Risk & Batch Analysis Studio

An industry-grade, modular Credit Risk Analysis Dashboard built with Streamlit, Scikit-Learn, SHAP (SHapley Additive exPlanations), and a Local LLM Copilot (Ollama - Qwen 2.5:7B).

This studio bridges quantitative machine learning with local conversational AI to deliver real-time risk scores, local feature attributions, automated underwriter guidance, and portfolio-wide batch evaluation without sending sensitive financial data to cloud endpoints.

---

## Key Features

* Single-Instance What-If Analysis:
  * Real-time credit risk scoring based on interactive profile adjustments (Credit Score, Income, Debt Ratio, Delinquencies, etc.).
  * Color-coded underwriting decision tiering (LOW RISK, MANUAL REVIEW, HIGH RISK).
  * Local SHAP force/impact visual bar charts showing top risk drivers and mitigating factors.
  * Counterfactual guidance offering actionable "Path to Risk Reduction" steps for applicants.
  * Dedicated single-applicant Ollama Copilot for automated memo drafting and custom underwriter queries.

* Mass Data Batch Evaluation:
  * CSV upload support with built-in auto-mapping for Kaggle/LendingClub dataset schemas.
  * High-level portfolio metrics (Average Portfolio Risk, High/Low-Risk Counts).
  * Interactive Plotly distributions and feature scatter plots.
  * Auto-polling capability (60s live check) for batch monitoring workflows.
  * Portfolio-wide local LLM Assistant equipped with structured snapshot metrics and strict anti-duplication reasoning prompts.

---

## Tech Stack

* Frontend: Streamlit
* Machine Learning: Scikit-Learn (Random Forest Classifier)
* Explainability (XAI): SHAP, Plotly
* Local LLM Inference: Ollama SDK (qwen2.5:7b)
* Persistence & Data Pipelines: Joblib, Pandas, NumPy

---

## Quick Start Guide

### 1. Prerequisites & Environment Setup

Ensure Python 3.9+ and Ollama are installed on your machine.

# Clone or navigate to the project root
cd READYNEST_TASK_6

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### 2. Local LLM Setup (Ollama)

Pull the Qwen 2.5 (7B) model via Ollama:

ollama pull qwen2.5:7b

### 3. Generate Dataset & Train Model

If you don't have raw data in data/raw/credit_risk.csv, run the synthetic dataset creation script:

# Create raw synthetic credit dataset
python create_data.py

# Train and persist the model
python src/model.py

### 4. Launch Application

Start the Streamlit Studio application:

streamlit run app.py

---

## Project Structure

READYNEST_TASK_6/
│
├── data/
│   └── raw/
│       └── credit_risk.csv          # Raw input dataset for training & evaluation
│
├── models/
│   └── credit_risk_rf.joblib        # Persisted Random Forest model file
│
├── src/
│   ├── __pycache__/                 # Compiled Python bytecode cache
│   ├── __init__.py                  # Package marker (kept lightweight)
│   ├── data_loader.py               # Data ingestion & feature preprocessing pipeline
│   ├── explainability.py            # SHAP engine wrapper & attribution calculator
│   └── model.py                     # Random Forest training & model persistence logic
│
├── app.py                           # Main Streamlit Dashboard (Mode 1 & Mode 2 UI/Copilot)
├── create_data.py                   # Helper script to generate mock synthetic credit data
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
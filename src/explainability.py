import pandas as pd
import numpy as np
import shap

class SHAPEngine:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        
        # Ensure base value corresponds to Class 1 (Default Risk)
        raw_base = self.explainer.expected_value
        if isinstance(raw_base, (list, tuple, np.ndarray)):
            base_val = raw_base[1] if len(raw_base) > 1 else raw_base[0]
        else:
            base_val = raw_base
            
        self.base_value = float(base_val)

    def explain_instance(self, input_dict):
        # 1. Convert input dict to DataFrame
        df_input = pd.DataFrame([input_dict])
        
        # 2. Force exact column ordering expected by the model
        if hasattr(self.model, "feature_names_in_"):
            df_input = df_input[list(self.model.feature_names_in_)]
            
        # 3. Extract Default Risk Probability (Class 1)
        probs = self.model.predict_proba(df_input)[0]
        prob_risk = float(probs[1]) if len(probs) > 1 else float(probs[0])
        
        # 4. Compute SHAP Values
        shap_vals = self.explainer(df_input)
        
        # Extract SHAP array for Class 1 (Default Risk)
        if len(shap_vals.values.shape) == 3:
            shap_array = shap_vals.values[0][:, 1]
            base_val = shap_vals.base_values[0][1]
        else:
            shap_array = shap_vals.values[0]
            base_val = shap_vals.base_values[0]
            
        contributions = pd.DataFrame({
            "Feature": df_input.columns,
            "Value": df_input.iloc[0].values,
            "SHAP_Value": shap_array
        }).sort_values(by="SHAP_Value", key=abs, ascending=False)
        
        return prob_risk, float(base_val), contributions
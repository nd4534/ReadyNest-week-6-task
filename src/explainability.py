import pandas as pd
import numpy as np
import shap

class SHAPEngine:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        
        # Extract base value and ensure it's a scalar float
        raw_base = self.explainer.expected_value
        if isinstance(raw_base, (list, tuple, np.ndarray)):
            # Handle multi-class or array outputs
            base_val = raw_base[1] if len(raw_base) > 1 else raw_base[0]
        else:
            base_val = raw_base
            
        self.base_value = float(base_val)

    def explain_instance(self, input_dict):
        df_input = pd.DataFrame([input_dict])
        prob = float(self.model.predict_proba(df_input)[0][1])
        shap_vals = self.explainer(df_input)
        
        # Extract SHAP array values cleanly
        if len(shap_vals.values.shape) == 3:
            shap_array = shap_vals.values[0][:, 1]
        else:
            shap_array = shap_vals.values[0]
        
        contributions = pd.DataFrame({
            "Feature": df_input.columns,
            "Value": df_input.iloc[0].values,
            "SHAP_Value": shap_array
        }).sort_values(by="SHAP_Value", key=abs, ascending=False)
        
        return prob, self.base_value, contributions
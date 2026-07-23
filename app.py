import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import ollama

# Clean module imports with fallback handling for train_model vs train_and_save_model
from src.data_loader import load_and_prep_data
import src.model as model

# Resolve train_model dynamically if aliased or named train_and_save_model
if hasattr(model, "train_model"):
    train_model = model.train_model
elif hasattr(model, "train_and_save_model"):
    train_model = model.train_and_save_model
else:
    raise AttributeError("src.model has neither 'train_model' nor 'train_and_save_model'")

if hasattr(model, "load_saved_model"):
    load_saved_model = model.load_saved_model
else:
    load_saved_model = None

# Ensure SHAPEngine matches your actual file name
try:
    from src.explainability import SHAPEngine
except ModuleNotFoundError:
    try:
        from src.shap_engine import SHAPEngine
    except ModuleNotFoundError:
        SHAPEngine = None  # Fallback check

# Page Config
st.set_page_config(
    page_title="What-If & Batch XAI Studio + Copilot",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Explainable AI Studio + Live Risk Copilot")
st.caption("Batch risk evaluation, interactive XAI assistant, future trend forecasting, and live file monitoring.")

# Initialize Model & Engine
@st.cache_resource
def init_engine():
    model_obj, X_train, X_test, y_train, y_test = train_model()

    # Define standard feature names expected by the model
    feature_names = [
        "Credit_Score",
        "Annual_Income",
        "Debt_Ratio",
        "Age",
        "Open_Credit_Lines",
        "Late_Payments",
    ]

    # Guarantee X_train is a Pandas DataFrame with standard column names
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)
    elif list(X_train.columns) != feature_names:
        X_train.columns = feature_names

    engine = SHAPEngine(model_obj)
    return engine, X_train


engine, X_train = init_engine()

# Select Mode
mode = st.radio(
    "Select Operating Mode:",
    ["👤 Single Instance What-If", "📁 Mass Data Batch Evaluation"],
    horizontal=True
)

st.markdown("---")

# ==========================================
# MODE 1: SINGLE INSTANCE WHAT-IF
# ==========================================
if mode == "👤 Single Instance What-If":
    col_controls, col_display = st.columns([1, 2])

    with col_controls:
        st.subheader("⚙️ Profile Input")
        credit_score = st.slider("Credit Score", 300, 850, int(X_train["Credit_Score"].mean()))
        annual_income = st.slider("Annual Income ($)", 15000, 200000, int(X_train["Annual_Income"].mean()), step=5000)
        debt_ratio = st.slider("Debt Ratio", 0.05, 0.95, float(X_train["Debt_Ratio"].mean()), step=0.01)
        age = st.slider("Age", 18, 75, int(X_train["Age"].mean()))
        open_lines = st.slider("Open Credit Lines", 1, 15, int(X_train["Open_Credit_Lines"].mean()))
        late_payments = st.slider("Late Payments", 0, 10, int(X_train["Late_Payments"].mean()))

        input_data = {
            "Credit_Score": credit_score,
            "Annual_Income": annual_income,
            "Debt_Ratio": debt_ratio,
            "Age": age,
            "Open_Credit_Lines": open_lines,
            "Late_Payments": late_payments
        }

    prob, base_value, contributions = engine.explain_instance(input_data)
    risk_pct = prob * 100

    # Determine Risk Status & Color
    if risk_pct < 30:
        status_label, status_color, status_icon = "LOW RISK - PRE-APPROVED", "green", "✅"
    elif risk_pct < 60:
        status_label, status_color, status_icon = "MEDIUM RISK - MANUAL REVIEW", "orange", "⚠️"
    else:
        status_label, status_color, status_icon = "HIGH RISK - DECLINE RECOMMENDED", "red", "🚨"

    with col_display:
        st.subheader("📊 Underwriting Decision & Attribution")

        # Top Executive Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Risk Score", f"{risk_pct:.1f}%")
        m2.metric("Portfolio Baseline", f"{base_value * 100:.1f}%")
        m3.metric("Decision Status", f"{status_icon} {status_label.split(' - ')[0]}")

        # Risk Banner
        if status_color == "green":
            st.success(f"**Status:** {status_label} | Risk score is **{base_value*100 - risk_pct:.1f}% lower** than portfolio average.")
        elif status_color == "orange":
            st.warning(f"**Status:** {status_label} | Elevated risk requires secondary underwriting review.")
        else:
            st.error(f"**Status:** {status_label} | Exceeds risk threshold for standard approval.")

        # SHAP Horizontal Bar Chart
        colors = ["#EF553B" if x > 0 else "#636EFA" for x in contributions["SHAP_Value"]]
        fig = go.Figure(go.Bar(
            x=contributions["SHAP_Value"],
            y=contributions["Feature"],
            orientation='h',
            marker_color=colors,
            text=[f"{val:+.3f}" for val in contributions["SHAP_Value"]],
            textposition="outside"
        ))
        fig.update_layout(
            title="Feature Impact Breakdown (SHAP Values)",
            xaxis_title="Impact on Risk Probability (+ Increases Risk, - Decreases Risk)",
            yaxis=dict(autorange="reversed"),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Dynamic Automated Insights & Counterfactual Guidance
        st.markdown("### 💡 Automated Insights & Mitigation")
        
        # Sort contributions to extract key drivers
        sorted_contribs = contributions.sort_values(by="SHAP_Value", ascending=False)
        top_risk_driver = sorted_contribs.iloc[0]
        top_mitigator = sorted_contribs.iloc[-1]

        c_ins1, c_ins2 = st.columns(2)
        with c_ins1:
            st.info(
                f"**Primary Risk Driver:** `{top_risk_driver['Feature']}` (+{top_risk_driver['SHAP_Value']:.3f})\n\n"
                f"This factor contributed the most toward pushing the applicant's risk score up."
            )
        with c_ins2:
            st.info(
                f"**Primary Mitigating Factor:** `{top_mitigator['Feature']}` ({top_mitigator['SHAP_Value']:.3f})\n\n"
                f"This strong attribute helped keep the overall risk score suppressed."
            )

        # What-If Improvement Advice (if not already lowest risk)
        if risk_pct >= 30:
            st.markdown("#### 🎯 Path to Risk Reduction (Counterfactual Simulation)")
            advice_list = []
            if debt_ratio > 0.35:
                advice_list.append(f"• **Lower Debt Ratio**: Reducing debt ratio from `{debt_ratio:.2f}` to `<= 0.30` will significantly decrease risk.")
            if late_payments > 0:
                advice_list.append(f"• **Resolve Delinquencies**: Clearing late payments (currently `{late_payments}`) will remove a major risk penalty.")
            if credit_score < 700:
                advice_list.append(f"• **Improve Credit Score**: Raising score from `{credit_score}` to `700+` will shift profile into prime tier.")

            if advice_list:
                for adv in advice_list:
                    st.write(adv)
            else:
                st.write("• Increasing annual income or establishing longer credit history will further improve approval terms.")

    # ---------------------------------------------------------
    # Single-Instance Local XAI Copilot
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🤖 Single Applicant Copilot (Powered by Qwen 2.5:7b)")
    st.write("Ask questions about this specific profile or generate an underwriter summary note:")

    if "single_chat_history" not in st.session_state:
        st.session_state.single_chat_history = []

    # Display history
    for msg in st.session_state.single_chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    single_query = st.chat_input("e.g., Draft an approval memorandum or explain why this profile was flagged...")

    if single_query:
        st.session_state.single_chat_history.append({"role": "user", "content": single_query})
        st.chat_message("user").write(single_query)

        single_system_prompt = f"""
            You are an expert Chief Risk Officer (CRO) and Loan Underwriter AI assistant.
            
            APPLICANT PROFILE:
            - Credit Score: {credit_score}
            - Annual Income: ${annual_income:,}
            - Debt Ratio: {debt_ratio:.2f}
            - Age: {age}
            - Open Credit Lines: {open_lines}
            - Late Payments: {late_payments}
            - Predicted Risk Score: {risk_pct:.1f}% (Baseline: {base_value*100:.1f}%)
            - Top Risk Driver: {top_risk_driver['Feature']}
            - Top Mitigating Factor: {top_mitigator['Feature']}

            STRICT INSTRUCTIONS:
            - Answer questions regarding this specific applicant directly and professionally.
            - Keep responses concise, objective, and actionable.
        """

        messages_for_ollama = [{"role": "system", "content": single_system_prompt}]
        for m in st.session_state.single_chat_history[-6:]:
            messages_for_ollama.append({"role": m["role"], "content": m["content"]})

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                stream = ollama.chat(
                    model="qwen2.5:7b",
                    messages=messages_for_ollama,
                    options={"temperature": 0.3, "num_ctx": 4096},
                    stream=True
                )

                for chunk in stream:
                    full_response += chunk['message']['content']
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"⚠️ Error communicating with local Ollama: {str(e)}"
                response_placeholder.error(full_response)

        st.session_state.single_chat_history.append({"role": "assistant", "content": full_response})

# ==========================================
# MODE 2: MASS DATA BATCH EVALUATION
# ==========================================
else:
    col_upload, col_refresh = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Upload a CSV file containing customer data", type=["csv"])
    
    with col_refresh:
        st.subheader("🔄 Live Refresh")
        enable_polling = st.checkbox("Enable 60s Auto-Poll", value=False)
        if enable_polling:
            st.caption("⚡ Auto-checking for data updates every 60s")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        # Kaggle Auto-Mapper
        kaggle_mapper = {
            "person_age": "Age", "person_income": "Annual_Income",
            "loan_int_rate": "Debt_Ratio", "cb_person_cred_hist_length": "Open_Credit_Lines",
            "cb_person_default_on_file": "Late_Payments", "age": "Age",
            "annual_inc": "Annual_Income", "dti": "Debt_Ratio",
            "open_acc": "Open_Credit_Lines", "delinq_2yrs": "Late_Payments"
        }
        batch_df = batch_df.rename(columns=kaggle_mapper)
        if "Credit_Score" not in batch_df.columns:
            batch_df["Credit_Score"] = 680

        required_cols = ["Credit_Score", "Annual_Income", "Debt_Ratio", "Age", "Open_Credit_Lines", "Late_Payments"]
        
        if not all(col in batch_df.columns for col in required_cols):
            st.error(f"Uploaded CSV must contain required columns: {required_cols}")
        else:
            # Batch Inference
            probs = engine.model.predict_proba(batch_df[required_cols])[:, 1]
            batch_df["Predicted_Risk_Score"] = np.round(probs * 100, 2)
            batch_df["Risk_Category"] = pd.cut(
                batch_df["Predicted_Risk_Score"], 
                bins=[-1, 25, 50, 100], 
                labels=["Low Risk", "Medium Risk", "High Risk"]
            )

            # High-Level Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records Evaluated", len(batch_df))
            col2.metric("Average Portfolio Risk", f"{batch_df['Predicted_Risk_Score'].mean():.1f}%")
            col3.metric("High-Risk Applicants", f"{(batch_df['Risk_Category'] == 'High Risk').sum()}")
            col4.metric("Low-Risk Applicants", f"{(batch_df['Risk_Category'] == 'Low Risk').sum()}")

            st.markdown("---")
            
            c1, c2 = st.columns(2)
            with c1:
                fig_dist = px.histogram(
                    batch_df, x="Predicted_Risk_Score", color="Risk_Category",
                    title="Portfolio Risk Score Distribution",
                    color_discrete_map={"Low Risk": "#636EFA", "Medium Risk": "#FECB52", "High Risk": "#EF553B"}
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with c2:
                fig_scatter = px.scatter(
                    batch_df, x="Credit_Score", y="Annual_Income", color="Risk_Category", size="Late_Payments",
                    title="Credit Score vs Income by Risk Category",
                    color_discrete_map={"Low Risk": "#636EFA", "Medium Risk": "#FECB52", "High Risk": "#EF553B"}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # 🔮 FUTURE PREDICTIONS & PROJECTIONS MODULE
            st.markdown("---")
            st.subheader("🤖 Local XAI Copilot (Powered by Qwen 2.5:7b)")
            st.write("Ask complex underwriting, policy, or risk-mitigation questions:")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [
                    {
                        "role": "assistant",
                        "content": f"Hello! I am running locally using **Qwen 2.5 (7B)**. I have analyzed your dataset of **{len(batch_df):,}** records with an average risk of **{batch_df['Predicted_Risk_Score'].mean():.1f}%**. How can I assist with your risk management strategy?"
                    }
                ]

            # Display conversation history
            for msg in st.session_state.chat_history:
                st.chat_message(msg["role"]).write(msg["content"])

            user_query = st.chat_input("Ask a quantitative risk question...")

            if user_query:
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.chat_message("user").write(user_query)

                # 1. Detect if the user is asking for alternatives / follow-up strategies
                repeat_keywords = ["more", "another", "other", "else", "different", "alternative", "again"]
                is_asking_more = any(kw in user_query.lower() for kw in repeat_keywords)

                data_keywords = [
                    "metric", "batch", "risk", "segment", "credit", "dti", "debt", 
                    "late", "near-prime", "prime", "count", "income", "portfolio", "rate", "strategy", "reduce"
                ]
                needs_data_context = any(kw in user_query.lower() for kw in data_keywords)

                # 2. Build Context-Aware System Prompt
                if needs_data_context:
                    total_count = len(batch_df)
                    high_risk_df = batch_df[batch_df['Risk_Category'] == 'High Risk']
                    low_risk_df = batch_df[batch_df['Risk_Category'] == 'Low Risk']

                    near_prime_mask = (
                        (batch_df['Credit_Score'] >= 620) & 
                        (batch_df['Credit_Score'] <= 680) & 
                        (batch_df['Debt_Ratio'] <= 0.40) & 
                        (batch_df['Late_Payments'] <= 1)
                    )
                    near_prime_df = batch_df[near_prime_mask]

                    system_prompt = f"""
                        You are an expert Chief Risk Officer (CRO) AI assistant.
                        
                        PORTFOLIO SNAPSHOT:
                        - Total Applicants: {total_count:,}
                        - High-Risk Ratio: {(len(high_risk_df)/total_count)*100:.1f}% (Avg DTI: {high_risk_df['Debt_Ratio'].mean():.2f}, Avg Late Payments: {high_risk_df['Late_Payments'].mean():.1f})
                        - Near-Prime Candidates: {len(near_prime_df):,} applicants

                        STRICT OUTPUT RULES (CRITICAL):
                        1. NEVER repeat or restructure information within the same response (e.g., do NOT list items 1-10 and then re-list them as Short/Medium/Long term).
                        2. Pick ONLY the top 3 most high-impact recommendations for this specific dataset and list them ONCE.
                        3. Be direct, concise, and punchy. Cut out fluff, intro warm-ups, and duplicative summaries.
                        """
                else:
                    system_prompt = """
                    You are an expert Chief Risk Officer (CRO) AI assistant for a Credit Risk Analysis Dashboard.
                    Be helpful, professional, and conversational.
                    """

                # 3. Include broader conversation history so the model knows what it already said
                recent_history = st.session_state.chat_history[-8:]
                messages_for_ollama = [{"role": "system", "content": system_prompt}]
                for m in recent_history:
                    messages_for_ollama.append({"role": m["role"], "content": m["content"]})

                # 4. Stream response with dynamic temperature
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""

                    try:
                        options = {
                            "num_ctx": 4096,
                            "temperature": 0.6 if is_asking_more else 0.3,
                            "top_p": 0.9
                        }

                        stream = ollama.chat(
                            model="qwen2.5:7b",
                            messages=messages_for_ollama,
                            options=options,
                            stream=True
                        )

                        for chunk in stream:
                            full_response += chunk['message']['content']
                            response_placeholder.markdown(full_response + "▌")

                        response_placeholder.markdown(full_response)

                    except Exception as e:
                        full_response = f"⚠️ Error communicating with local Ollama: {str(e)}"
                        response_placeholder.error(full_response)

                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            # Auto-polling rerun loop
            if enable_polling:
                time.sleep(60)
                st.rerun()
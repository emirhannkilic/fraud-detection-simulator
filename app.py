import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from sklearn.preprocessing import StandardScaler
from streamlit_shap import st_shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Fraud Detection Simulator",
    page_icon="",
    layout="wide"
)

st.markdown("<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True)

st.title("Fraud Detection Simulator")
st.caption("XGBoost pipeline with SHAP explainability and business impact simulation.")

@st.cache_resource
def load_model():
    with open('models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('data/creditcard.csv')
    return df

model, scaler = load_model()
df = load_data()

# Section 1: Model Summary
st.header("Model Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", "284,807")
col2.metric("Fraud Rate", "0.17%")
col3.metric("ROC-AUC", "0.97")
col4.metric("F1 Score", "0.86")

st.divider()
# Section 2: Threshold Simulator
st.header("Threshold Simulator")

with st.container(border=True):
    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.01)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test_scaled = scaler.transform(X_test)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

avg_fraud_amount = df[df['Class'] == 1]['Amount'].mean()
TP_VALUE = avg_fraud_amount
FP_COST = 10
FN_COST = avg_fraud_amount

y_pred = (y_proba >= threshold).astype(int)
TP = sum((y_pred == 1) & (y_test == 1))
FP = sum((y_pred == 1) & (y_test == 0))
FN = sum((y_pred == 0) & (y_test == 1))

saved_money = TP * TP_VALUE
lost_money = FN * FN_COST
profit = saved_money - (FP * FP_COST) - lost_money

col_left, col_right = st.columns([3, 2])

@st.cache_data
def compute_shap_values(_model, _X_test_scaled):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(_X_test_scaled)
    return shap_values, explainer.expected_value

shap_values, expected_value = compute_shap_values(model, X_test_scaled)

with col_left:
    st.write("### Financial Impact")
    f1, f2 = st.columns(2)
    f1.metric("Money Saved", f"${saved_money:,.0f}")
    f2.metric("Net Profit", f"${profit:,.0f}")
    
    st.write("### Model Performance")
    p1, p2 = st.columns(2)
    p1.metric("Fraud Caught", f"{TP} / {sum(y_test)}")
    p2.metric("Legitimate Blocked", FP)

    with st.expander("SHAP Feature Importance", expanded=False):
        st_shap(shap.summary_plot(shap_values, X_test, feature_names=X.columns.tolist(), show=False), height=400)

with col_right:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar=False,
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    st.pyplot(fig)
    plt.close('all')

st.divider()
# Section 3: Scenario Comparison
st.header("Scenario Comparison")

scenarios = {
    'Aggressive (0.3)': 0.3,
    'Balanced (0.5)': 0.5,
    'Conservative (0.7)': 0.7
}

scenario_data = []
for name, t in scenarios.items():
    y_pred_s = (y_proba >= t).astype(int)
    TP_s = sum((y_pred_s == 1) & (y_test == 1))
    FP_s = sum((y_pred_s == 1) & (y_test == 0))
    FN_s = sum((y_pred_s == 0) & (y_test == 1))
    saved = TP_s * TP_VALUE
    lost = FN_s * FN_COST
    profit_s = saved - (FP_s * FP_COST) - lost
    recall_s = TP_s / sum(y_test)
    
    scenario_data.append({
        'Scenario': name,
        'Fraud Caught': f"{TP_s}/{sum(y_test)}",
        'Recall': f"{recall_s:.2f}",
        'Blocked Legit': FP_s,
        'Money Saved': f"${saved:,.0f}",
        'Net Profit': f"${profit_s:,.0f}"
    })

st.dataframe(pd.DataFrame(scenario_data), use_container_width=True)
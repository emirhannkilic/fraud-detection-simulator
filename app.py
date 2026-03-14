import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Fraud Detection Simulator",
    page_icon="",
    layout="wide"
)

st.title("Fraud Detection + Business Impact Simulator")
st.markdown("XGBoost-based fraud detection pipeline with SHAP explainability and business impact simulation.")

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

st.sidebar.title("Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.01)

# Section 1: Model Summary
st.header("Model Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", "284,807")
    st.metric("Fraud Rate", "0.17%")

with col2:
    st.metric("Model", "XGBoost")
    st.metric("ROC-AUC", "0.97")

with col3:
    st.metric("Precision", "0.88")
    st.metric("Recall", "0.84")

# Section 2: Threshold Simulator
st.header("Threshold Simulator")

X = df.drop('Class', axis=1)
y = df['Class']

from sklearn.model_selection import train_test_split
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

col1, col2, col3, col4 = st.columns(4)
col1.metric("Fraud Caught", f"{TP} / {sum(y_test)}")
col2.metric("Legitimate Blocked", FP)
col3.metric("Money Saved", f"${saved_money:,.0f}")
col4.metric("Net Profit", f"${profit:,.0f}")

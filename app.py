# ==============================================================================
# PROJECT: Interactive Customer Churn Prediction App
# AUTHOR: Pranay
# FILE NAME: app.py
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# Application Header
st.title("📊 Customer Churn Prediction & Risk Analysis")
st.markdown("Developed by **Pranay** | AICTE Data Analytics Internship")
st.markdown("---")

# Load and train model on simulated RFM dataset
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_customers = 1000
    
    data = {
        'Recency': np.random.randint(1, 365, size=n_customers),
        'Frequency': np.random.randint(1, 50, size=n_customers),
        'Monetary': np.random.uniform(10, 5000, size=n_customers)
    }
    
    df = pd.DataFrame(data)
    # Target definition: High Recency (>180) & Low Frequency (<10)
    df['Churn'] = np.where((df['Recency'] > 180) & (df['Frequency'] < 10), 1, 0)
    
    X = df[['Recency', 'Frequency', 'Monetary']]
    y = df['Churn']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

model = train_model()

# Sidebar / Main Form for User Inputs
st.header("🔍 Input Customer RFM Metrics")

col1, col2 = st.columns(2)

with col1:
    recency = st.number_input(
        "Recency (Days since last purchase)", 
        min_value=1, 
        max_value=365, 
        value=120,
        help="How many days ago did the customer place their last order?"
    )
    
    frequency = st.number_input(
        "Frequency (Total number of orders)", 
        min_value=1, 
        max_value=100, 
        value=5,
        help="How many times has this customer ordered in total?"
    )

with col2:
    monetary = st.number_input(
        "Monetary Value ($ Total spend)", 
        min_value=1.0, 
        max_value=10000.0, 
        value=250.00,
        step=10.0,
        help="Total monetary amount spent by this customer."
    )

st.markdown("---")

# Make Prediction
if st.button("Predict Churn Risk", type="primary"):
    user_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary]
    })
    
    prediction = model.predict(user_data)[0]
    probabilities = model.predict_proba(user_data)[0]
    
    churn_prob = probabilities[1] * 100
    
    st.subheader("🎯 Prediction Result")
    
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk!**")
        st.write(f"Probability of Churn: **{churn_prob:.1f}%**")
        st.warning("👉 **Action:** Trigger retention email with exclusive discount code.")
    else:
        st.success(f"✅ **Low Churn Risk / Active Customer**")
        st.write(f"Probability of Retention: **{(100 - churn_prob):.1f}%**")
        st.info("👉 **Action:** Include in standard engagement campaigns or loyalty rewards.")

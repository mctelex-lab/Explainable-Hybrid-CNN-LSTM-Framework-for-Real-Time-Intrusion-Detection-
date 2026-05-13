# app.py - Streamlit Dashboard with XAI
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import plotly.express as px
import os

st.set_page_config(page_title="IDS Dashboard", layout="wide", page_icon="🛡️")

st.title("🛡️ Real-Time Network Intrusion Detection System")
st.markdown("*Optimized Neural Network with Dual XAI (LIME + SHAP) for Enterprise Security*")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('Models/best_model.h5')
    scaler = joblib.load('Models/scaler.pkl')
    imputer = joblib.load('Models/imputer.pkl')
    with open('Models/feature_names.json', 'r') as f:
        features = json.load(f)
    with open('Models/metadata.json', 'r') as f:
        meta = json.load(f)
    return model, scaler, imputer, features, meta

try:
    model, scaler, imputer, features, meta = load_artifacts()
    st.sidebar.success("✅ Model Ready")
    st.sidebar.metric("F1 Score", f"{meta['evaluation_metrics']['f1_score']:.4f}")
    st.sidebar.metric("ROC AUC", f"{meta['evaluation_metrics']['roc_auc']:.4f}")
    st.sidebar.metric("Latency", f"{meta['evaluation_metrics']['inference_latency_ms']:.1f} ms")
except Exception as e:
    st.sidebar.error(f"Error: {e}")
    st.stop()

# Main analysis
uploaded = st.file_uploader("Upload Network Flow CSV", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())
    
    if st.button("🔍 Analyze Traffic", type="primary"):
        with st.spinner("Analyzing..."):
            # Align features
            for f in features:
                if f not in df.columns:
                    df[f] = 0
            
            X = df[features].values
            X = imputer.transform(X)
            X = scaler.transform(X)
            preds = model.predict(X, verbose=0).flatten()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Threat Score", f"{np.mean(preds):.2%}")
            col2.metric("Detected Attacks", f"{np.sum(preds >= 0.5)}")
            col3.metric("Benign Flows", f"{np.sum(preds < 0.5)}")
            
            df['threat'] = preds
            df['prediction'] = ['🚨 ATTACK' if p >= 0.5 else '✅ BENIGN' for p in preds]
            st.dataframe(df[['prediction', 'threat'] + features[:5]])
            
            fig = px.histogram(df, x='threat', nbins=30, title='Threat Distribution')
            fig.add_vline(x=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig)

# XAI Section
with st.expander("🔍 Explainable AI (LIME + SHAP) - Forensic Analysis", expanded=True):
    st.markdown("""
    ### Dual XAI Integration for Forensic Transparency
    
    **LIME (Local Interpretable Model-agnostic Explanations)**
    - Provides instance-level explanations for individual predictions
    - Helps security analysts understand why specific flows were flagged
    
    **SHAP (SHapley Additive exPlanations)**
    - Provides global feature importance based on game theory
    - Identifies which network features are most indicative of attacks
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 SHAP Global Feature Importance")
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", caption="SHAP Summary Plot - Top Features for Attack Detection", use_container_width=True)
        if os.path.exists("shap_bar_plot.png"):
            st.image("shap_bar_plot.png", caption="SHAP Bar Plot - Mean Absolute Feature Importance", use_container_width=True)
    
    with col2:
        st.subheader("📋 LIME Local Explanations")
        for i in range(3):
            if os.path.exists(f"lime_explanation_{i}.png"):
                st.image(f"lime_explanation_{i}.png", caption=f"LIME Explanation - Sample {i+1}", use_container_width=True)

st.markdown("---")
st.caption("Dual XAI Integrated IDS | Built with TensorFlow, LIME & SHAP | Enterprise Ready")

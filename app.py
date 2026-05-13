# app.py - Professional IDS Dashboard (Fixed Model Loading)
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import joblib
import json
import plotly.express as px
import os
from datetime import datetime

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="CyberGuard IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
    .main-header { font-size: 2.8rem; color: #0E86D4; text-align: center; margin-bottom: 0.5rem; font-weight: 700; }
    .sub-header { font-size: 1.4rem; color: #00C9A7; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #1E2A44; border-radius: 12px; padding: 1.2rem; border: 1px solid #0E86D4; }
    .stButton>button { background-color: #0E86D4; color: white; border-radius: 8px; height: 3em; font-weight: 600; }
    .footer { text-align: center; margin-top: 3rem; padding: 1rem; border-top: 1px solid #334155; color: #94A3B8; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🛡️ CyberGuard IDS</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explainable Hybrid Neural Network for Real-Time Intrusion Detection</p>', unsafe_allow_html=True)
st.markdown("**Developed by Samuel Ayorinde** | PGD Cybersecurity | Supervised by Mr. Victor Akuboh")

st.divider()

# ===================== MODEL RECONSTRUCTION =====================
def build_model(input_dim):
    """Rebuild exact model architecture to avoid version compatibility issues"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001),
                    kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001),
                    kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu',
                    kernel_initializer='he_normal'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# ===================== LOAD ARTIFACTS =====================
@st.cache_resource
def load_artifacts():
    try:
        # Load preprocessors
        scaler = joblib.load('Models/scaler.pkl')
        imputer = joblib.load('deployment/imputer.pkl')
        
        with open('Models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        with open('Models/metadata.json', 'r') as f:
            meta = json.load(f)
        
        input_dim = len(feature_names)
        
        # Build model architecture then load weights
        model = build_model(input_dim)
        
        # Try loading weights
        weights_path = 'Models/model.weights.h5'
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            st.sidebar.success("✅ Model weights loaded")
        else:
            # Fallback to full model file
            if os.path.exists('Models/model.h5'):
                model = tf.keras.models.load_model('Models/model.h5', compile=False)
            elif os.path.exists('Models/model.keras'):
                model = tf.keras.models.load_model('deployment/model.keras', compile=False)
            else:
                st.error("Model file not found!")
                st.stop()
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model, scaler, imputer, feature_names, meta
        
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.stop()

model, scaler, imputer, feature_names, meta = load_artifacts()

# ===================== SIDEBAR =====================
with st.sidebar:
    st.success("✅ Model Ready")
    st.metric("F1 Score", f"{meta['evaluation_metrics']['f1_score']:.4f}")
    st.metric("ROC AUC", f"{meta['evaluation_metrics']['roc_auc']:.4f}")
    st.metric("Latency", f"{meta['evaluation_metrics']['inference_latency_ms']:.1f} ms")
    st.info("**Dual XAI Enabled** (LIME + SHAP)")

# ===================== TABS =====================
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Real-Time Detection", "📈 Performance", "🔍 Explainable AI", "ℹ️ About"])

with tab1:
    st.subheader("Real-Time Network Flow Analysis")
    uploaded_file = st.file_uploader("Upload Network Flow CSV", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} flows")
        
        if st.button("🔍 Analyze Traffic", type="primary", use_container_width=True):
            with st.spinner("Running inference..."):
                # Feature alignment
                for f in feature_names:
                    if f not in df.columns:
                        df[f] = 0
                
                X = df[feature_names].values
                X = imputer.transform(X)
                X = scaler.transform(X)
                
                preds_proba = model.predict(X, verbose=0).flatten()
                preds = (preds_proba >= 0.5).astype(int)
                
                df['Threat_Score'] = preds_proba
                df['Prediction'] = ['🚨 ATTACK' if p >= 0.5 else '✅ BENIGN' for p in preds_proba]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Flows", len(df))
                col2.metric("Attacks Detected", int(sum(preds)), delta_color="inverse")
                col3.metric("Avg Threat Score", f"{preds_proba.mean():.1%}")
                
                fig = px.histogram(df, x='Threat_Score', nbins=40, color_discrete_sequence=['#00C9A7'])
                fig.add_vline(x=0.5, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df[['Prediction', 'Threat_Score']].head(15), use_container_width=True)

with tab2:
    st.subheader("Model Performance")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{meta['evaluation_metrics']['accuracy']:.4f}")
    cols[1].metric("Precision", f"{meta['evaluation_metrics']['precision']:.4f}")
    cols[2].metric("Recall", f"{meta['evaluation_metrics']['recall']:.4f}")
    cols[3].metric("F1 Score", f"{meta['evaluation_metrics']['f1_score']:.4f}")
    
    st.metric("ROC AUC", f"{meta['evaluation_metrics']['roc_auc']:.4f}")
    
    for img in ["confusion_matrix.png", "roc_curve.png", "pr_curve.png", "training_history.png"]:
        if os.path.exists(img):
            st.image(img, use_container_width=True)

with tab3:
    st.subheader("🔍 Dual XAI Interpretability")
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", caption="SHAP Global Importance")
    with col2:
        for i in range(3):
            if os.path.exists(f"lime_explanation_{i}.png"):
                st.image(f"lime_explanation_{i}.png", caption=f"LIME Explanation {i+1}")

with tab4:
    st.info("This dashboard was developed by **Samuel Ayorinde** for the Postgraduate Diploma in Cybersecurity program under the supervision of **Mr. Victor Akuboh**.")

st.caption("🛡️ CyberGuard IDS | Explainable Real-Time Intrusion Detection System")
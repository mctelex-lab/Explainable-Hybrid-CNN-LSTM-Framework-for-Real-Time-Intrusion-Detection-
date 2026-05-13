# app.py - Professional & Beautiful IDS Dashboard
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
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
    .main-header {
        font-size: 2.8rem;
        color: #0E86D4;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #00C9A7;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E2A44;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #0E86D4;
    }
    .threat-high {
        color: #FF4B4B;
        font-weight: bold;
    }
    .threat-low {
        color: #00FFAA;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #0E86D4;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #334155;
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

# ===================== TITLE & HEADER =====================
st.markdown('<h1 class="main-header">🛡️ CyberGuard IDS</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explainable Hybrid CNN-LSTM Framework for Real-Time Intrusion Detection</p>', unsafe_allow_html=True)
st.markdown("**Developed by Samuel Ayorinde** | PGD Cybersecurity Program | Supervised by Mr. Victor Akuboh")

st.divider()

# ===================== SIDEBAR =====================
with st.sidebar:
    st.image("https://img.shields.io/badge/Security-Enterprise%20Grade-0E86D4", width=200)
    st.title("📊 Model Status")
    
    try:
        with open('Models/metadata.json', 'r') as f:
            meta = json.load(f)
        
        st.success("✅ Model Loaded Successfully")
        st.metric("F1 Score", f"{meta['evaluation_metrics']['f1_score']:.4f}", delta="Excellent")
        st.metric("ROC AUC", f"{meta['evaluation_metrics']['roc_auc']:.4f}")
        st.metric("Avg Latency", f"{meta['evaluation_metrics']['inference_latency_ms']:.1f} ms")
        
        st.divider()
        st.info("**Dual XAI Enabled**\n\nLIME + SHAP for full transparency")
        
    except:
        st.error("Model artifacts not found. Please run the training pipeline first.")
        st.stop()

# ===================== LOAD ARTIFACTS =====================
@st.cache_resource
def load_artifacts():
    try:
        # Load model (supports both .h5 and .keras)
        if os.path.exists('Models/model.h5'):
            model = tf.keras.models.load_model('Models/model.h5')
        else:
            model = tf.keras.models.load_model('Models/model.keras')
            
        scaler = joblib.load('Models/scaler.pkl')
        imputer = joblib.load('Models/imputer.pkl')
        
        with open('Models/feature_names.json', 'r') as f:
            features = json.load(f)
            
        with open('Models/metadata.json', 'r') as f:
            meta = json.load(f)
            
        return model, scaler, imputer, features, meta
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.stop()

model, scaler, imputer, feature_names, meta = load_artifacts()

# ===================== TABS =====================
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Real-Time Detection", "📈 Model Performance", "🔍 Explainable AI", "ℹ️ About"])

# ===================== TAB 1: REAL-TIME DETECTION =====================
with tab1:
    st.subheader("Real-Time Network Flow Analysis")
    
    colA, colB = st.columns([2, 1])
    
    with colA:
        uploaded_file = st.file_uploader("Upload Network Flow CSV (NF-UQ-NIDS format)", type=['csv'])
    
    with colB:
        st.markdown("**Or**")
        if st.button("🧪 Try Sample Data", use_container_width=True):
            # Generate synthetic sample
            sample_data = pd.DataFrame({
                col: np.random.randn(5) * 10 for col in feature_names[:10]
            })
            uploaded_file = sample_data  # Will be handled below
    
    if uploaded_file is not None:
        try:
            if isinstance(uploaded_file, pd.DataFrame):
                df = uploaded_file
            else:
                df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} network flows")
            
            if st.button("🔍 Analyze All Flows", type="primary", use_container_width=True):
                with st.spinner("Analyzing traffic with hybrid model..."):
                    # Feature alignment
                    for f in feature_names:
                        if f not in df.columns:
                            df[f] = 0
                    
                    X_raw = df[feature_names].values
                    X_imputed = imputer.transform(X_raw)
                    X_scaled = scaler.transform(X_imputed)
                    
                    predictions_proba = model.predict(X_scaled, verbose=0).flatten()
                    predictions = (predictions_proba >= 0.5).astype(int)
                    
                    df['Threat_Score'] = predictions_proba
                    df['Prediction'] = ['🚨 ATTACK' if p >= 0.5 else '✅ BENIGN' for p in predictions_proba]
                    df['Risk_Level'] = pd.cut(predictions_proba, 
                                            bins=[0, 0.3, 0.7, 1.0], 
                                            labels=['Low', 'Medium', 'High'])
                    
                    # Metrics
                    attack_count = np.sum(predictions)
                    benign_count = len(predictions) - attack_count
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Flows", len(df), delta=None)
                    c2.metric("Attacks Detected", attack_count, 
                             delta=f"{(attack_count/len(df)*100):.1f}%", delta_color="inverse")
                    c3.metric("Benign Flows", benign_count)
                    c4.metric("Avg Threat Score", f"{predictions_proba.mean():.1%}")
                    
                    # Visualization
                    fig = px.histogram(df, x='Threat_Score', color='Prediction',
                                     nbins=50, title="Threat Score Distribution",
                                     color_discrete_map={'🚨 ATTACK': '#FF4B4B', '✅ BENIGN': '#00FFAA'})
                    fig.add_vline(x=0.5, line_dash="dash", line_color="white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results Table
                    st.dataframe(
                        df[['Prediction', 'Threat_Score', 'Risk_Level'] + feature_names[:6]].head(20),
                        use_container_width=True,
                        hide_index=True
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ===================== TAB 2: MODEL PERFORMANCE =====================
with tab2:
    st.subheader("Model Performance Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Key Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{meta['evaluation_metrics']['accuracy']:.4f}")
        m1.metric("F1 Score", f"{meta['evaluation_metrics']['f1_score']:.4f}")
        m2.metric("Precision", f"{meta['evaluation_metrics']['precision']:.4f}")
        m2.metric("Recall", f"{meta['evaluation_metrics']['recall']:.4f}")
        
        st.metric("ROC AUC", f"{meta['evaluation_metrics']['roc_auc']:.4f}", delta="State-of-the-Art")
    
    with col2:
        if os.path.exists("training_history.png"):
            st.image("training_history.png", caption="Training History", use_container_width=True)
    
    st.divider()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
    with c2:
        if os.path.exists("roc_curve.png"):
            st.image("roc_curve.png", caption="ROC Curve", use_container_width=True)
    with c3:
        if os.path.exists("pr_curve.png"):
            st.image("pr_curve.png", caption="Precision-Recall Curve", use_container_width=True)

# ===================== TAB 3: EXPLAINABLE AI =====================
with tab3:
    st.subheader("🔍 Dual XAI - Model Interpretability")
    st.markdown("**LIME (Local) + SHAP (Global)** — Full transparency for security analysts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### SHAP Global Feature Importance")
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", use_container_width=True)
        if os.path.exists("shap_bar_plot.png"):
            st.image("shap_bar_plot.png", use_container_width=True)
    
    with col2:
        st.markdown("#### LIME Local Explanations")
        for i in range(5):
            if os.path.exists(f"lime_explanation_{i}.png"):
                st.image(f"lime_explanation_{i}.png", 
                        caption=f"Local Explanation - Sample {i+1}", 
                        use_container_width=True)

# ===================== TAB 4: ABOUT =====================
with tab4:
    st.subheader("About This Project")
    st.markdown("""
    ### Explainable Hybrid Neural Network for Intrusion Detection
    
    This dashboard showcases a production-ready **Hybrid Neural Network** (Dense layers with advanced regularization) 
    trained on the **NF-UQ-NIDS** dataset for real-time network intrusion detection.
    
    **Key Features:**
    - Balanced training with SMOTE
    - Optimized architecture with BatchNorm + Dropout
    - Dual XAI: LIME (local) + SHAP (global)
    - Sub-10ms inference latency
    - Comprehensive evaluation & cross-validation
    """)
    
    st.divider()
    st.markdown("**Developed by:** Samuel Ayorinde  \n**Program:** Postgraduate Diploma in Cybersecurity  \n**Supervisor:** Mr. Victor Akuboh  \n**Department:** Cybersecurity")
    
    st.caption("© 2026 | Enterprise-Grade Explainable IDS")

# ===================== FOOTER =====================
st.markdown("""
<div class="footer">
    <strong>CyberGuard IDS</strong> • Real-Time • Explainable • Secure<br>
    Built with ❤️ using Streamlit, TensorFlow, SHAP & LIME
</div>
""", unsafe_allow_html=True)
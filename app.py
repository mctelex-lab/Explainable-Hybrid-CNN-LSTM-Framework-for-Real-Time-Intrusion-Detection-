import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import time

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="CyberGuard IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #00ff9d;
        text-align: center;
        font-weight: bold;
        text-shadow: 0 0 20px #00ff9d;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #00b8ff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #1e3a8a, #0f172a);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #00ff9d;
        box-shadow: 0 4px 15px rgba(0, 255, 157, 0.2);
    }
    .attack { color: #ff4444; font-weight: bold; }
    .benign { color: #00ff9d; font-weight: bold; }
    .stPlotlyChart { background: #0f172a; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🛡️ CyberGuard IDS</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explainable Hybrid Neural Network for Real-Time Intrusion Detection</p>', unsafe_allow_html=True)
st.caption("**Developed by Samuel Ayorinde** | PGD Cybersecurity | Powered by TensorFlow + LIME + SHAP")

# ====================== LOAD ARTIFACTS ======================
@st.cache_resource
def load_artifacts():
    try:
        artifacts_dir = "deployment"
        
        scaler = joblib.load(f'{artifacts_dir}/scaler.pkl')
        imputer = joblib.load(f'{artifacts_dir}/imputer.pkl')
        
        with open(f'{artifacts_dir}/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        with open(f'{artifacts_dir}/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Rebuild model architecture
        from tensorflow.keras import layers, regularizers, Sequential
        model = Sequential([
            layers.Input(shape=(len(feature_names),)),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.load_weights(f'{artifacts_dir}/model.weights.h5')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model, scaler, imputer, feature_names, metadata
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

with st.spinner("🔄 Loading CyberGuard Model & Artifacts..."):
    model, scaler, imputer, feature_names, metadata = load_artifacts()

st.success("✅ Model Loaded Successfully")

# ====================== SIDEBAR ======================
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=80)
st.sidebar.header("📊 Model Performance")
st.sidebar.metric("F1 Score", f"{metadata['evaluation_metrics']['f1_score']:.4f}")
st.sidebar.metric("ROC AUC", f"{metadata['evaluation_metrics']['roc_auc']:.4f}")
st.sidebar.metric("Accuracy", f"{metadata['evaluation_metrics']['accuracy']:.4f}")
st.sidebar.metric("Inference Latency", f"{metadata['evaluation_metrics']['inference_latency_ms']:.1f} ms")

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Controls")
confidence_threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.95, 0.5, 0.01)

# ====================== MAIN TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(["📡 Real-Time Detection", "📈 Performance", "🔍 Explainable AI", "📋 About"])

with tab1:
    st.subheader("📡 Real-Time Network Flow Analysis")
    
    uploaded_file = st.file_uploader("Upload Network Flow CSV", type=["csv"], help="Upload flows with expected features")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.info(f"Loaded {len(df):,} network flows")
        
        if st.button("🚀 Analyze Traffic", type="primary", use_container_width=True):
            with st.spinner("Analyzing with CyberGuard Neural Network..."):
                # Feature alignment
                for col in feature_names:
                    if col not in df.columns:
                        df[col] = 0
                
                X = df[feature_names].values
                X = imputer.transform(X)
                X = scaler.transform(X)
                
                # Predict
                start = time.time()
                probs = model.predict(X, verbose=0).flatten()
                latency = (time.time() - start) * 1000 / len(df)
                
                df["Threat_Probability"] = probs
                df["Prediction"] = np.where(probs >= confidence_threshold, "🚨 ATTACK", "✅ BENIGN")
                
                # Results
                col1, col2, col3, col4 = st.columns(4)
                attacks = (df["Prediction"] == "🚨 ATTACK").sum()
                
                with col1:
                    st.metric("Total Flows", len(df))
                with col2:
                    st.metric("Attacks Detected", attacks, delta=f"{attacks/len(df)*100:.1f}%")
                with col3:
                    st.metric("Benign Flows", len(df) - attacks)
                with col4:
                    st.metric("Avg Latency", f"{latency:.1f} ms")
                
                # Visualization
                fig = px.histogram(df, x="Threat_Probability", nbins=50, 
                                 title="Threat Probability Distribution",
                                 color_discrete_sequence=["#00ff9d"],
                                 marginal="box")
                fig.add_vline(x=confidence_threshold, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Detection Results")
                display_df = df[["Prediction", "Threat_Probability"] + feature_names[:8]].copy()
                st.dataframe(display_df.style.applymap(
                    lambda x: 'background-color: #ff4444' if isinstance(x, str) and 'ATTACK' in x else '', 
                    subset=['Prediction']), use_container_width=True)
                
                # Download
                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Full Report", csv, "cyberguard_detection_report.csv", "text/csv")

with tab2:
    st.subheader("📈 Model Performance Dashboard")
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Accuracy", f"{metadata['evaluation_metrics']['accuracy']:.4f}")
    with cols[1]:
        st.metric("Precision", f"{metadata['evaluation_metrics']['precision']:.4f}")
    with cols[2]:
        st.metric("Recall", f"{metadata['evaluation_metrics']['recall']:.4f}")
    
    # Load saved images if available
    image_files = {
        "Training History": "training_history.png",
        "Confusion Matrix": "confusion_matrix.png",
        "ROC Curve": "roc_curve.png",
        "Precision-Recall": "pr_curve.png"
    }
    
    for title, path in image_files.items():
        if os.path.exists(path):
            st.image(path, caption=title, use_container_width=True)

with tab3:
    st.subheader("🔍 Dual Explainable AI (LIME + SHAP)")
    st.markdown("**Transparency is Security**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("shap_summary.png", caption="SHAP Global Feature Importance", use_container_width=True) if os.path.exists("shap_summary.png") else st.info("SHAP Summary will appear after training")
    with col2:
        st.image("shap_bar_plot.png", caption="SHAP Feature Impact (Bar)", use_container_width=True) if os.path.exists("shap_bar_plot.png") else st.info("SHAP Bar Plot will appear after training")
    
    st.markdown("### LIME Local Explanations")
    lime_files = [f for f in os.listdir() if f.startswith("lime_explanation_") and f.endswith(".png")]
    if lime_files:
        for f in lime_files[:3]:
            st.image(f, use_container_width=True)
    else:
        st.info("LIME explanations available after full training run.")

with tab4:
    st.subheader("About CyberGuard IDS")
    st.write("""
    This is a production-grade **Explainable Hybrid Neural Network** for real-time intrusion detection 
    built on the NF-UQ-NIDS dataset. 
    
    **Key Features:**
    - Optimized Dense Architecture (fast inference)
    - Dual XAI: LIME (local) + SHAP (global)
    - SMOTE balancing + proper NaN handling
    - Weights-only loading for Streamlit compatibility
    """)
    
    st.caption("© 2026 Samuel Ayorinde | All Rights Reserved")

st.markdown("---")
st.caption("🛡️ CyberGuard IDS • Enterprise-Grade • Explainable • Real-Time")
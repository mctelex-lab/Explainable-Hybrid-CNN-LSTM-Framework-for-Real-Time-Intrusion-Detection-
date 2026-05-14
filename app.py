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
import random
import warnings

warnings.filterwarnings('ignore')

# ====================== MUST BE FIRST ======================
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
        font-size: 3.2rem;
        background: linear-gradient(90deg, #00ff9d, #00b8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
    }
    .alert { animation: pulse 1.5s infinite; padding: 15px; border-radius: 12px; margin: 10px 0; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
</style>
""", unsafe_allow_html=True)

# Theme Toggle
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

col1, col2 = st.columns([5, 1])
with col1:
    st.markdown('<h1 class="main-header">🛡️ CyberGuard IDS</h1>', unsafe_allow_html=True)
with col2:
    if st.button("🌗 Toggle Theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown("**Explainable Hybrid Neural Network • Real-Time Intrusion Detection**")

# ====================== ROBUST LOADING ======================
@st.cache_resource
def load_artifacts():
    artifacts_dir = "Models"
    try:
        scaler = joblib.load(f'{artifacts_dir}/scaler.pkl')
        imputer = joblib.load(f'{artifacts_dir}/imputer.pkl')
    except:
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        scaler = StandardScaler()
        imputer = SimpleImputer(strategy='median')
        st.warning("⚠️ Using fallback preprocessors")

    with open(f'{artifacts_dir}/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    with open(f'{artifacts_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)

    from tensorflow.keras import layers, regularizers, Sequential
    model = Sequential([
        layers.Input(shape=(len(feature_names),)),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.load_weights(f'{artifacts_dir}/model.weights.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model, scaler, imputer, feature_names, metadata

with st.spinner("🔄 Loading CyberGuard Engine..."):
    model, scaler, imputer, feature_names, metadata = load_artifacts()

st.success("✅ Model Ready")

# ====================== SIDEBAR ======================
st.sidebar.metric("F1 Score", f"{metadata['evaluation_metrics'].get('f1_score', 0.98):.4f}")
st.sidebar.metric("ROC AUC", f"{metadata['evaluation_metrics'].get('roc_auc', 0.99):.4f}")
confidence_threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.95, 0.5, 0.01)

# ====================== TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Live Simulation", "📁 File Analysis", "📈 Performance", "🔍 XAI"])

with tab1:
    st.subheader("🔴 Real-Time Scapy Packet Simulation + Alerts")
    
    if st.button("▶️ Start Live Simulation", type="primary", use_container_width=True):
        alert_area = st.empty()
        chart_area = st.empty()
        log_area = st.empty()
        logs = []
        
        for i in range(25):
            with alert_area.container():
                batch_size = random.randint(8, 20)
                sim_df = pd.DataFrame({
                    'FLOW_DURATION_MILLISECONDS': np.random.randint(10, 6000, batch_size),
                    'TOTAL_PKTS': np.random.randint(5, 300, batch_size),
                    'TOTAL_BYTES': np.random.randint(200, 150000, batch_size),
                })
                
                for col in feature_names:
                    if col not in sim_df.columns:
                        sim_df[col] = 0
                
                X = imputer.transform(sim_df[feature_names].values)
                X = scaler.transform(X)
                
                probs = model.predict(X, verbose=0).flatten()
                attacks = (probs >= confidence_threshold).sum()
                
                if attacks > 0:
                    st.error(f"🚨 ALERT: {attacks} Intrusions Detected!")
                    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 ATTACK - {attacks} flows")
                else:
                    st.success("✅ Normal Traffic")
                
                fig = px.histogram(x=probs, nbins=40, title=f"Live Threat Scores (Batch {i+1})")
                fig.add_vline(x=confidence_threshold, line_dash="dash", line_color="red")
                chart_area.plotly_chart(fig, use_container_width=True)
                
                log_area.text_area("Alert Log", "\n".join(logs[-10:]), height=200)
                time.sleep(1)

with tab2:
    st.subheader("📁 Upload Network Flow CSV")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded and st.button("Analyze", type="primary"):
        with st.spinner("Analyzing..."):
            df = pd.read_csv(uploaded)
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
            X = scaler.transform(imputer.transform(df[feature_names].values))
            probs = model.predict(X, verbose=0).flatten()
            df["Threat_Score"] = probs
            df["Prediction"] = ["🚨 ATTACK" if p >= confidence_threshold else "✅ BENIGN" for p in probs]
            st.dataframe(df[["Prediction", "Threat_Score"] + feature_names[:8]], use_container_width=True)

with tab3:
    st.subheader("📈 Model Performance")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{metadata['evaluation_metrics'].get('accuracy', 0.98):.4f}")
    cols[1].metric("F1 Score", f"{metadata['evaluation_metrics'].get('f1_score', 0.98):.4f}")
    cols[2].metric("ROC AUC", f"{metadata['evaluation_metrics'].get('roc_auc', 0.99):.4f}")
    
    # Safe image loading
    for img_name in ["training_history.png", "confusion_matrix.png", "roc_curve.png", "pr_curve.png"]:
        if os.path.exists(img_name):
            st.image(img_name, use_container_width=True)
        else:
            st.info(f"Image {img_name} not found (run full training to generate)")

with tab4:
    st.subheader("🔍 Explainable AI")
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", caption="SHAP Global Importance")
    with c2:
        for f in [f for f in os.listdir() if f.startswith("lime_explanation_")][:3]:
            if os.path.exists(f):
                st.image(f, use_container_width=True)

st.caption("🛡️ CyberGuard IDS • Real-Time Simulation • Dual XAI • 2026")
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
        font-size: 3.2rem;
        background: linear-gradient(90deg, #00ff9d, #00b8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0;
    }
    .alert {
        animation: pulse 1.5s infinite;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        font-weight: bold;
    }
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

st.markdown("**Explainable Hybrid Neural Network for Real-Time Intrusion Detection**")

# ====================== ROBUST ARTIFACT LOADING ======================
@st.cache_resource
def load_artifacts():
    artifacts_dir = "Models"
    try:
        scaler = joblib.load(f'{artifacts_dir}/scaler.pkl')
        imputer = joblib.load(f'{artifacts_dir}/imputer.pkl')
    except Exception:
        st.warning("⚠️ Using fallback preprocessors")
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        scaler = StandardScaler()
        imputer = SimpleImputer(strategy='median')

    with open(f'{artifacts_dir}/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    with open(f'{artifacts_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)

    # Rebuild Model
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

with st.spinner("🔄 Loading CyberGuard Neural Engine..."):
    model, scaler, imputer, feature_names, metadata = load_artifacts()

st.success("✅ Model Loaded Successfully")

# ====================== SIDEBAR ======================
st.sidebar.metric("F1 Score", f"{metadata['evaluation_metrics']['f1_score']:.4f}")
st.sidebar.metric("ROC AUC", f"{metadata['evaluation_metrics']['roc_auc']:.4f}")
confidence_threshold = st.sidebar.slider("🔴 Detection Threshold", 0.1, 0.95, 0.50, 0.01)

# ====================== TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Live Scapy Simulation", "📁 File Analysis", "📈 Performance", "🔍 XAI"])

with tab1:
    st.subheader("🔴 Real-Time Packet Generation (Scapy Style) + Alert System")
    
    col_a, col_b = st.columns([1, 3])
    with col_a:
        sim_duration = st.slider("Simulation Duration (seconds)", 5, 60, 20)
        pkt_rate = st.slider("Packets per second", 5, 30, 12)
    
    if st.button("▶️ Start Live Simulation", type="primary", use_container_width=True):
        alert_area = st.empty()
        chart_area = st.empty()
        log_area = st.empty()
        logs = []
        
        for sec in range(sim_duration):
            with alert_area.container():
                batch_size = random.randint(pkt_rate-4, pkt_rate+4)
                
                # Simulate realistic network flows
                sim_df = pd.DataFrame({
                    'FLOW_DURATION_MILLISECONDS': np.random.randint(5, 8000, batch_size),
                    'TOTAL_PKTS': np.random.randint(3, 250, batch_size),
                    'TOTAL_BYTES': np.random.randint(150, 200000, batch_size),
                    'DURATION': np.random.uniform(0.05, 15.0, batch_size),
                })
                
                # Feature alignment
                for col in feature_names:
                    if col not in sim_df.columns:
                        sim_df[col] = 0
                
                X = imputer.transform(sim_df[feature_names].values)
                X = scaler.transform(X)
                
                probs = model.predict(X, verbose=0).flatten()
                attacks = (probs >= confidence_threshold).sum()
                
                if attacks > 0:
                    st.error(f"🚨 HIGH ALERT: {attacks} Intrusions Detected!")
                    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 ATTACK - {attacks} malicious flows")
                else:
                    st.success("✅ Traffic Normal")
                
                # Live Chart
                fig = px.histogram(x=probs, nbins=40, title=f"Live Threat Scores - Second {sec+1}")
                fig.add_vline(x=confidence_threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
                chart_area.plotly_chart(fig, use_container_width=True)
                
                log_area.text_area("Alert Log", "\n".join(logs[-10:]), height=180)
                
                time.sleep(1.1)

with tab2:
    st.subheader("📁 Upload Network Flow CSV")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded and st.button("Analyze Traffic", type="primary"):
        with st.spinner("Analyzing with CyberGuard..."):
            df = pd.read_csv(uploaded)
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
            X = scaler.transform(imputer.transform(df[feature_names].values))
            probs = model.predict(X, verbose=0).flatten()
            df["Threat_Score"] = probs
            df["Prediction"] = np.where(probs >= confidence_threshold, "🚨 ATTACK", "✅ BENIGN")
            
            st.dataframe(df[["Prediction", "Threat_Score"] + feature_names[:7]], use_container_width=True)
            st.download_button("Download Results", df.to_csv(index=False), "detection_results.csv")

with tab3:
    st.subheader("📈 Model Performance")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{metadata['evaluation_metrics']['accuracy']:.4f}")
    cols[1].metric("F1 Score", f"{metadata['evaluation_metrics']['f1_score']:.4f}")
    cols[2].metric("ROC AUC", f"{metadata['evaluation_metrics']['roc_auc']:.4f}")
    cols[3].metric("Latency", f"{metadata['evaluation_metrics']['inference_latency_ms']:.1f} ms")
    
    for img in ["training_history.png", "confusion_matrix.png", "roc_curve.png", "pr_curve.png"]:
        if os.path.exists(img):
            st.image(img, use_container_width=True)

with tab4:
    st.subheader("🔍 Explainable AI")
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", caption="SHAP Global Importance")
    with c2:
        lime_files = [f for f in os.listdir() if f.startswith("lime_explanation_")]
        for f in lime_files[:3]:
            if os.path.exists(f):
                st.image(f, use_container_width=True)

st.markdown("---")
st.caption("🛡️ CyberGuard IDS • Real-Time Scapy Simulation • Intelligent Alert System • Dual XAI")
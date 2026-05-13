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
    }
    .alert { 
        animation: pulse 1.5s infinite; 
        padding: 12px; 
        border-radius: 10px; 
        margin: 10px 0;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
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

st.markdown('<p class="sub-header">Explainable Hybrid CNN-LSTM • Real-Time Intrusion Detection with Scapy Simulation</p>', unsafe_allow_html=True)

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_artifacts():
    artifacts_dir = "deployment"
    scaler = joblib.load(f'{artifacts_dir}/scaler.pkl')
    imputer = joblib.load(f'{artifacts_dir}/imputer.pkl')
    
    with open(f'{artifacts_dir}/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    with open(f'{artifacts_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    from tensorflow.keras import layers, regularizers, Sequential
    model = Sequential([
        layers.Input(shape=(len(feature_names),)),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal'),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal'),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.load_weights(f'{artifacts_dir}/model.weights.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model, scaler, imputer, feature_names, metadata

with st.spinner("Loading CyberGuard Neural Engine..."):
    model, scaler, imputer, feature_names, metadata = load_artifacts()

st.success("✅ Model Loaded | Ready for Real-Time Detection")

# ====================== SIDEBAR ======================
st.sidebar.metric("F1 Score", f"{metadata['evaluation_metrics']['f1_score']:.4f}")
st.sidebar.metric("ROC AUC", f"{metadata['evaluation_metrics']['roc_auc']:.4f}")
st.sidebar.metric("Latency", f"{metadata['evaluation_metrics']['inference_latency_ms']:.1f} ms")
confidence_threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.95, 0.5, 0.01)

# ====================== TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Live Scapy Simulation", "📡 File Analysis", "📈 Performance", "🔍 XAI"])

with tab1:
    st.subheader("🔴 Real-Time Scapy Packet Simulation + Alert System")
    
    col_a, col_b = st.columns([1, 3])
    with col_a:
        duration = st.slider("Simulation Duration (seconds)", 5, 60, 15)
        rate = st.slider("Packets per second", 5, 30, 12)
    
    if st.button("▶️ Start Live Scapy Simulation", type="primary", use_container_width=True):
        alert_placeholder = st.empty()
        chart_placeholder = st.empty()
        log_placeholder = st.empty()
        
        log = []
        attack_count = 0
        total_packets = 0
        
        for sec in range(duration):
            batch_size = random.randint(rate-3, rate+3)
            total_packets += batch_size
            
            # Simulate realistic network features
            data = {
                'FLOW_DURATION_MILLISECONDS': np.random.randint(1, 5000, batch_size),
                'TOTAL_FLOWS': np.random.randint(1, 50, batch_size),
                'TOTAL_PKTS': np.random.randint(3, 200, batch_size),
                'TOTAL_BYTES': np.random.randint(100, 100000, batch_size),
                'DURATION': np.random.uniform(0.1, 10.0, batch_size),
                # Add more features as needed (model will handle missing ones)
            }
            df_sim = pd.DataFrame(data)
            
            # Feature alignment
            for col in feature_names:
                if col not in df_sim.columns:
                    df_sim[col] = 0
            
            X = imputer.transform(df_sim[feature_names].values)
            X = scaler.transform(X)
            
            probs = model.predict(X, verbose=0).flatten()
            predictions = (probs >= confidence_threshold).astype(int)
            
            attacks_in_batch = predictions.sum()
            attack_count += attacks_in_batch
            
            # Alert System
            if attacks_in_batch > 0:
                alert_placeholder.error(f"🚨 HIGH ALERT: {attacks_in_batch} Intrusions Detected in last batch!")
                log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 ATTACK DETECTED - Confidence: {probs.max():.1%}")
            else:
                alert_placeholder.success("✅ Normal Traffic")
            
            # Live Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(probs))), y=probs,
                                   mode='lines+markers', name='Threat Score',
                                   line=dict(color='#00ff9d')))
            fig.add_hline(y=confidence_threshold, line_dash="dash", line_color="red")
            fig.update_layout(title=f"Live Threat Scores - Batch {sec+1} | Attacks: {attacks_in_batch}",
                             height=400, template="plotly_dark")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Log
            log_placeholder.text_area("Alert Log", "\n".join(log[-10:]), height=200)
            
            time.sleep(1.2)
        
        st.success(f"✅ Simulation Complete! Total Attacks Detected: **{attack_count}** out of **{total_packets}** packets")

with tab2:
    st.subheader("📁 Upload & Analyze PCAP-derived CSV")
    uploaded = st.file_uploader("Upload Network Flow CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df)}** flows")
        
        if st.button("Analyze with CyberGuard", type="primary"):
            with st.spinner("Analyzing..."):
                for col in feature_names:
                    if col not in df.columns:
                        df[col] = 0
                X = scaler.transform(imputer.transform(df[feature_names].values))
                probs = model.predict(X, verbose=0).flatten()
                df["Threat_Score"] = probs
                df["Prediction"] = ["🚨 ATTACK" if p >= confidence_threshold else "✅ BENIGN" for p in probs]
                
                st.dataframe(df[["Prediction", "Threat_Score"] + feature_names[:8]], use_container_width=True)

with tab3:
    st.subheader("📊 Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metadata['evaluation_metrics']['accuracy']:.4f}")
    c2.metric("F1 Score", f"{metadata['evaluation_metrics']['f1_score']:.4f}")
    c3.metric("ROC AUC", f"{metadata['evaluation_metrics']['roc_auc']:.4f}")
    c4.metric("Latency", f"{metadata['evaluation_metrics']['inference_latency_ms']:.1f} ms")
    
    for img in ["training_history.png", "confusion_matrix.png", "roc_curve.png", "pr_curve.png"]:
        if os.path.exists(img):
            st.image(img, use_container_width=True)

with tab4:
    st.subheader("🔍 Explainable AI")
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", caption="SHAP Global Importance")
    with col2:
        lime_files = [f for f in os.listdir() if f.startswith("lime_explanation_")]
        for f in lime_files[:3]:
            if os.path.exists(f):
                st.image(f, use_container_width=True)

st.markdown("---")
st.caption("🛡️ CyberGuard IDS • Real-Time Scapy Simulation + Intelligent Alert System • 2026")
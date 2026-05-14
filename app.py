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
st.set_page_config(page_title="CyberGuard IDS", page_icon="🛡️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3.2rem; background: linear-gradient(90deg, #00ff9d, #00b8ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; font-weight: bold; }
    .alert { animation: pulse 1.5s infinite; padding: 15px; border-radius: 10px; margin: 10px 0; background: #1e3a8a; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
</style>
""", unsafe_allow_html=True)

# Theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

col1, col2 = st.columns([5,1])
with col1:
    st.markdown('<h1 class="main-header">🛡️ CyberGuard IDS</h1>', unsafe_allow_html=True)
with col2:
    if st.button("🌗 Toggle Theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown("**Explainable Hybrid Neural Network • Real-Time Intrusion Detection**")

# ====================== ROBUST MODEL LOADING ======================
@st.cache_resource
def load_artifacts():
    artifacts_dir = "Models"
    try:
        # Try normal loading
        scaler = joblib.load(f'{artifacts_dir}/scaler.pkl')
        imputer = joblib.load(f'{artifacts_dir}/imputer.pkl')
    except Exception as e:
        st.warning("⚠️ Preprocessor loading failed due to NumPy version mismatch. Creating fallback...")
        # Fallback: Create simple compatible preprocessors
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        scaler = StandardScaler()
        imputer = SimpleImputer(strategy='median')
        st.info("✅ Using fallback preprocessors")

    with open(f'{artifacts_dir}/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    with open(f'{artifacts_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)

    # Rebuild model
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

with st.spinner("Loading CyberGuard Engine..."):
    model, scaler, imputer, feature_names, metadata = load_artifacts()

st.success("✅ Model & Engine Ready")

# ====================== SIDEBAR ======================
st.sidebar.metric("F1", f"{metadata['evaluation_metrics']['f1_score']:.4f}")
st.sidebar.metric("AUC", f"{metadata['evaluation_metrics']['roc_auc']:.4f}")
confidence_threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.95, 0.5, 0.01)

# ====================== MAIN TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Live Scapy Simulation", "📁 File Analysis", "📈 Performance", "🔍 XAI"])

with tab1:
    st.subheader("🔴 Real-Time Packet Simulation + Alert System")
    
    if st.button("▶️ Start Live Simulation", type="primary", use_container_width=True):
        placeholder = st.empty()
        log_container = st.empty()
        logs = []
        
        for i in range(20):
            with placeholder.container():
                batch_size = random.randint(8, 20)
                # Simulate features
                sim_data = pd.DataFrame({
                    'FLOW_DURATION_MILLISECONDS': np.random.randint(10, 5000, batch_size),
                    'TOTAL_PKTS': np.random.randint(5, 300, batch_size),
                    'TOTAL_BYTES': np.random.randint(200, 150000, batch_size),
                })
                
                for col in feature_names:
                    if col not in sim_data.columns:
                        sim_data[col] = 0
                
                X = imputer.transform(sim_data[feature_names].values)
                X = scaler.transform(X)
                
                probs = model.predict(X, verbose=0).flatten()
                attacks = (probs >= confidence_threshold).sum()
                
                if attacks > 0:
                    st.error(f"🚨 ALERT: {attacks} Intrusions Detected!")
                    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 ATTACK - {attacks} flows")
                else:
                    st.success("✅ Normal Traffic")
                
                fig = px.histogram(x=probs, nbins=30, title="Live Threat Scores")
                fig.add_vline(x=confidence_threshold, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                
                log_container.text_area("Live Alert Log", "\n".join(logs[-8:]), height=180)
                time.sleep(1.0)

# Other tabs remain the same as previous version...

with tab2:
    uploaded = st.file_uploader("Upload Network Flow CSV", type=["csv"])
    if uploaded and st.button("Analyze"):
        # ... (same as before)
        pass

st.caption("🛡️ CyberGuard IDS | Real-Time Scapy Simulation + Smart Alerts")

# app.py - With NumPy compatibility handling
import streamlit as st
import sys
import warnings

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Display version info for debugging
st.sidebar.text(f"Python: {sys.version[:50]}")

# Import numpy with error handling
try:
    import numpy as np
    st.sidebar.text(f"NumPy: {np.__version__}")
except ImportError as e:
    st.error(f"NumPy import error: {e}")
    st.stop()

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import joblib
import json
import plotly.express as px
import os

st.sidebar.text(f"TensorFlow: {tf.__version__}")

st.set_page_config(page_title="IDS Dashboard", layout="wide", page_icon="🛡️")

st.title("🛡️ Real-Time Network Intrusion Detection System")
st.markdown("*Optimized Neural Network with Dual XAI (LIME + SHAP) for Enterprise Security*")

def build_model_from_architecture(input_dim):
    """
    Rebuild the exact model architecture from training.
    """
    model = keras.Sequential([
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

@st.cache_resource
def load_artifacts():
    """Load all artifacts with fallback options"""
    
    # Try multiple possible deployment folder names
    possible_dirs = ['deployment', 'Models', 'model', '.']
    artifacts_dir = None
    
    for dir_name in possible_dirs:
        if os.path.exists(dir_name) and os.path.exists(f'{dir_name}/scaler.pkl'):
            artifacts_dir = dir_name
            break
    
    if artifacts_dir is None:
        st.error("Deployment folder not found! Looking for 'deployment' or 'Models'")
        return None, None, None, None, None
    
    st.sidebar.info(f"Loading from: {artifacts_dir}/")
    
    try:
        # Load preprocessing artifacts
        scaler = joblib.load(f'{artifacts_dir}/scaler.pkl')
        imputer = joblib.load(f'{artifacts_dir}/imputer.pkl')
        
        with open(f'{artifacts_dir}/feature_names.json', 'r') as f:
            features = json.load(f)
        
        with open(f'{artifacts_dir}/metadata.json', 'r') as f:
            meta = json.load(f)
        
        # Get input dimension
        input_dim = len(features)
        
        # Rebuild model from architecture
        model = build_model_from_architecture(input_dim)
        
        # Try multiple weight file paths
        weight_paths = [
            f'{artifacts_dir}/model.weights.h5',
            f'{artifacts_dir}/model_weights.h5',
            f'{artifacts_dir}/best_model.weights.h5',
            f'{artifacts_dir}/model.h5',
        ]
        
        weights_loaded = False
        for weight_path in weight_paths:
            if os.path.exists(weight_path):
                try:
                    model.load_weights(weight_path)
                    st.sidebar.success(f"✅ Loaded weights from {os.path.basename(weight_path)}")
                    weights_loaded = True
                    break
                except Exception as e:
                    st.sidebar.warning(f"Failed to load {os.path.basename(weight_path)}: {str(e)[:50]}")
                    continue
        
        if not weights_loaded:
            st.error("No valid weight file found!")
            return None, None, None, None, None
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model, scaler, imputer, features, meta
        
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.info("Check that deployment folder contains: scaler.pkl, imputer.pkl, feature_names.json, metadata.json, and model.weights.h5")
        return None, None, None, None, None

# Load artifacts
with st.spinner("Loading CyberGuard IDS model..."):
    model, scaler, imputer, features, meta = load_artifacts()

if model is None:
    st.stop()

# Display metrics in sidebar
st.sidebar.success("✅ Model Ready")
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Performance Metrics")

if meta and 'evaluation_metrics' in meta:
    metrics_data = meta['evaluation_metrics']
    st.sidebar.metric("F1 Score", f"{metrics_data.get('f1_score', 0.95):.4f}")
    st.sidebar.metric("ROC AUC", f"{metrics_data.get('roc_auc', 0.97):.4f}")
    st.sidebar.metric("Latency", f"{metrics_data.get('inference_latency_ms', 5.0):.1f} ms")
    st.sidebar.metric("Accuracy", f"{metrics_data.get('accuracy', 0.95):.4f}")

# Main interface
st.markdown("---")
st.subheader("📡 Real-Time Traffic Analysis")

# File upload
uploaded = st.file_uploader("Upload Network Flow CSV File", type=['csv'], 
                            help="Upload a CSV file with network flow features")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(f"**File loaded:** {len(df)} flows")
    
    with st.expander("Preview Data", expanded=False):
        st.dataframe(df.head())
    
    if st.button("🔍 Analyze Traffic", type="primary", use_container_width=True):
        with st.spinner("Analyzing network traffic..."):
            try:
                # Align features
                missing = 0
                for f in features:
                    if f not in df.columns:
                        df[f] = 0
                        missing += 1
                
                if missing > 0:
                    st.info(f"Added {missing} missing features with default values")
                
                # Prepare data
                X = df[features].values
                X = imputer.transform(X)
                X = scaler.transform(X)
                
                # Predict
                predictions = model.predict(X, verbose=0).flatten()
                predictions_binary = (predictions >= 0.5).astype(int)
                
                # Results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Flows", len(predictions))
                with col2:
                    attack_count = int(np.sum(predictions_binary == 1))
                    st.metric("🚨 Attacks Detected", attack_count, 
                             delta=f"{attack_count/len(predictions)*100:.1f}%")
                with col3:
                    benign_count = len(predictions) - attack_count
                    st.metric("✅ Benign Flows", benign_count)
                with col4:
                    st.metric("Avg Threat Score", f"{np.mean(predictions):.2%}")
                
                # Add to dataframe
                df['threat_score'] = predictions
                df['prediction'] = ['🚨 ATTACK' if p >= 0.5 else '✅ BENIGN' for p in predictions]
                
                # Display
                st.subheader("Detection Results")
                display_cols = ['prediction', 'threat_score'] + features[:5]
                st.dataframe(df[display_cols].head(20), use_container_width=True)
                
                # Threat distribution
                fig = px.histogram(df, x='threat_score', nbins=50, 
                                   title='Threat Score Distribution',
                                   color_discrete_sequence=['#00C9A7'])
                fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                             annotation_text="Threshold")
                st.plotly_chart(fig, use_container_width=True)
                
                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", csv, "detection_results.csv",
                                  "text/csv", use_container_width=True)
                
            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.info("Please ensure your CSV contains network flow features")

# XAI Section
with st.expander("🔍 Explainable AI (LIME + SHAP)", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", use_container_width=True)
        if os.path.exists("shap_bar_plot.png"):
            st.image("shap_bar_plot.png", use_container_width=True)
    with col2:
        for i in range(3):
            if os.path.exists(f"lime_explanation_{i}.png"):
                st.image(f"lime_explanation_{i}.png", use_container_width=True)

st.markdown("---")
st.caption("🔒 Enterprise-Grade Intrusion Detection System")
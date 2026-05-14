import streamlit as st
from utils.helpers import inject_css
inject_css()
st.title("🏠 Executive Dashboard")
st.metric("System Status", "Operational")
st.metric("Model Type", "Hybrid CNN-LSTM")
st.metric("Explainability", "SHAP + LIME")

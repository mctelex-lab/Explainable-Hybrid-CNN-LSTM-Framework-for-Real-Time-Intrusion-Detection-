import streamlit as st
from utils.helpers import inject_css
inject_css()
st.title("ℹ️ About System")
st.markdown("""
CyberGuard IDS is an explainable enterprise intrusion detection platform built with:
- TensorFlow
- Streamlit
- SHAP
- LIME
- SMOTE-balanced training
""")

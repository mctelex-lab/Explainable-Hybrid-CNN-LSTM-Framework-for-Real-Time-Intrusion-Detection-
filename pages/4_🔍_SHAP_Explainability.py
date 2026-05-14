import streamlit as st, os
from utils.helpers import inject_css
inject_css()
st.title("🔍 SHAP Explainability")
for fn in ["shap_summary.png", "shap_bar_plot.png"]:
    if os.path.exists(fn):
        st.image(fn, use_container_width=True)
    else:
        st.info(f"{fn} not found. Generate it during model training.")

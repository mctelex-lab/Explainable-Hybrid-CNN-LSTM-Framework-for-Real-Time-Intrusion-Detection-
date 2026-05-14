import streamlit as st, json, os
from utils.helpers import inject_css
inject_css()
st.title("📈 Model Performance")
if os.path.exists("deployment/metadata.json"):
    with open("deployment/metadata.json") as f:
        meta = json.load(f)
    st.json(meta.get("evaluation_metrics", {}))
else:
    st.info("metadata.json not found.")

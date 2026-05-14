import streamlit as st, os, glob
from utils.helpers import inject_css
inject_css()
st.title("🧠 LIME Explainability")
files = sorted(glob.glob("lime_explanation_*.png"))
if files:
    for f in files[:5]:
        st.image(f, caption=f, use_container_width=True)
else:
    st.info("No LIME explanation images found.")

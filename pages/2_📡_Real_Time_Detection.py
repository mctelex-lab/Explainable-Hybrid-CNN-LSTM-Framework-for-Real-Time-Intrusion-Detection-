import streamlit as st, pandas as pd, numpy as np
from utils.helpers import inject_css, severity_label
inject_css()
st.title("📡 Real-Time Detection")
uploaded = st.file_uploader("Upload Network Flow CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    scores = np.random.rand(len(df))
    df["Threat Score"] = scores
    df["Severity"] = [severity_label(s) for s in scores]
    df["Prediction"] = np.where(scores >= 0.5, "ATTACK", "BENIGN")
    st.dataframe(df.head(50), use_container_width=True)
    st.download_button("Download Results", df.to_csv(index=False), "results.csv")

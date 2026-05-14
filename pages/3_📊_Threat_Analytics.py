import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from utils.helpers import inject_css
inject_css()
st.title("📊 Threat Analytics")
scores = np.random.rand(1000)
fig = px.histogram(x=scores, nbins=40, title="Threat Score Distribution")
st.plotly_chart(fig, use_container_width=True)

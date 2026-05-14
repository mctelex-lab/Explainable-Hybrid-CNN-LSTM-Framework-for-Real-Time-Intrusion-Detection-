import streamlit as st, psutil
from utils.helpers import inject_css
inject_css()
st.title("⚙️ System Monitor")
st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
st.metric("Memory Usage", f"{psutil.virtual_memory().percent}%")

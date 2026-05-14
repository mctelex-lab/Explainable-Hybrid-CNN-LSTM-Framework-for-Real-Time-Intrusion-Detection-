import streamlit as st

def inject_css():
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(135deg, #0f172a, #111827); color: white;}
    h1, h2, h3 {color: #38bdf8 !important;}
    </style>
    """, unsafe_allow_html=True)

def severity_label(score: float) -> str:
    if score < 0.30:
        return "Low"
    if score < 0.60:
        return "Medium"
    if score < 0.85:
        return "High"
    return "Critical"

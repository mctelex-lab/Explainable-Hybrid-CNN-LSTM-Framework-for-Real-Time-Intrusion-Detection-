# CyberGuard IDS

Multi-page Streamlit application for real-time intrusion detection using TensorFlow, SHAP, and LIME.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required Deployment Files
Create a `deployment/` folder containing:
- scaler.pkl
- imputer.pkl
- feature_names.json
- metadata.json
- model.weights.h5

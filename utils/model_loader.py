import os, json, joblib

def load_artifacts():
    result = {"model": None, "scaler": None, "imputer": None, "features": [], "meta": {}}
    deployment = "deployment"
    if not os.path.exists(deployment):
        return result
    try:
        if os.path.exists(f"{deployment}/scaler.pkl"):
            result["scaler"] = joblib.load(f"{deployment}/scaler.pkl")
        if os.path.exists(f"{deployment}/imputer.pkl"):
            result["imputer"] = joblib.load(f"{deployment}/imputer.pkl")
        if os.path.exists(f"{deployment}/feature_names.json"):
            with open(f"{deployment}/feature_names.json") as f:
                result["features"] = json.load(f)
        if os.path.exists(f"{deployment}/metadata.json"):
            with open(f"{deployment}/metadata.json") as f:
                result["meta"] = json.load(f)
        # Placeholder to indicate artifacts present even if TensorFlow weights are not loaded here.
        if os.path.exists(f"{deployment}/model.weights.h5"):
            result["model"] = "loaded"
    except Exception:
        pass
    return result

"""Load the saved Task 7 Decision Tree model and run leakage-safe inference."""
from __future__ import annotations
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "trained_model.joblib"


def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict_one(artifact: dict, values: dict) -> dict:
    predictors = artifact["predictors"]
    row = pd.DataFrame([{col: values[col] for col in predictors}])
    encoded = pd.get_dummies(row, drop_first=True)
    encoded = encoded.reindex(columns=artifact["model_columns"], fill_value=0)
    model = artifact["model"]
    predicted = int(model.predict(encoded)[0])
    probs = model.predict_proba(encoded)[0]
    classes = list(model.classes_)
    risk = float(probs[classes.index(0)]) if 0 in classes else 0.0
    completion = float(probs[classes.index(1)]) if 1 in classes else 0.0
    return {
        "predicted_class": "Completed" if predicted == 1 else "Not Completed",
        "risk_probability": risk,
        "completion_probability": completion,
        "model_version": artifact.get("model_version", "Task 7 Decision Tree"),
    }


def predict_frame(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    predictors = artifact["predictors"]
    x = pd.get_dummies(df[predictors], drop_first=True).reindex(columns=artifact["model_columns"], fill_value=0)
    probs = artifact["model"].predict_proba(x)
    classes = list(artifact["model"].classes_)
    risk_idx = classes.index(0) if 0 in classes else None
    risk = probs[:, risk_idx] if risk_idx is not None else [0.0] * len(df)
    out = df.copy()
    out["Predicted_Risk_Percent"] = (pd.Series(risk, index=out.index) * 100).round(1)
    out["Predicted_Outcome"] = artifact["model"].predict(x)
    out["Predicted_Outcome"] = out["Predicted_Outcome"].map({1: "Completed", 0: "Not Completed"})
    out["Review_Flag"] = out["Predicted_Risk_Percent"].apply(lambda x: "Review recommended" if x >= 65 else ("Check details" if x >= 40 else "Routine handling"))
    return out

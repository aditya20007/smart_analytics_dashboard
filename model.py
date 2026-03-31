"""
model.py
--------
Trains Linear Regression / Logistic Regression, Random Forest, and XGBoost
on any tabular dataset.  Returns metrics, feature importances, and a
callable predict function.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings("ignore")


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() <= 20 else None)


# ─── CLASSIFICATION ──────────────────────────────────────────────────────────

def train_classifiers(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train Logistic Regression, Random Forest Classifier, XGBoost Classifier.
    Returns a results dict per model.
    """
    X_train, X_test, y_train, y_test = _split(X, y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                              use_label_encoder=False, eval_metric="logloss",
                                              verbosity=0),
    }

    results = {}
    for name, model in models.items():
        X_tr = X_train_s if name == "Logistic Regression" else X_train
        X_te = X_test_s  if name == "Logistic Regression" else X_test

        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        acc   = accuracy_score(y_test, preds)

        fi = None
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        results[name] = {
            "model":              model,
            "scaler":             scaler if name == "Logistic Regression" else None,
            "accuracy":           round(acc * 100, 2),
            "report":             classification_report(y_test, preds, output_dict=True),
            "confusion_matrix":   confusion_matrix(y_test, preds),
            "feature_importance": fi,
            "X_test":             X_test,
            "y_test":             y_test,
            "y_pred":             preds,
        }

    return results


# ─── REGRESSION ──────────────────────────────────────────────────────────────

def train_regressors(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train Linear Regression, Random Forest Regressor, XGBoost Regressor.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost":           XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }

    results = {}
    for name, model in models.items():
        X_tr = X_train_s if name == "Linear Regression" else X_train
        X_te = X_test_s  if name == "Linear Regression" else X_test

        model.fit(X_tr, y_train)
        preds = model.predict(X_te)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)

        fi = None
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        elif hasattr(model, "coef_"):
            fi = pd.Series(np.abs(model.coef_), index=X.columns).sort_values(ascending=False)

        results[name] = {
            "model":              model,
            "scaler":             scaler if name == "Linear Regression" else None,
            "rmse":               round(rmse, 4),
            "mae":                round(mae, 4),
            "r2":                 round(r2, 4),
            "feature_importance": fi,
            "X_test":             X_test,
            "y_test":             y_test,
            "y_pred":             preds,
        }

    return results


# ─── BEST MODEL SELECTOR ─────────────────────────────────────────────────────

def best_model(results: dict, problem_type: str) -> str:
    """Return the name of the best-performing model."""
    if problem_type == "classification":
        return max(results, key=lambda k: results[k]["accuracy"])
    else:
        return max(results, key=lambda k: results[k]["r2"])


# ─── SINGLE-ROW PREDICTION ───────────────────────────────────────────────────

def predict_single(model_info: dict, input_df: pd.DataFrame) -> np.ndarray:
    """
    Use a trained model (+ optional scaler) to predict on a single row.
    input_df must have the same columns as the training features.
    """
    model  = model_info["model"]
    scaler = model_info.get("scaler")
    X = input_df.values
    if scaler is not None:
        X = scaler.transform(X)
    return model.predict(X)


# ─── FUTURE TREND (regression only) ──────────────────────────────────────────

def extrapolate_trend(y_pred: np.ndarray, steps: int = 10) -> np.ndarray:
    """
    Naive linear extrapolation of the model's test predictions.
    Useful for a 'future trend' chart when no time column is available.
    """
    x = np.arange(len(y_pred))
    coef = np.polyfit(x, y_pred, 1)
    future_x = np.arange(len(y_pred), len(y_pred) + steps)
    return np.polyval(coef, future_x)

"""
data_processing.py
------------------
Handles all data ingestion, cleaning, and EDA logic.
Keeps the Streamlit app clean by separating concerns.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ─── 1. LOADING ──────────────────────────────────────────────────────────────

import pandas as pd
import streamlit as st

def load_data(uploaded_file) -> pd.DataFrame:
    """Read CSV from a Streamlit UploadedFile object."""
    try:
        if uploaded_file is None:
            st.warning("Please upload a file.")
            return None

        uploaded_file.seek(0)  # Fix pointer issue

        df = pd.read_csv(uploaded_file, encoding='latin1')

        if df.empty:
            st.error("The uploaded file is empty.")
            return None

        st.success(f"File loaded successfully ✅ Shape: {df.shape}")
        return df

    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        return None

# ─── 2. VALIDATION ───────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Basic sanity checks.
    Returns a dict with 'valid' bool and 'issues' list.
    """
    issues = []
    if df.shape[0] < 10:
        issues.append("Dataset has fewer than 10 rows — too small for meaningful analysis.")
    if df.shape[1] < 2:
        issues.append("Dataset needs at least 2 columns.")
    all_missing = df.columns[df.isnull().all()].tolist()
    if all_missing:
        issues.append(f"Columns with ALL values missing: {all_missing}")
    return {"valid": len(issues) == 0, "issues": issues}


# ─── 3. CLEANING ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Automated cleaning pipeline.
    Returns cleaned df + a summary dict describing what was done.
    """
    report = {}
    original_shape = df.shape

    # 3a. Drop columns where >60 % values are missing
    thresh = 0.6
    high_null_cols = [c for c in df.columns if df[c].isnull().mean() > thresh]
    df = df.drop(columns=high_null_cols)
    report["dropped_high_null_cols"] = high_null_cols

    # 3b. Drop exact duplicate rows
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    report["dropped_duplicate_rows"] = int(dupes)

    # 3c. Fill remaining nulls
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
        else:
            fill_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

    report["null_fills"] = "numeric → median, categorical → mode"
    report["original_shape"] = original_shape
    report["cleaned_shape"] = df.shape

    return df, report


# ─── 4. COLUMN PROFILING ─────────────────────────────────────────────────────

def profile_columns(df: pd.DataFrame) -> dict:
    """Return per-column summary statistics."""
    profile = {}
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "nulls": int(df[col].isnull().sum()),
            "unique": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            })
        profile[col] = col_info
    return profile


# ─── 5. COLUMN CLASSIFICATION ────────────────────────────────────────────────

def classify_columns(df: pd.DataFrame) -> dict:
    """Split columns into numeric, categorical, datetime, and text buckets."""
    numeric_cols, categorical_cols, datetime_cols, text_cols = [], [], [], []

    for col in df.columns:
        # Try datetime detection
        if "date" in col.lower() or "time" in col.lower() or "timestamp" in col.lower():
            try:
                pd.to_datetime(df[col], errors="raise")
                datetime_cols.append(col)
                continue
            except Exception:
                pass

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif df[col].nunique() / len(df) > 0.5 and df[col].dtype == object:
            # High-cardinality string → treat as free text
            text_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "text": text_cols,
    }


# ─── 6. ENCODING (for ML) ────────────────────────────────────────────────────

def encode_for_ml(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all object/category columns so sklearn models can use them.
    Returns encoded df and a mapping of {col: LabelEncoder}.
    """
    df_enc = df.copy()
    encoders = {}
    for col in df_enc.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders


# ─── 7. PROBLEM TYPE DETECTION ───────────────────────────────────────────────

def detect_problem_type(series: pd.Series) -> str:
    """
    Given the target column, decide 'classification' or 'regression'.
    Heuristic: ≤20 unique values OR dtype is object → classification.
    """
    if series.dtype == object or series.nunique() <= 20:
        return "classification"
    return "regression"

"""
app.py
------
Smart Business Analytics Dashboard with AI Insights
Run with:  streamlit run app.py
"""

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── local modules ─────────────────────────────────────────────────────────────
from data_processing import (
    load_data, validate_dataframe, clean_data,
    classify_columns, encode_for_ml, detect_problem_type, profile_columns,
)
from model import (
    train_classifiers, train_regressors, best_model,
    predict_single, extrapolate_trend,
)
from utils import (
    plot_distributions, plot_correlation_heatmap, plot_categorical_counts,
    plot_trend, plot_boxplots, plot_feature_importance, plot_confusion_matrix,
    plot_actual_vs_predicted, plot_future_trend, generate_ai_insights, pairplot_bytes,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Smart Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ─── Fonts ─────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ─── Global background ─────────────────────────────────────────────────── */
.stApp { background: #090c12; }

/* ─── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #131920 100%);
    border-right: 1px solid #1e2a38;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ─── Metric cards ──────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 2rem !important; }
div[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.8rem !important; }

/* ─── Section headers ───────────────────────────────────────────────────── */
h1 { color: #e6edf3 !important; font-weight: 700 !important; }
h2, h3 { color: #c9d1d9 !important; font-weight: 600 !important; }

/* ─── Info / success / warning ──────────────────────────────────────────── */
.stAlert { border-radius: 10px !important; }

/* ─── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    color: #fff !important;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ─── Dataframe ─────────────────────────────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ─── Insight cards ─────────────────────────────────────────────────────── */
.insight-card {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-left: 4px solid #58a6ff;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #c9d1d9;
    line-height: 1.6;
}

/* ─── Divider ───────────────────────────────────────────────────────────── */
hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

for key in ["df_raw", "df_clean", "col_types", "clean_report",
            "target_col", "problem_type", "model_results", "encoders",
            "X_cols", "y_col", "df_encoded"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Smart Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home & Upload", "🔍 EDA & Analysis", "🤖 ML Models", "🔮 Predictions", "🧠 AI Insights"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean
        st.markdown("**📋 Dataset Info**")
        st.markdown(f"- Rows: **{df.shape[0]:,}**")
        st.markdown(f"- Cols: **{df.shape[1]}**")
        if st.session_state.target_col:
            st.markdown(f"- Target: **{st.session_state.target_col}**")
        if st.session_state.problem_type:
            emoji = "🔵" if st.session_state.problem_type == "classification" else "🟢"
            st.markdown(f"- Task: {emoji} **{st.session_state.problem_type.title()}**")
    st.markdown("---")
    st.caption("Built with ❤️ by Aditya · Powered by Streamlit + XGBoost")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME & UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home & Upload":
    st.title("📊 Smart Business Analytics Dashboard")
    st.markdown("##### *Upload any CSV → automated EDA → ML models → AI insights in seconds.*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔍 EDA", "Auto")
    col2.metric("🤖 Models", "3")
    col3.metric("🔮 Predictions", "Live")
    col4.metric("🧠 Insights", "AI")

    st.markdown("---")
    st.subheader("📂 Upload Your Dataset")

    uploaded = st.file_uploader(
        "Drop a CSV file here",
        type=["csv"],
        help="Make sure your CSV has a header row and at least 10 records.",
    )

    if uploaded:
        with st.spinner("Reading file…"):
            try:
                df_raw = load_data(uploaded)
            except ValueError as e:
                st.error(f"❌ {e}")
                st.stop()

        validation = validate_dataframe(df_raw)
        if not validation["valid"]:
            for issue in validation["issues"]:
                st.warning(f"⚠️ {issue}")

        # Automatic cleaning
        with st.spinner("Cleaning data…"):
            df_clean, clean_report = clean_data(df_raw)

        col_types = classify_columns(df_clean)
        st.session_state.df_raw     = df_raw
        st.session_state.df_clean   = df_clean
        st.session_state.col_types  = col_types
        st.session_state.clean_report = clean_report

        st.success(f"✅ Loaded **{df_raw.shape[0]:,}** rows × **{df_raw.shape[1]}** columns.")

        # Cleaning summary
        with st.expander("🧹 Cleaning Report", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Columns Dropped (>60% null)", len(clean_report["dropped_high_null_cols"]))
            c2.metric("Duplicate Rows Removed", clean_report["dropped_duplicate_rows"])
            c3.metric("Final Shape", f"{df_clean.shape[0]} × {df_clean.shape[1]}")
            if clean_report["dropped_high_null_cols"]:
                st.info(f"Dropped: {clean_report['dropped_high_null_cols']}")

        # Preview
        st.subheader("👀 Data Preview")
        st.dataframe(df_clean.head(20), use_container_width=True)

        # Target selection
        st.subheader("🎯 Select Target Column (for ML)")
        all_cols = df_clean.columns.tolist()
        target = st.selectbox("Which column do you want to predict?", ["— skip ML —"] + all_cols)

        if target != "— skip ML —":
            st.session_state.target_col   = target
            st.session_state.problem_type = detect_problem_type(df_clean[target])
            emoji = "🔵" if st.session_state.problem_type == "classification" else "🟢"
            st.info(f"{emoji} Detected task: **{st.session_state.problem_type.title()}**")

        st.markdown("---")
        st.success("✅ Navigate to **EDA & Analysis** in the sidebar to continue.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 EDA & Analysis":
    if st.session_state.df_clean is None:
        st.warning("⚠️ Please upload a dataset first (Home & Upload).")
        st.stop()

    df        = st.session_state.df_clean
    col_types = st.session_state.col_types
    numeric   = col_types["numeric"]
    cat       = col_types["categorical"]
    dt        = col_types["datetime"]

    st.title("🔍 Exploratory Data Analysis")
    st.markdown("---")

    # ── Summary stats ──
    st.subheader("📋 Summary Statistics")
    if numeric:
        st.dataframe(df[numeric].describe().T.round(4), use_container_width=True)
    else:
        st.info("No numeric columns detected.")

    # ── Column profile ──
    with st.expander("🔬 Column Profile", expanded=False):
        profile = profile_columns(df)
        prof_df = pd.DataFrame(profile).T
        st.dataframe(prof_df, use_container_width=True)

    st.markdown("---")

    # ── Distributions ──
    st.subheader("📊 Feature Distributions")
    if numeric:
        st.plotly_chart(plot_distributions(df, numeric), use_container_width=True)
    else:
        st.info("No numeric features to plot.")

    # ── Box plots ──
    if numeric:
        st.subheader("📦 Box Plots")
        st.plotly_chart(plot_boxplots(df, numeric), use_container_width=True)

    # ── Correlation heatmap ──
    st.subheader("🔥 Correlation Heatmap")
    if len(numeric) >= 2:
        st.plotly_chart(plot_correlation_heatmap(df, numeric), use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for a correlation heatmap.")

    # ── Categorical counts ──
    if cat:
        st.subheader("🗂️ Categorical Feature Counts")
        st.plotly_chart(plot_categorical_counts(df, cat), use_container_width=True)

    # ── Time-series trend ──
    if dt and numeric:
        st.subheader("📈 Trend Analysis")
        dt_col  = st.selectbox("Datetime column", dt)
        num_col = st.selectbox("Numeric column to trend", numeric)
        st.plotly_chart(plot_trend(df, dt_col, num_col), use_container_width=True)

    # ── Pair plot ──
    if len(numeric) >= 2:
        st.subheader("🔗 Pair Plot")
        hue_opts = [None] + [c for c in cat if df[c].nunique() <= 10]
        hue = st.selectbox("Color by (optional)", hue_opts)
        with st.spinner("Rendering pair plot…"):
            img_bytes = pairplot_bytes(df, numeric, hue=hue)
        st.image(img_bytes, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 ML Models":
    if st.session_state.df_clean is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()
    if not st.session_state.target_col:
        st.warning("⚠️ Please select a target column on the Upload page.")
        st.stop()

    df           = st.session_state.df_clean
    target_col   = st.session_state.target_col
    problem_type = st.session_state.problem_type
    col_types    = st.session_state.col_types

    st.title("🤖 Machine Learning Models")
    st.markdown(f"**Task**: {problem_type.title()} | **Target**: `{target_col}`")
    st.markdown("---")

    # Prepare features
    drop_cols = col_types["datetime"] + col_types["text"] + [target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    df_enc, encoders = encode_for_ml(df[feature_cols + [target_col]])
    X = df_enc[feature_cols]
    y = df_enc[target_col]

    st.session_state.encoders    = encoders
    st.session_state.X_cols      = feature_cols
    st.session_state.df_encoded  = df_enc

    # Train button
    if st.button("🚀 Train All Models"):
        with st.spinner("Training models… this may take 30–60 seconds for large datasets."):
            if problem_type == "classification":
                results = train_classifiers(X, y)
            else:
                results = train_regressors(X, y)

        st.session_state.model_results = results
        st.success("✅ Training complete!")

    if st.session_state.model_results is None:
        st.info("Click **Train All Models** to begin.")
        st.stop()

    results = st.session_state.model_results

    # ── Metrics summary ──
    st.subheader("📊 Model Performance Summary")
    if problem_type == "classification":
        cols = st.columns(len(results))
        for i, (name, res) in enumerate(results.items()):
            cols[i].metric(name, f"{res['accuracy']}%", "Accuracy")
    else:
        cols = st.columns(len(results))
        for i, (name, res) in enumerate(results.items()):
            cols[i].metric(name, f"R²={res['r2']}", f"RMSE={res['rmse']}")

    # Best model badge
    best = best_model(results, problem_type)
    st.success(f"🏆 Best model: **{best}**")

    # ── Per-model detail ──
    st.subheader("🔎 Model Details")
    selected_model = st.selectbox("Inspect model", list(results.keys()))
    res = results[selected_model]

    if problem_type == "classification":
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{res['accuracy']}%")
        report_df = pd.DataFrame(res["report"]).T.round(3)
        c2.metric("Precision (macro)", f"{report_df.loc['macro avg','precision']:.3f}")
        c3.metric("Recall (macro)",    f"{report_df.loc['macro avg','recall']:.3f}")

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Confusion Matrix")
            labels = [str(l) for l in sorted(res["y_test"].unique())]
            st.plotly_chart(plot_confusion_matrix(res["confusion_matrix"], labels),
                            use_container_width=True)
        with col_right:
            if res["feature_importance"] is not None:
                st.subheader("Feature Importance")
                st.plotly_chart(plot_feature_importance(res["feature_importance"],
                                                         selected_model),
                                use_container_width=True)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("R²",   res["r2"])
        c2.metric("RMSE", res["rmse"])
        c3.metric("MAE",  res["mae"])

        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(plot_actual_vs_predicted(res["y_test"], res["y_pred"]),
                            use_container_width=True)
        with col_right:
            if res["feature_importance"] is not None:
                st.plotly_chart(plot_feature_importance(res["feature_importance"],
                                                         selected_model),
                                use_container_width=True)

        # Future trend
        st.subheader("🔮 Future Trend Forecast")
        steps = st.slider("Forecast steps ahead", 5, 50, 10)
        future = extrapolate_trend(res["y_pred"], steps=steps)
        st.plotly_chart(plot_future_trend(res["y_pred"], future), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Predictions":
    if st.session_state.model_results is None:
        st.warning("⚠️ Train models first (ML Models page).")
        st.stop()

    df           = st.session_state.df_clean
    target_col   = st.session_state.target_col
    problem_type = st.session_state.problem_type
    col_types    = st.session_state.col_types
    feature_cols = st.session_state.X_cols
    encoders     = st.session_state.encoders
    results      = st.session_state.model_results
    df_enc       = st.session_state.df_encoded

    st.title("🔮 Live Predictions")
    st.markdown(f"**Target**: `{target_col}` | **Task**: {problem_type.title()}")
    st.markdown("---")

    # Model selector
    best = best_model(results, problem_type)
    model_name = st.selectbox("Choose model for prediction", list(results.keys()), index=list(results.keys()).index(best))
    model_info = results[model_name]

    st.subheader("✏️ Enter Feature Values")
    st.markdown("Adjust the sliders/inputs below then click **Predict**.")

    user_input = {}
    col_batch = st.columns(3)
    for i, col in enumerate(feature_cols):
        with col_batch[i % 3]:
            if col in encoders:
                # Show original labels
                le     = encoders[col]
                labels = list(le.classes_)
                choice = st.selectbox(col, labels, key=f"pred_{col}")
                user_input[col] = le.transform([choice])[0]
            elif pd.api.types.is_float_dtype(df_enc[col]):
                mn = float(df_enc[col].min())
                mx = float(df_enc[col].max())
                med = float(df_enc[col].median())
                user_input[col] = st.slider(col, mn, mx, med, key=f"pred_{col}")
            else:
                mn = int(df_enc[col].min())
                mx = int(df_enc[col].max())
                med = int(df_enc[col].median())
                user_input[col] = st.number_input(col, mn, mx, med, key=f"pred_{col}")

    if st.button("⚡ Predict"):
        input_df = pd.DataFrame([user_input])
        pred = predict_single(model_info, input_df)[0]

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        if problem_type == "classification" and target_col in encoders:
            le = encoders[target_col]
            label = le.inverse_transform([int(round(pred))])[0]
            st.success(f"**Predicted Class**: `{label}`")
        else:
            st.success(f"**Predicted Value**: `{pred:.4f}`")

        # Confidence (classification only — use predict_proba if available)
        mdl = model_info["model"]
        scaler = model_info.get("scaler")
        X_in = input_df.values
        if scaler:
            X_in = scaler.transform(X_in)
        if problem_type == "classification" and hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(X_in)[0]
            st.markdown("**Class Probabilities**")
            proba_df = pd.DataFrame({
                "Class": [str(c) for c in mdl.classes_],
                "Probability": [f"{p*100:.1f}%" for p in proba],
            })
            st.dataframe(proba_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AI INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🧠 AI Insights":
    if st.session_state.df_clean is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    st.title("🧠 AI-Generated Insights")
    st.markdown("Smart, plain-English summaries generated from your data and model results.")
    st.markdown("---")

    with st.spinner("Generating insights…"):
        insights = generate_ai_insights(
            df          = st.session_state.df_clean,
            col_types   = st.session_state.col_types,
            target_col  = st.session_state.target_col,
            model_results = st.session_state.model_results,
            problem_type  = st.session_state.problem_type,
        )

    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📤 Export Insights")
    report_text = "\n\n".join(insights)
    # Strip markdown bold markers for plain text export
    report_plain = report_text.replace("**", "").replace("*", "")
    st.download_button(
        "⬇️ Download Insights as TXT",
        data=report_plain,
        file_name="ai_insights.txt",
        mime="text/plain",
    )

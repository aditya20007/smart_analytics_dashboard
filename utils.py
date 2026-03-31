"""
utils.py
--------
Visualization helpers (Plotly / Seaborn) and the AI-Insights engine.
All plot functions return a Plotly Figure so Streamlit can render them
with st.plotly_chart(fig, use_container_width=True).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io


# ─── COLOUR PALETTE ──────────────────────────────────────────────────────────

PALETTE   = px.colors.qualitative.Bold
BG_DARK   = "#0f1117"
GRID_COL  = "#2d2d2d"
TEXT_COL  = "#e0e0e0"

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_DARK,
    font=dict(color=TEXT_COL, family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
    yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
)


def _apply_layout(fig, title=""):
    fig.update_layout(title=dict(text=title, font=dict(size=16)), **LAYOUT_DEFAULTS)
    return fig


# ─── 1. DISTRIBUTION PLOTS ───────────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame, numeric_cols: list, max_cols: int = 6) -> go.Figure:
    cols = numeric_cols[:max_cols]
    n    = len(cols)
    if n == 0:
        return go.Figure()
    rows = (n + 2) // 3
    fig  = make_subplots(rows=rows, cols=3, subplot_titles=cols)
    for i, col in enumerate(cols):
        r, c = divmod(i, 3)
        trace = go.Histogram(
            x=df[col], name=col, showlegend=False,
            marker_color=PALETTE[i % len(PALETTE)], opacity=0.8,
        )
        fig.add_trace(trace, row=r + 1, col=c + 1)
    fig.update_layout(height=280 * rows, **LAYOUT_DEFAULTS, title_text="📊 Feature Distributions")
    return fig


# ─── 2. CORRELATION HEATMAP ──────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> go.Figure:
    if len(numeric_cols) < 2:
        return go.Figure()
    corr = df[numeric_cols].corr()
    fig  = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10),
            showscale=True,
        )
    )
    _apply_layout(fig, "🔥 Correlation Heatmap")
    fig.update_layout(height=500)
    return fig


# ─── 3. CATEGORICAL BAR PLOTS ────────────────────────────────────────────────

def plot_categorical_counts(df: pd.DataFrame, cat_cols: list, max_cols: int = 4) -> go.Figure:
    cols = [c for c in cat_cols if df[c].nunique() <= 20][:max_cols]
    if not cols:
        return go.Figure()
    rows = (len(cols) + 1) // 2
    fig  = make_subplots(rows=rows, cols=2, subplot_titles=cols)
    for i, col in enumerate(cols):
        r, c = divmod(i, 2)
        vc   = df[col].value_counts().head(15)
        fig.add_trace(
            go.Bar(x=vc.index.tolist(), y=vc.values, name=col, showlegend=False,
                   marker_color=PALETTE[i % len(PALETTE)]),
            row=r + 1, col=c + 1,
        )
    fig.update_layout(height=350 * rows, **LAYOUT_DEFAULTS, title_text="📊 Categorical Distributions")
    return fig


# ─── 4. TREND / TIME-SERIES PLOT ─────────────────────────────────────────────

def plot_trend(df: pd.DataFrame, datetime_col: str, numeric_col: str) -> go.Figure:
    tmp = df[[datetime_col, numeric_col]].copy()
    tmp[datetime_col] = pd.to_datetime(tmp[datetime_col], errors="coerce")
    tmp = tmp.dropna().sort_values(datetime_col)
    tmp_grouped = tmp.groupby(datetime_col)[numeric_col].mean().reset_index()
    fig = px.line(tmp_grouped, x=datetime_col, y=numeric_col,
                  title=f"📈 Trend: {numeric_col} over time",
                  color_discrete_sequence=[PALETTE[0]])
    _apply_layout(fig, f"📈 Trend: {numeric_col} over time")
    return fig


# ─── 5. BOX PLOTS ────────────────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame, numeric_cols: list, max_cols: int = 6) -> go.Figure:
    cols = numeric_cols[:max_cols]
    fig  = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Box(y=df[col], name=col, marker_color=PALETTE[i % len(PALETTE)]))
    _apply_layout(fig, "📦 Box Plots — Spread & Outliers")
    return fig


# ─── 6. FEATURE IMPORTANCE ───────────────────────────────────────────────────

def plot_feature_importance(fi_series: pd.Series, title: str = "Feature Importance") -> go.Figure:
    top = fi_series.head(15).sort_values()
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index.tolist(), orientation="h",
        marker=dict(color=top.values, colorscale="Viridis"),
    ))
    _apply_layout(fig, f"🏆 {title}")
    return fig


# ─── 7. CONFUSION MATRIX ─────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, labels: list = None) -> go.Figure:
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues", text=cm, texttemplate="%{text}",
    ))
    _apply_layout(fig, "🎯 Confusion Matrix")
    return fig


# ─── 8. ACTUAL vs PREDICTED ──────────────────────────────────────────────────

def plot_actual_vs_predicted(y_test, y_pred) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                             marker=dict(color=PALETTE[1], opacity=0.6), name="Predictions"))
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                             line=dict(color="white", dash="dash"), name="Perfect Fit"))
    _apply_layout(fig, "🎯 Actual vs Predicted")
    return fig


# ─── 9. FUTURE TREND CHART ───────────────────────────────────────────────────

def plot_future_trend(y_pred: np.ndarray, y_future: np.ndarray) -> go.Figure:
    n = len(y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n)), y=y_pred,
                             mode="lines", name="Test Predictions", line=dict(color=PALETTE[0])))
    fig.add_trace(go.Scatter(x=list(range(n, n + len(y_future))), y=y_future,
                             mode="lines+markers", name="Future Forecast",
                             line=dict(color=PALETTE[2], dash="dot")))
    _apply_layout(fig, "🔮 Future Trend Forecast")
    return fig


# ─── 10. AI INSIGHTS ENGINE ──────────────────────────────────────────────────

def generate_ai_insights(df: pd.DataFrame, col_types: dict, target_col: str | None,
                          model_results: dict | None, problem_type: str | None) -> list[str]:
    """
    Rule-based insight generator that produces human-readable bullets.
    Returns a list of strings.
    """
    insights = []
    numeric = col_types.get("numeric", [])
    cat     = col_types.get("categorical", [])

    # Dataset overview
    insights.append(
        f"📋 **Dataset Overview**: Your data has **{df.shape[0]:,} rows** and "
        f"**{df.shape[1]} columns**, giving a solid foundation for analysis."
    )

    # Missing data
    miss_pct = df.isnull().mean().mean() * 100
    if miss_pct < 1:
        insights.append("✅ **Data Quality**: Excellent — less than 1 % of values are missing.")
    elif miss_pct < 10:
        insights.append(f"⚠️ **Data Quality**: Moderate missing data ({miss_pct:.1f} %). "
                        "Median/mode imputation has been applied automatically.")
    else:
        insights.append(f"🚨 **Data Quality**: High missingness ({miss_pct:.1f} %). "
                        "Interpret results with caution and consider collecting more complete data.")

    # Numeric correlations with target
    if target_col and target_col in df.columns and numeric:
        try:
            num_df = df[numeric].copy()
            if target_col in num_df.columns:
                corr_with_target = num_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
                top_feat = corr_with_target.index[0]
                top_corr = corr_with_target.iloc[0]
                insights.append(
                    f"🔗 **Top Predictor**: **{top_feat}** has the strongest linear relationship "
                    f"with **{target_col}** (|correlation| = {top_corr:.2f})."
                )
        except Exception:
            pass

    # Outlier detection (IQR)
    outlier_cols = []
    for col in numeric:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        if n_out / len(df) > 0.05:
            outlier_cols.append(f"{col} ({n_out} outliers)")
    if outlier_cols:
        insights.append(
            f"📌 **Outliers Detected**: {', '.join(outlier_cols[:3])}. "
            "Consider robust models (Random Forest / XGBoost) which handle outliers better than linear models."
        )

    # Skewness
    for col in numeric[:5]:
        skew = df[col].skew()
        if abs(skew) > 1:
            direction = "right (positive)" if skew > 0 else "left (negative)"
            insights.append(
                f"📐 **Skewed Feature**: **{col}** is heavily skewed {direction} (skew = {skew:.2f}). "
                "Log-transformation might improve model performance."
            )

    # Class imbalance (classification)
    if target_col and target_col in df.columns and problem_type == "classification":
        vc = df[target_col].value_counts(normalize=True)
        minority = vc.iloc[-1]
        if minority < 0.2:
            insights.append(
                f"⚖️ **Class Imbalance**: The minority class in **{target_col}** represents only "
                f"{minority * 100:.1f} % of records. Consider SMOTE or class-weight balancing."
            )
        else:
            insights.append(
                f"✅ **Balanced Classes**: **{target_col}** is reasonably balanced "
                f"(minority = {minority * 100:.1f} %)."
            )

    # Model performance commentary
    if model_results:
        if problem_type == "classification":
            best_name = max(model_results, key=lambda k: model_results[k]["accuracy"])
            best_acc  = model_results[best_name]["accuracy"]
            if best_acc >= 90:
                grade = "🏆 Excellent"
            elif best_acc >= 75:
                grade = "✅ Good"
            elif best_acc >= 60:
                grade = "⚠️ Moderate"
            else:
                grade = "🚨 Needs Improvement"
            insights.append(
                f"🤖 **Best Model**: **{best_name}** achieved **{best_acc} % accuracy** — {grade}. "
                f"Consider tuning hyperparameters for further gains."
            )
        else:
            best_name = max(model_results, key=lambda k: model_results[k]["r2"])
            best_r2   = model_results[best_name]["r2"]
            insights.append(
                f"🤖 **Best Model**: **{best_name}** explains **{best_r2 * 100:.1f} %** of variance "
                f"in the target (R² = {best_r2:.3f})."
            )

        # Feature importance insight
        for name, res in model_results.items():
            fi = res.get("feature_importance")
            if fi is not None and len(fi) > 0:
                top3 = fi.head(3).index.tolist()
                insights.append(
                    f"💡 **Key Drivers** ({name}): The top 3 features influencing predictions are "
                    f"**{top3[0]}**, **{top3[1] if len(top3) > 1 else 'N/A'}**, and "
                    f"**{top3[2] if len(top3) > 2 else 'N/A'}**."
                )
                break

    # General recommendation
    insights.append(
        "🚀 **Next Steps**: Validate these results with domain experts, collect more data if R² < 0.7, "
        "and consider deploying the best model via a REST API (FastAPI) for production use."
    )

    return insights


# ─── 11. SEABORN PAIRPLOT (returns bytes for st.image) ───────────────────────

def pairplot_bytes(df: pd.DataFrame, numeric_cols: list, hue: str | None = None,
                   max_features: int = 4) -> bytes:
    cols = numeric_cols[:max_features]
    plot_df = df[cols + ([hue] if hue and hue not in cols else [])].copy()
    plt.style.use("dark_background")
    g = sns.pairplot(plot_df, hue=hue if hue in plot_df.columns else None,
                     palette="husl", plot_kws={"alpha": 0.5})
    g.figure.suptitle("Pair Plot", y=1.02, color="white")
    buf = io.BytesIO()
    g.figure.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                     facecolor="#0f1117")
    plt.close("all")
    buf.seek(0)
    return buf.read()

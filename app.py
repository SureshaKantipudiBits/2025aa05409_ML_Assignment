# =====================================================
# ML MODEL EVALUATOR - FINAL PROFESSIONAL VERSION
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
)

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="ML Model Evaluator",
    layout="wide",
    page_icon="📊"
)

# -----------------------------------------------------
# PROFESSIONAL UI STYLING (Font + Colors + Tabs)
# -----------------------------------------------------
st.markdown("""
<style>

/* Background */
.main {
    background-color: #f4f6f9;
}

/* Main Title */
h1 {
    font-size: 56px !important;
    font-weight: 700 !important;
    color: #0d47a1;
}

/* Subheadings */
h2, h3 {
    font-size: 32px !important;
    font-weight: 600 !important;
    color: #1a237e;
}

/* General Text */
p, div, span, label {
    font-size: 20px !important;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

/* Metric Label */
div[data-testid="metric-container"] label {
    font-size: 22px !important;
    font-weight: 600 !important;
    color: #37474f !important;
}

/* Metric Value */
div[data-testid="metric-container"] div {
    font-size: 42px !important;
    font-weight: 700 !important;
    color: #0d47a1 !important;
}

/* Tabs Styling */
button[role="tab"] {
    font-size: 22px !important;
    font-weight: 600 !important;
    color: #37474f !important;
}

button[role="tab"][aria-selected="true"] {
    color: white !important;
    background-color: #1976d2 !important;
    border-radius: 8px;
}

/* Success Message */
.stSuccess {
    font-size: 22px !important;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# HEADER
# -----------------------------------------------------
# Display BITS logo and student info
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊 Machine Learning Model Evaluator")
    st.caption("Interactive classification model comparison dashboard")
with col2:
    st.image("Data/BITS_logo.png", width=400)
    st.markdown("""
    <div style='background-color:#e8f5e9; padding:12px; border-radius:8px; border-left:4px solid #4caf50; margin-top:10px;'>
        <p style='margin:0; font-size:16px;'><b>Assignment 2</b></p>
        <p style='margin:0; font-size:15px;'>Suresha Kantipudi</p>
        <p style='margin:0; font-size:15px;'>2025aa05409</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

models = {
    "Logistic Regression": load_model("model/logistic_regression.pkl"),
    "Decision Tree": load_model("model/decision_tree_classifier.pkl"),
    "KNN": load_model("model/k-nearest_neighbor_classifier.pkl"),
    "Naive Bayes": load_model("model/naive_bayes_classifier.pkl"),
    "Random Forest": load_model("model/random_forest_classifier.pkl"),
    "XGBoost": load_model("model/xgboost_classifier.pkl"),
}

# -----------------------------------------------------
# LAYOUT
# -----------------------------------------------------
left_col, right_col = st.columns([1, 2])

# =====================================================
# LEFT PANEL - INPUTS
# =====================================================
with left_col:
    with st.container(border=True):

        st.subheader("📥 Input Configuration")

        # -----------------------------
        # Download test.csv
        # -----------------------------
        st.markdown("#### 📄 Sample Test File")

        try:
            with open("Data/test.csv", "rb") as f:
                test_data = f.read()

            st.download_button(
                label="⬇️ Download test.csv",
                data=test_data,
                file_name="test.csv",
                mime="text/csv",
                width="stretch"
            )

        except FileNotFoundError:
            st.warning("Sample test.csv not found in Data folder.")

        st.divider()

        # Upload CSV
        uploaded_file = st.file_uploader("Upload Test CSV", type="csv")

        # Model Selection
        selected_model_name = st.selectbox(
            "Select Model",
            list(models.keys())
        )

        model_info = {
            "Logistic Regression": "Linear classifier",
            "Decision Tree": "Tree-based classifier",
            "KNN": "Instance-based classifier",
            "Naive Bayes": "Probabilistic classifier",
            "Random Forest": "Ensemble of trees",
            "XGBoost": "Gradient boosting ensemble"
        }

        st.info(model_info[selected_model_name])

        run_button = st.button(
            "🚀 Run Prediction",
            width="stretch",
            type="primary"
        )

# =====================================================
# RIGHT PANEL - RESULTS
# =====================================================
with right_col:

    if run_button and uploaded_file is not None:

        with st.spinner("Running model prediction..."):

            df = pd.read_csv(uploaded_file)

            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            model = models[selected_model_name]
            y_pred = model.predict(X)

            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]

            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
            mcc = matthews_corrcoef(y, y_pred)

            auc_score = None
            if y_prob is not None and len(np.unique(y)) >= 2:
                auc_score = roc_auc_score(y, y_prob)

        st.success("Model evaluation completed successfully.")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Metrics", "📋 Classification Report", "📌 Confusion Matrix & ROC", "📁 Predictions"]
        )

        # ---------------- METRICS ----------------
        with tab1:
            m1, m2, m3 = st.columns(3)
            m4, m5, m6 = st.columns(3)

            m1.metric("Accuracy", f"{accuracy:.4f}")
            m2.metric("Precision", f"{precision:.4f}")
            m3.metric("Recall", f"{recall:.4f}")
            m4.metric("F1 Score", f"{f1:.4f}")
            m5.metric("MCC", f"{mcc:.4f}")
            m6.metric("AUC", f"{auc_score:.4f}" if auc_score else "N/A")

        # ---------------- CLASSIFICATION REPORT ----------------
        with tab2:
            report_dict = classification_report(
                y, y_pred,
                output_dict=True,
                zero_division=0
            )
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format("{:.3f}"), width="stretch")

        # ---------------- CONFUSION MATRIX & ROC ----------------
        with tab3:

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Confusion Matrix")

                cm = confusion_matrix(y, y_pred)

                fig, ax = plt.subplots()
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    annot_kws={"size": 16},
                    ax=ax
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with col2:
                if y_prob is not None and len(np.unique(y)) >= 2:
                    st.subheader("ROC Curve")

                    fpr, tpr, _ = roc_curve(y, y_prob)

                    fig2, ax2 = plt.subplots()
                    ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
                    ax2.plot([0,1], [0,1], linestyle="--")
                    ax2.set_xlabel("False Positive Rate")
                    ax2.set_ylabel("True Positive Rate")
                    ax2.legend()
                    st.pyplot(fig2)

        # ---------------- PREDICTIONS ----------------
        with tab4:
            result_df = X.copy()
            result_df["Prediction"] = y_pred

            st.dataframe(result_df.head(), width="stretch")

            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Full Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
                width="stretch"
            )

            if selected_model_name in ["Random Forest", "XGBoost"]:
                st.subheader("Top 10 Feature Importances")

                importances = model.feature_importances_
                feat_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": importances
                }).sort_values("Importance", ascending=False).head(10)

                st.bar_chart(feat_df.set_index("Feature"))

    else:
        st.markdown("""
        <div style='padding:20px; border-radius:10px; background-color:#eef2f7;'>
        <h4>📂 Awaiting Input</h4>
        Upload a dataset and click <b>Run Prediction</b> to view results.
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.divider()
st.caption("Developed by Suresha Kantipudi | BITS Pilani | ML Assignment 2")

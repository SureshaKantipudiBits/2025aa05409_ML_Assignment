
import pickle
from pathlib import Path
import io
import csv
import numpy as np

import streamlit as st
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score, # Added AUC Score import
    matthews_corrcoef, # Added MCC Score import
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# -------------------------
# Helpers
# -------------------------
@st.cache_resource
def load_model(model_path: str):
    """Loads a machine learning model from a pickle file."""
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def resolve_model_path(path_str: str) -> str:
    """
    Tries the given path first (e.g., '/content/..' from Colab).
    If it doesn't exist, falls back to a file with the same name in the app folder.
    """
    p = Path(path_str)
    if p.exists():
        return str(p)

    app_dir = Path(__file__).resolve().parent
    fallback = app_dir / p.name
    return str(fallback)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Reads an uploaded CSV robustly (auto-detects delimiter)."""
    raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")

    # First try comma. If it looks like a single-column file, try auto-sniff / semicolon.
    df = pd.read_csv(io.StringIO(raw))
    if df.shape[1] >= 2:
        return df

    # Try sniffer with common delimiters
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=[",", ";", "\t", "|"])
        df2 = pd.read_csv(io.StringIO(raw), sep=dialect.delimiter)
        if df2.shape[1] >= 2:
            return df2
    except Exception:
        pass

    # Final fallback: semicolon (common for bank-full.csv)
    return pd.read_csv(io.StringIO(raw), sep=";")


def to_1d_int(arr) -> np.ndarray:
    """Ensures 1D integer numpy array."""
    return np.asarray(arr).astype(int).ravel()


# -------------------------
# Models
# -------------------------
models = {
    "Decision Tree Classifier": load_model(resolve_model_path("Model/decision_tree_classifier.pkl")),
    "Random Forest Classifier": load_model(resolve_model_path("Model/random_forest_classifier.pkl")),
    "Naive Bayes Classifier": load_model(resolve_model_path("Model/naive_bayes_classifier.pkl")),
    "XGBoost Classifier": load_model(resolve_model_path("Model/xgboost_classifier.pkl")),
    "K-Nearest Neighbor Classifier": load_model(resolve_model_path("Model/k-nearest_neighbor_classifier.pkl")),
    "Logistic Regression": load_model(resolve_model_path("Model/logistic_regression.pkl")),
}

# -------------------------
# Streamlit UI
# -------------------------
st.title("Machine Learning Model Evaluator")
st.write("Upload a CSV file, select a pre-trained model, and evaluate its performance.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

model_names = list(models.keys())
selected_model_name = st.selectbox("Select a Model for Prediction", model_names)

run_prediction_button = st.button("Run Prediction")

st.info("Once you've uploaded your data and selected a model, click 'Run Prediction' to see the results.")


# -------------------------
# Session-state to avoid losing results on widget changes (e.g., radio buttons)
# -------------------------
current_signature = (
    uploaded_file.name if uploaded_file is not None else None,
    getattr(uploaded_file, "size", None) if uploaded_file is not None else None,
    selected_model_name,
)
if st.session_state.get("signature") != current_signature:
    st.session_state["signature"] = current_signature
    st.session_state["results"] = None


# -------------------------
# Prediction / Evaluation
# -------------------------
if run_prediction_button:
    if uploaded_file is None:
        st.warning("Please upload a CSV file to run the prediction and evaluation.")
    else:
        try:
            st.subheader("Data Processing and Model Evaluation")

            # Read CSV robustly
            dataframe = read_uploaded_csv(uploaded_file)

            if dataframe.empty:
                st.warning("Uploaded CSV file is empty. Please upload a valid file.")
                st.stop()

            if dataframe.shape[1] < 2:
                st.warning(
                    "The uploaded CSV file was read as a single column. "
                    "This usually means the delimiter is not a comma. "
                    "Try saving as comma-separated, or use a semicolon-separated file (e.g., bank-full.csv)."
                )
                st.stop()

            # --- Feature Engineering Steps --- #
            # 1) Target Variable Conversion (assumed last column)
            last_col = dataframe.columns[-1]
            if dataframe[last_col].dtype == "object":
                uniq = set(map(str, dataframe[last_col].dropna().unique()))
                if uniq.issubset({"yes", "no"}):
                    dataframe[last_col] = dataframe[last_col].map({"yes": 1, "no": 0})
                    st.write("Target variable converted from 'yes'/'no' to 1/0.")

            # 2) One-hot encode remaining categorical columns (excluding target if it's still object)
            #    (We treat the LAST column as target)
            feature_df = dataframe.iloc[:, :-1]
            target_series = dataframe.iloc[:, -1]

            categorical_cols = feature_df.select_dtypes(include="object").columns.tolist()
            if categorical_cols:
                feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=True)
                st.write(f"Categorical columns {categorical_cols} one-hot encoded.")

            X = feature_df.copy()
            y = target_series.copy()

            if X.shape[0] == 0 or X.shape[1] == 0:
                st.warning(
                    "No usable feature columns found after preprocessing. "
                    "Please ensure your CSV contains feature columns + a target column."
                )
                st.stop()

            # Keep a copy of X before scaling for display alongside predictions
            X_display = X.copy()

            # 3) Feature Scaling for numerical columns
            scaler = StandardScaler()
            numerical_cols_to_scale = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols_to_scale) > 0:
                X[numerical_cols_to_scale] = scaler.fit_transform(X[numerical_cols_to_scale])
                st.write("Numerical features scaled using StandardScaler.")
            else:
                st.write("No numerical features found for scaling.")

            st.write("Data loaded and preprocessed successfully. Features (X) and Target (y) separated.")

            st.subheader("Uploaded Data (after preprocessing)")
            st.dataframe(pd.concat([X_display, y.rename("target")], axis=1))

            # Retrieve the selected model
            model = models[selected_model_name]
            st.write(f"Making predictions using: **{selected_model_name}**")

            # Predict
            y_pred = model.predict(X)

            # Ensure correct shapes/types
            y_true_int = to_1d_int(y)
            y_pred_int = to_1d_int(y_pred)

            # Save results so UI widgets (radio) don't erase them on rerun
            st.session_state["results"] = {
                "X_display": X_display,
                "y_true": y_true_int,
                "y_pred": y_pred_int,
            }

        except Exception as e:
            st.error(f"An error occurred during data processing or prediction: {e}")
            st.warning("Please ensure your CSV file is properly formatted with numerical features and a target column.")
            st.session_state["results"] = None


# -------------------------
# Render results (persists across reruns)
# -------------------------
results = st.session_state.get("results")
if results is not None:
    X_display = results["X_display"]
    y_true_int = results["y_true"]
    y_pred_int = results["y_pred"]

    predicted_output_df = X_display.copy()
    predicted_output_df["Prediction"] = y_pred_int
    st.subheader("Predicted Data with Input Features")
    st.dataframe(predicted_output_df)

    # Add download button for predicted output
    csv_output = predicted_output_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predicted Output as CSV",
        data=csv_output,
        file_name="predicted_output.csv",
        mime="text/csv",
    )

    st.subheader("Classification Metrics")
    accuracy = accuracy_score(y_true_int, y_pred_int)
    # Only calculate AUC if there are at least two unique classes
    if len(np.unique(y_true_int)) >= 2:
        auc_score = roc_auc_score(y_true_int, y_pred_int)
        st.write(f"- **AUC Score**: {auc_score:.4f}")
    else:
        st.write("- **AUC Score**: N/A (requires at least two classes)")
    precision = precision_score(y_true_int, y_pred_int, average="weighted", zero_division=0)
    recall = recall_score(y_true_int, y_pred_int, average="weighted", zero_division=0)
    f1 = f1_score(y_true_int, y_pred_int, average="weighted", zero_division=0)
    mcc_score = matthews_corrcoef(y_true_int, y_pred_int)

    st.write(f"- **Accuracy**: {accuracy:.4f}")
    st.write(f"- **Precision (weighted)**: {precision:.4f}")
    st.write(f"- **Recall (weighted)**: {recall:.4f}")
    st.write(f"- **F1-Score (weighted)**: {f1:.4f}")
    st.write(f"- **Matthews Correlation Coefficient (MCC)**: {mcc_score:.4f}")

    st.subheader("Detailed Evaluation")
    report_type = st.radio("Choose detailed report type:", ("Classification Report", "Confusion Matrix"))

    if report_type == "Classification Report":
        st.text("Classification Report:")
        st.code(classification_report(y_true_int, y_pred_int, zero_division=0))
    else:
        st.text("Confusion Matrix:")

        # Robust label handling:
        all_labels = np.union1d(np.unique(y_true_int), np.unique(y_pred_int)).astype(int)

        if len(all_labels) < 2:
            st.info(f"Cannot generate a meaningful confusion matrix for a single class. Only class(es) found: {', '.join(map(str, all_labels))}")
            st.write(f"True counts: {pd.Series(y_true_int).value_counts().to_dict()}")
            st.write(f"Predicted counts: {pd.Series(y_pred_int).value_counts().to_dict()}")
        else:
            cm = confusion_matrix(y_true_int, y_pred_int, labels=all_labels)

            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
            disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            plt.close(fig)

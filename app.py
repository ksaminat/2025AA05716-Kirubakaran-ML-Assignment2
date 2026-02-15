import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# sklearn metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Import models
from models.logistic_regression_model import LogisticModel
from models.decision_tree_model import DecisionTreeModel
from models.knn_model import KNNModel
from models.naive_bayes_model import NaiveBayesModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel


# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Prediction Dashboard")


# ----------------------------
# SIDEBAR (LEFT PANEL)
# ----------------------------

st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# MODEL FACTORY
def get_models():

    return {
        "Logistic Regression": LogisticModel(),
        "Decision Tree": DecisionTreeModel(),
        "KNN": KNNModel(),
        "Naive Bayes": NaiveBayesModel(),
        "Random Forest": RandomForestModel(),
        "XGBoost": XGBoostModel()
    }


model_name = st.sidebar.selectbox(
    "Select Model",
    list(get_models().keys())
)

compare = st.sidebar.checkbox("Compare All Models", value=True)


# ----------------------------
# LOAD MODEL
# ----------------------------

@st.cache_resource
def load_model(name):

    model = get_models()[name]
    model.load()
    return model


# ----------------------------
# MAIN PANEL (RIGHT SIDE)
# ----------------------------

if uploaded_file is None:

    st.info("⬅️ Please upload diabetes test dataset from the left panel to begin.")
    st.stop()


# Load dataset
df = pd.read_csv(uploaded_file)

if "Outcome" not in df.columns:
    st.error("Dataset must contain 'Outcome' column")
    st.markdown("""
        ⚠️ **Please upload a dataset with target column named 'Outcome'.**
        Outcome won't be used for prediction, but it will be used for evaluation.
        """)
    st.stop()


y_true = df["Outcome"]
X = df.drop("Outcome", axis=1)

st.success(
    f"Dataset loaded successfully ({df.shape[0]} rows, {df.shape[1]} columns)"
)

model = load_model(model_name)

st.markdown(
    f"<h3 style='color:green;'><b> Model Loaded: {model_name}</b></h3>",
    unsafe_allow_html=True
)


# ----------------------------
# PREDICTIONS
# ----------------------------

y_pred = model.predict(X)

try:
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y_true, y_prob)
except:
    y_prob = None
    auc = None


# ----------------------------
# PERFORMANCE METRICS
# ----------------------------

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

st.subheader("Performance Metrics")

col1, col2, col3, col4, col5, col6 = st.columns(6)
#col4, col5, col6 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.4f}")
if auc is not None:
    col2.metric("AUC", f"{auc:.4f}")
else:
    col6.metric("AUC", "Not Available")
col3.metric("Precision", f"{precision:.4f}")
col4.metric("Recall", f"{recall:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC", f"{mcc:.4f}")

# ----------------------------
# MODEL COMPARISON
# ----------------------------

if compare:

    st.subheader("Model Comparison")

    results = []

    for name, m in get_models().items():

        m.load()

        pred = m.predict(X)

        try:
            prob = m.predict_proba(X)[:, 1]
            auc_val = roc_auc_score(y_true, prob)
        except:
            auc_val = None

        results.append({

            "Model": name,
            "Accuracy": accuracy_score(y_true, pred),
            "AUC": auc_val,
            "Precision": precision_score(y_true, pred),
            "Recall": recall_score(y_true, pred),
            "F1 Score": f1_score(y_true, pred),
            "MCC": matthews_corrcoef(y_true, pred)
        })

    df_results = pd.DataFrame(results)
    df_results.index = df_results.index + 1
    df_results.index.name = "No."

    st.dataframe(df_results, use_container_width=True)


# ----------------------------
# CONFUSION MATRIX
# ----------------------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots()

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Diabetes", "Diabetes"]
)

disp.plot(ax=ax)

st.pyplot(fig)


# ----------------------------
# ROC CURVE
# ----------------------------

if y_prob is not None:

    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr)

    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    st.pyplot(fig)


# ----------------------------
# PREDICTIONS TABLE
# ----------------------------

st.subheader("Predictions")

result_df = X.copy()

result_df["Actual"] = y_true.values
result_df["Prediction"] = y_pred

if y_prob is not None:

    result_df["Probability_No"] = model.predict_proba(X)[:, 0]
    result_df["Probability_Yes"] = model.predict_proba(X)[:, 1]

st.dataframe(result_df, use_container_width=True)


# ----------------------------
# DOWNLOAD BUTTON
# ----------------------------

csv = result_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Predictions",
    csv,
    "predictions.csv",
    "text/csv"
)

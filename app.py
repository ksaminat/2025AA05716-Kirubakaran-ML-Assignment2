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

# Import dataloader
from dataloader import load_split_data


# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Prediction Dashboard")

st.write("Upload dataset or use default dataset")


# ----------------------------
# LOAD DEFAULT DATASET
# ----------------------------

@st.cache_data
def load_default_dataset():

    X_train, X_test, y_train, y_test = load_split_data()

    return X_test, y_test


# ----------------------------
# DATASET SELECTION
# ----------------------------

dataset_option = st.radio(
    "Dataset Option",
    ("Use Default Dataset", "Upload Dataset")
)


# ----------------------------
# SESSION STATE
# ----------------------------

if "X_test" not in st.session_state:
    st.session_state.X_test = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None


# ----------------------------
# DEFAULT DATASET
# ----------------------------

if dataset_option == "Use Default Dataset":

    X_test, y_test = load_default_dataset()

    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.dataset_name = "Default Dataset"

    st.success("Default dataset loaded")


# ----------------------------
# UPLOAD DATASET
# ----------------------------

elif dataset_option == "Upload Dataset":

    uploaded_file = st.file_uploader(
        "Upload CSV with Outcome column",
        type=["csv"]
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        if "Outcome" not in df.columns:

            st.error("Dataset must contain Outcome column")
            st.stop()

        st.session_state.y_test = df["Outcome"]

        st.session_state.X_test = df.drop(
            "Outcome",
            axis=1
        )

        st.session_state.dataset_name = "Uploaded Dataset"

        st.success("Dataset uploaded successfully")


# ----------------------------
# MODEL FACTORY
# ----------------------------

def get_models():

    return {

        "Logistic Regression": LogisticModel(),
        "Decision Tree": DecisionTreeModel(),
        "KNN": KNNModel(),
        "Naive Bayes": NaiveBayesModel(),
        "Random Forest": RandomForestModel(),
        "XGBoost": XGBoostModel()

    }


# ----------------------------
# LOAD MODEL
# ----------------------------

@st.cache_resource
def load_model(name):

    model = get_models()[name]

    model.load()

    return model


# ----------------------------
# MODEL SELECTION
# ----------------------------

model_name = st.selectbox(
    "Select Model",
    list(get_models().keys())
)

model = load_model(model_name)

st.success(f"Model: {model_name}")

if st.session_state.dataset_name:
    st.info(f"Dataset: {st.session_state.dataset_name}")


# ----------------------------
# METRICS
# ----------------------------

if st.session_state.X_test is not None:

    X = st.session_state.X_test
    y_true = st.session_state.y_test

    y_pred = model.predict(X)

    try:
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    except:
        y_prob = None
        auc = None


    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    mcc = matthews_corrcoef(y_true, y_pred)


    st.subheader("Performance Metrics")

    col1, col2, col3 = st.columns(3)

    col4, col5, col6 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")

    col2.metric("Precision", f"{precision:.4f}")

    col3.metric("Recall", f"{recall:.4f}")

    col4.metric("F1 Score", f"{f1:.4f}")

    col5.metric("MCC", f"{mcc:.4f}")

    if auc:
        col6.metric("AUC", f"{auc:.4f}")
    else:
        col6.metric("AUC", "Not Available")


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

    ax.set_title("ROC Curve")

    st.pyplot(fig)


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
# PREDICTIONS TABLE
# ----------------------------

st.subheader("Predictions")

result_df = X.copy()

result_df["Prediction"] = y_pred

result_df["Actual"] = y_true.values

if y_prob is not None:

    result_df["Probability_No"] = model.predict_proba(X)[:, 0]

    result_df["Probability_Yes"] = model.predict_proba(X)[:, 1]

st.dataframe(result_df)


# ----------------------------
# DOWNLOAD RESULTS
# ----------------------------

csv = result_df.to_csv(index=False).encode("utf-8")

st.download_button(

    "Download Predictions",

    csv,

    "predictions.csv",

    "text/csv"
)


# ----------------------------
# MODEL COMPARISON
# ----------------------------

if st.checkbox("Compare All Models"):

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

            "Precision": precision_score(y_true, pred),

            "Recall": recall_score(y_true, pred),

            "F1": f1_score(y_true, pred),

            "MCC": matthews_corrcoef(y_true, pred),

            "AUC": auc_val

        })

    df_results = pd.DataFrame(results)

    st.dataframe(df_results)

    fig, ax = plt.subplots()

    ax.bar(df_results["Model"], df_results["Accuracy"])

    plt.xticks(rotation=45)

    ax.set_title("Accuracy Comparison")

    st.pyplot(fig)

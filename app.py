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
    roc_auc_score
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
st.write("Choose default dataset OR upload your own dataset.")


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
    "Select Dataset Option",
    ("Use Default Dataset", "Upload Dataset")
)


# ----------------------------
# SESSION STATE INIT
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

    X_test_default, y_test_default = load_default_dataset()

    st.session_state.X_test = X_test_default
    st.session_state.y_test = y_test_default
    st.session_state.dataset_name = "Default Test Dataset"

    st.success("Using Default Dataset")


# ----------------------------
# UPLOAD DATASET
# ----------------------------

elif dataset_option == "Upload Dataset":

    uploaded_file = st.file_uploader(
        "Upload CSV with Outcome column",
        type=["csv"]
    )

    if uploaded_file is not None:

        df_uploaded = pd.read_csv(uploaded_file)

        if "Outcome" not in df_uploaded.columns:

            st.error("Dataset must contain 'Outcome' column")
            st.stop()

        y_uploaded = df_uploaded["Outcome"]
        X_uploaded = df_uploaded.drop("Outcome", axis=1)

        st.session_state.X_test = X_uploaded
        st.session_state.y_test = y_uploaded
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
def load_model(model_name):

    model = get_models()[model_name]

    model.load()

    return model


# ----------------------------
# MODEL SELECTION
# ----------------------------

model_name = st.selectbox(
    "Select Model",
    list(get_models().keys())
)

st.success(f"Model being used: {model_name}")

if st.session_state.dataset_name:
    st.info(f"Dataset being used: {st.session_state.dataset_name}")


# ----------------------------
# LOAD SELECTED MODEL
# ----------------------------

model = load_model(model_name)


# ----------------------------
# EVALUATION METRICS
# ----------------------------

if (
    st.session_state.X_test is not None and
    st.session_state.y_test is not None
):

    X = st.session_state.X_test
    y_true = st.session_state.y_test

    y_pred = model.predict(X)

    # AUC
    try:
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = None

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    st.subheader("Model Performance Metrics")

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
# PREDICTIONS
# ----------------------------

if st.session_state.X_test is not None:

    st.subheader("Prediction Results")

    try:

        predictions = model.predict(st.session_state.X_test)

        result_df = st.session_state.X_test.copy()

        result_df["Prediction"] = predictions

        if st.session_state.y_test is not None:
            result_df["Actual"] = st.session_state.y_test.values

        try:
            prob = model.predict_proba(st.session_state.X_test)
            result_df["Probability_No_Diabetes"] = prob[:, 0]
            result_df["Probability_Diabetes"] = prob[:, 1]
        except:
            pass

        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions CSV",
            csv,
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# ----------------------------
# MODEL COMPARISON
# ----------------------------

if (
    st.checkbox("Show Accuracy Comparison of All Models")
    and st.session_state.y_test is not None
):

    st.subheader("All Models Comparison")

    X = st.session_state.X_test
    y_true = st.session_state.y_test

    results = []

    for name, model_obj in get_models().items():

        model_obj.load()

        y_pred = model_obj.predict(X)

        try:
            y_prob = model_obj.predict_proba(X)[:, 1]
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = None

        results.append({

            "Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "AUC": auc

        })

    results_df = pd.DataFrame(results)

    st.dataframe(results_df)

    # Plot Accuracy
    fig, ax = plt.subplots()

    ax.bar(results_df["Model"], results_df["Accuracy"])

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")

    plt.xticks(rotation=45)

    st.pyplot(fig)

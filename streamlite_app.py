import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
# LOAD DEFAULT TEST DATASET
# ----------------------------

@st.cache_data
def load_default_dataset():

    X_train, X_test, y_train, y_test = load_split_data()

    return X_test, y_test


# ----------------------------
# DATASET SELECTION OPTION
# ----------------------------

dataset_option = st.radio(
    "Select Dataset Option",
    ("Use Default Dataset", "Upload Dataset")
)


# ----------------------------
# INITIALIZE SESSION STATE
# ----------------------------

if "X_test" not in st.session_state:
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.dataset_name = None


# ----------------------------
# HANDLE DEFAULT DATASET
# ----------------------------

if dataset_option == "Use Default Dataset":

    X_test_default, y_test_default = load_default_dataset()

    st.session_state.X_test = X_test_default
    st.session_state.y_test = y_test_default
    st.session_state.dataset_name = "Default Test Dataset"

    st.success("Using Default Dataset")


# ----------------------------
# HANDLE UPLOADED DATASET
# ----------------------------

elif dataset_option == "Upload Dataset":

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df_uploaded = pd.read_csv(uploaded_file)

        if "Outcome" in df_uploaded.columns:

            st.session_state.X_test = df_uploaded.drop("Outcome", axis=1)
            st.session_state.y_test = df_uploaded["Outcome"]

            st.session_state.dataset_name = "Uploaded Dataset (with Outcome)"

        else:

            st.session_state.X_test = df_uploaded
            st.session_state.y_test = None

            st.session_state.dataset_name = "Uploaded Dataset (prediction only)"

        st.success("Uploaded dataset loaded successfully")


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

if st.session_state.dataset_name is not None:
    st.info(f"Dataset being used: {st.session_state.dataset_name}")


# ----------------------------
# LOAD MODEL
# ----------------------------

model = load_model(model_name)


# ----------------------------
# EVALUATION
# ----------------------------

if st.session_state.y_test is not None:

    accuracy = model.evaluate(
        st.session_state.X_test,
        st.session_state.y_test
    )

    st.info(f"{model_name} Accuracy: {accuracy:.4f}")


# ----------------------------
# PREDICTION
# ----------------------------

if st.session_state.X_test is not None:

    st.subheader("Prediction Results")

    try:

        predictions = model.predict(st.session_state.X_test)

        result_df = st.session_state.X_test.copy()

        result_df["Prediction"] = predictions


        try:

            prob = model.predict_proba(st.session_state.X_test)

            result_df["Probability_No_Diabetes"] = prob[:, 0]
            result_df["Probability_Diabetes"] = prob[:, 1]

        except:
            pass


        st.dataframe(result_df)


        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )


    except Exception as e:

        st.error(f"Prediction Error: {e}")


# ----------------------------
# OPTIONAL MODEL COMPARISON
# ----------------------------

if (
    st.checkbox("Show Accuracy Comparison of All Models")
    and st.session_state.y_test is not None
):

    results = {}

    for name, model_obj in get_models().items():

        model_obj.load()

        acc = model_obj.evaluate(
            st.session_state.X_test,
            st.session_state.y_test
        )

        results[name] = acc


    accuracy_df = pd.DataFrame(
        results.items(),
        columns=["Model", "Accuracy"]
    )

    st.dataframe(accuracy_df)


    fig, ax = plt.subplots()

    ax.bar(
        accuracy_df["Model"],
        accuracy_df["Accuracy"]
    )

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")

    plt.xticks(rotation=45)

    st.pyplot(fig)

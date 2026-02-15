# config.py

DATA_URL = "./data/diabetes_train.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_DIR = "saved_models"

# Logistic Regression
LOGISTIC_MODEL = f"{MODEL_DIR}/logistic.pkl"
LOGISTIC_SCALER = f"{MODEL_DIR}/logistic_scaler.pkl"

# Decision Tree
TREE_MODEL = f"{MODEL_DIR}/decision_tree.pkl"

# KNN
KNN_MODEL = f"{MODEL_DIR}/knn.pkl"
KNN_SCALER = f"{MODEL_DIR}/knn_scaler.pkl"

# Naive Bayes
NB_MODEL = f"{MODEL_DIR}/naive_bayes.pkl"
NB_SCALER = f"{MODEL_DIR}/naive_bayes_scaler.pkl"

# Random Forest
RF_MODEL = f"{MODEL_DIR}/random_forest.pkl"

# XGBoost
XGB_MODEL = f"{MODEL_DIR}/xgboost.pkl"

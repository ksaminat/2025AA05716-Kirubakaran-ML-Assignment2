# logistic_model.py

import pickle
import numpy as np
import pandas as pd

import os
import sys

# Add project root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import config

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class LogisticModel:

    def __init__(self, alpha=0.1, penalty="l2"):

        # alpha = regularization strength
        self.alpha = alpha

        # sklearn uses inverse
        self.C = 1 / alpha

        self.penalty = penalty

        self.scaler = StandardScaler()

        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver="liblinear",
            max_iter=1000
        )


    # TRAIN MODEL
    def train(self, X_train, y_train):

        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model.fit(X_train_scaled, y_train)


    # EVALUATE MODEL
    def evaluate(self, X_test, y_test):

        X_test_scaled = self.scaler.transform(X_test)

        predictions = self.model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, predictions)

        return accuracy


    # PREDICT NEW DATA
    def predict(self, X):

        # convert to numpy if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)

        return prediction
    
    # PREDICT CSV FILE
    def predict_csv(self, input_csv, output_csv="predictions.csv"):

        # load csv
        df = pd.read_csv(input_csv)

        print("Input data shape:", df.shape)

        # scale
        X_scaled = self.scaler.transform(df)

        # predict
        predictions = self.model.predict(X_scaled)

        probabilities = self.model.predict_proba(X_scaled)

        df["Prediction"] = predictions

        df["Probability_No_Diabetes"] = probabilities[:, 0]

        df["Probability_Diabetes"] = probabilities[:, 1]

        df.to_csv(output_csv, index=False)

        # add predictions column
        df["Prediction"] = predictions

        # save output
        df.to_csv(output_csv, index=False)

        print("Predictions saved to:", output_csv)

        return df
    
    # PREDICT PROBABILITIES (REQUIRED FOR AUC)
    def predict_proba(self, X):

        # convert to numpy if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # scale input
        X_scaled = self.scaler.transform(X)

        # get probabilities
        probabilities = self.model.predict_proba(X_scaled)

        return probabilities


    # SAVE MODEL
    def save(self,
             model_path="diabetes_model.pkl",
             scaler_path="diabetes_scaler.pkl"):

        pickle.dump(self.model, open(config.LOGISTIC_MODEL, "wb"))
        pickle.dump(self.scaler, open(config.LOGISTIC_SCALER, "wb"))


    # LOAD MODEL
    def load(self,
             model_path="diabetes_model.pkl",
             scaler_path="diabetes_scaler.pkl"):

        self.model = pickle.load(open(config.LOGISTIC_MODEL, "rb"))

        self.scaler = pickle.load(open(config.LOGISTIC_SCALER, "rb"))

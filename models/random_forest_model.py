import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import os
import sys

# Add project root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import config


class RandomForestModel:

    def __init__(self):

        self.model = RandomForestClassifier(n_estimators=100)


    def train(self, X_train, y_train):

        self.model.fit(X_train, y_train)


    def evaluate(self, X_test, y_test):

        pred = self.model.predict(X_test)

        return accuracy_score(y_test, pred)


    def predict(self, X):

        return self.model.predict(X)

    # ----------------------------
    # PREDICT PROBABILITY (REQUIRED FOR AUC)
    # ----------------------------
    def predict_proba(self, X):

        # ensure numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        probabilities = self.model.predict_proba(X)

        return probabilities

    def save(self):

        pickle.dump(self.model, open(config.RF_MODEL, "wb"))


    def load(self):

        self.model = pickle.load(open(config.RF_MODEL, "rb"))

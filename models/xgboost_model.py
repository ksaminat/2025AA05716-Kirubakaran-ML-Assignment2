import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import os
import sys

# Add project root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import config


class XGBoostModel:

    def __init__(self):

        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )


    def train(self, X_train, y_train):

        self.model.fit(X_train, y_train)


    def evaluate(self, X_test, y_test):

        pred = self.model.predict(X_test)

        return accuracy_score(y_test, pred)


    def predict(self, X):

        return self.model.predict(X)


    def save(self):

        pickle.dump(self.model, open(config.XGB_MODEL, "wb"))


    def load(self):

        self.model = pickle.load(open(config.XGB_MODEL, "rb"))

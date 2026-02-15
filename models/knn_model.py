import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import os
import sys

# Add project root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import config


class KNNModel:

    def __init__(self):

        self.scaler = StandardScaler()

        self.model = KNeighborsClassifier(n_neighbors=5)


    def train(self, X_train, y_train):

        X_scaled = self.scaler.fit_transform(X_train)

        self.model.fit(X_scaled, y_train)


    def evaluate(self, X_test, y_test):

        X_scaled = self.scaler.transform(X_test)

        pred = self.model.predict(X_scaled)

        return accuracy_score(y_test, pred)


    def predict(self, X):

        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)


    def save(self):

        pickle.dump(self.model, open(config.KNN_MODEL, "wb"))

        pickle.dump(self.scaler, open(config.KNN_SCALER, "wb"))


    def load(self):

        self.model = pickle.load(open(config.KNN_MODEL, "rb"))

        self.scaler = pickle.load(open(config.KNN_SCALER, "rb"))

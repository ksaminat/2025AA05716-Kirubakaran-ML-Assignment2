import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import os
import sys

# Add project root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import config


class DecisionTreeModel:

    def __init__(self):

        self.model = DecisionTreeClassifier()


    def train(self, X_train, y_train):

        self.model.fit(X_train, y_train)


    def evaluate(self, X_test, y_test):

        pred = self.model.predict(X_test)

        return accuracy_score(y_test, pred)


    def predict(self, X):

        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)    

    def save(self):

        pickle.dump(self.model, open(config.TREE_MODEL, "wb"))


    def load(self):

        self.model = pickle.load(open(config.TREE_MODEL, "rb"))

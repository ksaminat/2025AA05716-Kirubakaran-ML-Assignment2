from dataloader import load_split_data

from models.logistic_regression_model import LogisticModel
from models.decision_tree_model import DecisionTreeModel
from models.knn_model import KNNModel
from models.naive_bayes_model import NaiveBayesModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel


def train_model(model, name, X_train, X_test, y_train, y_test):

    print(f"\nTraining {name}")

    model.train(X_train, y_train)

    acc = model.evaluate(X_test, y_test)

    print(f"{name} Accuracy: {acc:.4f}")

    model.save()

    return acc


def main():

    X_train, X_test, y_train, y_test = load_split_data()

    models = [

        ("Logistic Regression", LogisticModel()),
        ("Decision Tree", DecisionTreeModel()),
        ("KNN", KNNModel()),
        ("Naive Bayes", NaiveBayesModel()),
        ("Random Forest", RandomForestModel()),
        ("XGBoost", XGBoostModel())

    ]

    results = {}

    for name, model in models:

        acc = train_model(model, name, X_train, X_test, y_train, y_test)

        results[name] = acc


    print("\nSummary")

    for name, acc in results.items():

        print(name, acc)


if __name__ == "__main__":
    main()

# dataloader.py

import pandas as pd
from sklearn.model_selection import train_test_split
import config



def load_data():
    """
    Load diabetes dataset
    """
    df = pd.read_csv(config.DATA_URL)
    return df


def split_data(test_size=0.2, random_state=42):
    """
    Split dataset into train and test
    """
    test_size = config.TEST_SIZE
    random_state=config.RANDOM_STATE
    df = load_data()

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def load_split_data():
    return split_data()


# prepare UNSW-NB15 for machine learning
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# dataset files
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"


def load_data():
    """Load the training and testing datasets."""

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print("Training records:", len(train))
    print("Testing records:", len(test))

    return train, test


def prepare_features(train, test):
    """Separate network features from attack labels."""

    # remove columns that shouldnt be used as predictors
    drop_columns = ["id", "attack_cat", "label"]

    X_train = train.drop(columns=drop_columns)
    X_test = test.drop(columns=drop_columns)

    # 0 = normal, 1 = attack
    y_train = train["label"]
    y_test = test["label"]

    # replace infinite values with missing values
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    return X_train, X_test, y_train, y_test


def create_preprocessor(X_train):
    """Create transformations for numeric and text features."""

    # categorical and numerical columns
    categorical_columns = X_train.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print("\nCategorical columns:", categorical_columns)
    print("Numerical columns:", len(numerical_columns))

    # fill missing numerical values and scale them
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # fill missing text values and convert to numeric columns
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="missing"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # transform to each column type
    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    return preprocessor


def preprocess_data():
    """Load and preprocess both datasets."""

    train, test = load_data()

    X_train, X_test, y_train, y_test = prepare_features(
        train, test
    )

    preprocessor = create_preprocessor(X_train)

    # learn preprocessing parameters from training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # same transformations applied to testing data
    X_test_processed = preprocessor.transform(X_test)

    print("\nPreprocessing completed successfully!")

    print("Training shape:", X_train_processed.shape)
    print("Testing shape:", X_test_processed.shape)

    print("\nTraining labels:")
    print(y_train.value_counts())

    print("\nTesting labels:")
    print(y_test.value_counts())

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test
    )


if __name__ == "__main__":
    preprocess_data()

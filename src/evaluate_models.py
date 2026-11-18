# compare KNN, Decision Tree, and Random Forest models
# UNSW-NB15 dataset

import time
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# importing preprocessing functions
from preprocess_data import (
    load_data,
    prepare_features,
    create_preprocessor
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# sample sizes consistentfor all models
TRAIN_SIZE = 10000
TEST_SIZE = 2000
RANDOM_STATE = 42


def load_and_prepare_data():
    """Load, sample, and preprocess the network traffic data."""

    # load original training and testing datasets
    train, test = load_data()

    # separate features from classification labels
    X_train, X_test, y_train, y_test = prepare_features(
        train, test
    )

    # stratified training sample
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train,
        y_train,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    # stratified testing sample
    X_test_sample, _, y_test_sample, _ = train_test_split(
        X_test,
        y_test,
        train_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_test
    )

    #  create preprocessing pipeline
    preprocessor = create_preprocessor(X_train_sample)

    # fit preprocessing using only the selected training data
    X_train_processed = preprocessor.fit_transform(
        X_train_sample
    )

    # apply the learned transformations to testing data
    X_test_processed = preprocessor.transform(
        X_test_sample
    )

    print("\n===== Prepared Dataset =====")
    print("Training shape:", X_train_processed.shape)
    print("Testing shape:", X_test_processed.shape)

    return (
        X_train_processed,
        X_test_processed,
        y_train_sample,
        y_test_sample
    )


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train one model and calculate its performance."""

    print(f"\n===== Evaluating {name} =====")

    # training time
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    # prediction time
    start = time.perf_counter()
    predictions = model.predict(X_test)
    prediction_time = time.perf_counter() - start

    # confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # false-positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # store performance measurements
    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "False-positive rate": fpr,
        "Training time (s)": training_time,
        "Prediction time (s)": prediction_time,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    print("Confusion Matrix:")
    print(cm)

    print(f"Accuracy: {results['Accuracy']:.4f}")
    print(f"Precision: {results['Precision']:.4f}")
    print(f"Recall: {results['Recall']:.4f}")
    print(f"F1-score: {results['F1-score']:.4f}")
    print(f"False-positive rate: {fpr:.4f}")

    return results


def main():
    # same dataset for every model
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    models = {
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    # results from each algorithm
    all_results = []

    # train and evaluate each model
    for name, model in models.items():
        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        all_results.append(result)

    # convert results into pandas dataframe
    comparison = pd.DataFrame(all_results)

    print("\n===== Final Model Comparison =====")
    print(comparison.round(4).to_string(index=False))

    # create results folder if it does not exist already
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # save results to csv file
    output_path = RESULTS_DIR / "model_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

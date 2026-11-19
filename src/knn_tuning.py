# knn Hyperparameter Tuning
#
# testing different values of K to find which one works best for detecting network attacks

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# preprocessing functions
from preprocess_data import (
    load_data,
    prepare_features,
    create_preprocessor
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"


def main():

    # load data
    train, test = load_data()

    # Separate the features from labels
    X_train, _, y_train, _ = prepare_features(
        train, test
    )

    # same sample size and random state as decision tree tuning
    X_sample, _, y_sample, _ = train_test_split(
        X_train,
        y_train,
        train_size=10000,
        random_state=42,
        stratify=y_train
    )

    # 80% training 20% validation
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_sample,
        y_sample,
        test_size=0.20,
        random_state=42,
        stratify=y_sample
    )

    print("\n===== KNN Hyperparameter Tuning =====")
    print("Training samples:", len(y_fit))
    print("Validation samples:", len(y_val))

    # small K values can be sensitive to individual records
    # larger K values produce smoother decision boundaries
    k_values = [3, 5, 7, 9, 11]

    # store results for each K value
    results = []

    for k in k_values:

        print(f"\nTesting K = {k}...")

        # preprocess the network traffic features
        # train knn using the current K value
        model = Pipeline([
            ("preprocessor", create_preprocessor(X_fit)),
            ("classifier", KNeighborsClassifier(
                n_neighbors=k,
                n_jobs=-1
            ))
        ])

        model.fit(X_fit, y_fit)

        # predictions
        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)

        # predicted attacks that are real
        precision = precision_score(
            y_val, predictions, zero_division=0
        )

        # actual attacks successfully detected
        recall = recall_score(
            y_val, predictions, zero_division=0
        )

        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        # confusion matrix values
        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        # false-positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # results
        results.append({
            "K": k,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "False-positive rate": fpr
        })

    comparison = pd.DataFrame(results)

    print("\n===== KNN Validation Results =====")
    print(comparison.round(4).to_string(index=False))

    # K value with the highest f1 wins
    best_index = comparison["F1-score"].idxmax()
    best = comparison.loc[best_index]

    print("\n===== Best KNN Configuration =====")
    print("K:", int(best["K"]))
    print(f"Validation F1-score: {best['F1-score']:.4f}")
    print(f"Validation accuracy: {best['Accuracy']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # save knn experiments
    output_path = RESULTS_DIR / "knn_tuning.csv"
    comparison.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

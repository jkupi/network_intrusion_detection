# Random Forest Hyperparameter Tuning
# testing different numbers of trees and tree depths

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocess_data import (
    load_data,
    prepare_features,
    create_preprocessor
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    # load dataset
    train, test = load_data()

    X_train, _, y_train, _ = prepare_features(
        train, test
    )

    # same random sample
    X_sample, _, y_sample, _ = train_test_split(
        X_train,
        y_train,
        train_size=10000,
        random_state=42,
        stratify=y_train
    )

    # training and validation sets
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_sample,
        y_sample,
        test_size=0.20,
        random_state=42,
        stratify=y_sample
    )

    print("\n===== Random Forest Hyperparameter Tuning =====")
    print("Training samples:", len(y_fit))
    print("Validation samples:", len(y_val))

    # how many trees are created
    n_estimators_values = [50, 100, 200]

    # how deep each tree can grow.
    max_depth_values = [5, 10, 15, None]

    results = []

    # testing every combination
    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:

            print(
                f"\nTesting n_estimators={n_estimators}, "
                f"max_depth={max_depth}"
            )

            # fresh pipeline
            model = Pipeline([
                ("preprocessor", create_preprocessor(X_fit)),
                ("classifier", RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                ))
            ])

            # training
            model.fit(X_fit, y_fit)

            # predictions
            predictions = model.predict(X_val)

            accuracy = accuracy_score(y_val, predictions)

            precision = precision_score(
                y_val, predictions, zero_division=0
            )

            recall = recall_score(
                y_val, predictions, zero_division=0
            )

            f1 = f1_score(
                y_val, predictions, zero_division=0
            )

            # confusion matrix
            tn, fp, fn, tp = confusion_matrix(
                y_val, predictions, labels=[0, 1]
            ).ravel()

            # false positive rate
            fpr = fp / (fp + tn) if fp + tn else 0

            results.append({
                "N Estimators": n_estimators,
                "Max Depth": str(max_depth),
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "False-positive rate": fpr
            })

    # display results
    comparison = pd.DataFrame(results)

    print("\n===== Random Forest Validation Results =====")
    print(comparison.round(4).to_string(index=False))

    # highest f1-score wins
    best_index = comparison["F1-score"].idxmax()
    best = comparison.loc[best_index]

    print("\n===== Best Random Forest Configuration =====")
    print("N Estimators:", int(best["N Estimators"]))
    print("Max Depth:", best["Max Depth"])
    print(f"Validation F1-score: {best['F1-score']:.4f}")
    print(f"Validation accuracy: {best['Accuracy']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "random_forest_tuning.csv"
    comparison.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

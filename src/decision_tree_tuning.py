# experimenting with different decision tree depths
# validation set to select the best setting

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
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
    # original datasets
    train, test = load_data()

    # separate features and labels
    X_train, _, y_train, _ = prepare_features(
        train, test
    )

    # 10,000 records from the training dataset
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

    print("\n===== Decision Tree Hyperparameter Tuning =====")
    print("Training samples:", len(y_fit))
    print("Validation samples:", len(y_val))

    # different maximum depths for experimentation
    depths = [3, 5, 10, 15, 20, None]

    results = []

    for depth in depths:
        # fresh preprocessing and model pipeline
        model = Pipeline([
            ("preprocessor", create_preprocessor(X_fit)),
            ("classifier", DecisionTreeClassifier(
                max_depth=depth,
                random_state=42
            ))
        ])

        # training
        model.fit(X_fit, y_fit)

        # predictions
        predictions = model.predict(X_val)

        # accuracy, precision, recall, f1-score
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

        fpr = fp / (fp + tn) if fp + tn else 0

        results.append({
            "Max Depth": str(depth),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "False-positive rate": fpr
        })

    # results of experiment
    comparison = pd.DataFrame(results)

    print("\n===== Validation Results =====")
    print(comparison.round(4).to_string(index=False))

    # depth with the highest f1-score wins
    best_index = comparison["F1-score"].idxmax()
    best = comparison.loc[best_index]

    print("\n===== Best Decision Tree Configuration =====")
    print("Max depth:", best["Max Depth"])
    print(f"Validation F1-score: {best['F1-score']:.4f}")
    print(f"Validation accuracy: {best['Accuracy']:.4f}")

    # results folder
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # save to csv files
    output_path = RESULTS_DIR / "decision_tree_tuning.csv"
    comparison.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

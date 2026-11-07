# train and evaluate KNN model
import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from preprocess_data import preprocess_data


def main():
    # preprocessed UNSW-NB15 dataset
    X_train, X_test, y_train, y_test = preprocess_data()

    # less samples for first KNN experiment
    # stratification to preserve the normal/attack class proportions
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train,
        y_train,
        train_size=10000,
        random_state=42,
        stratify=y_train
    )

    X_test_sample, _, y_test_sample, _ = train_test_split(
        X_test,
        y_test,
        train_size=2000,
        random_state=42,
        stratify=y_test
    )

    print("\n===== KNN Intrusion Detection =====")
    print("Training samples:", len(y_train_sample))
    print("Testing samples:", len(y_test_sample))

    # KNN classifier using five neighbors
    model = KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    )

    # train the model and measure training time
    start_time = time.perf_counter()
    model.fit(X_train_sample, y_train_sample)
    training_time = time.perf_counter() - start_time

    # predict whether test traffic is normal or malicious
    start_time = time.perf_counter()
    predictions = model.predict(X_test_sample)
    prediction_time = time.perf_counter() - start_time

    # classification metrics
    accuracy = accuracy_score(y_test_sample, predictions)
    precision = precision_score(
        y_test_sample, predictions, zero_division=0
    )
    recall = recall_score(
        y_test_sample, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test_sample, predictions, zero_division=0
    )

    # confusion matrix
    cm = confusion_matrix(
        y_test_sample, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # false positive rate
    false_positive_rate = fp / (fp + tn) if fp + tn else 0

    # display results
    print("\n===== Model Performance =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"False-positive rate: {false_positive_rate:.4f}")

    print(f"\nTraining time: {training_time:.4f} seconds")
    print(f"Prediction time: {prediction_time:.4f} seconds")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_test_sample,
        predictions,
        target_names=["Normal", "Attack"],
        zero_division=0
    ))


if __name__ == "__main__":
    main()

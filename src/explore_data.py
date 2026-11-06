# UNSW-NB15 network intrusion detection dataset
from pathlib import Path
import pandas as pd

# Find project folder/dataset files
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"


def explore_dataset(data, name):
    """Display basic information about a dataset."""

    print(f"\n===== {name} =====")

    # number of rows and columns
    print("\nDataset shape:", data.shape)

    # display first five network traffic records
    print("\nFirst five records:")
    print(data.head())

    # all available column names
    print("\nColumn names:")
    print(data.columns.tolist())

    # missing values
    print("\nMissing values:")
    print(data.isnull().sum())

    # number of normal/malicious traffic records
    print("\nTraffic classification:")
    print(data["label"].value_counts())

    # different attack categories
    print("\nAttack categories:")
    print(data["attack_cat"].value_counts(dropna=False))


def main():
    # Check if both dataset files exist
    for path in [TRAIN_PATH, TEST_PATH]:
        if not path.exists():
            print(f"Dataset file not found: {path}")
            return

    # load CSV files into dataFrames
    training_data = pd.read_csv(TRAIN_PATH)
    testing_data = pd.read_csv(TEST_PATH)

    explore_dataset(training_data, "Training Dataset")
    explore_dataset(testing_data, "Testing Dataset")


if __name__ == "__main__":
    main()

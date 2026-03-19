import pandas as pd

from paths import RAW_DATA_DIR
from utils.download_kaggle_dataset import download_dataset


def load_iris_data() -> pd.DataFrame:
    """
    Download (if needed) and load the Iris dataset as a pandas DataFrame.
    """
    iris_path = download_dataset("uciml/iris", RAW_DATA_DIR / "iris")

    csv_file = iris_path / "Iris.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_file}")

    df = pd.read_csv(csv_file)

    if df.empty:
        raise ValueError("Loaded dataset is empty")

    return df


def main():
    iris_df = load_iris_data()

    # Quick inspection
    print("First 5 rows:")
    print(iris_df.head())

    print("\nDataset info:")
    print(iris_df.info())

    print("\nSummary statistics:")
    print(iris_df.describe())


if __name__ == "__main__":
    main()

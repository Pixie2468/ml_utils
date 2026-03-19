# ml_utils

Utilities and preprocessing pipelines for machine learning workflows.

This repository currently focuses on:

- CSV classification preprocessing with scikit-learn + PyTorch tensors
- Downloading Kaggle datasets into a local data lake layout
- Shared path helpers for raw and processed data directories

## Project Layout

```text
ml_utils/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── paths.py
│   ├── pipelines/
│   │   ├── base_pipeline.py
│   │   └── csv_pipelines/classification.py
│   └── utils/
│       ├── download_kaggle_dataset.py
│       └── get_env.py
├── pyproject.toml
├── requirements.in
└── requirements.txt
```

## Requirements

- Python 3.9+
- pip

## Setup

### 1 Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2 Install dependencies

Choose one approach:

```bash
# Minimal reproducible install
pip install -r requirements.txt
```

```bash
# Editable install from pyproject metadata
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev,notebook]"
```

## Environment Variables

Copy the example file and add your Kaggle credentials:

```bash
cp .env.example .env
```

Expected variables:

- KAGGLE_USERNAME
- KAGGLE_KEY

The Kaggle API can also authenticate using ~/.kaggle/kaggle.json.

## Usage

### Run the classification preprocessing pipeline

Example using the bundled Iris dataset:

```python
import pandas as pd
from pipelines.csv_pipelines.classification import ClassificationPipeline_V1

df = pd.read_csv("data/raw/iris/Iris.csv")

# Optional: drop ID column before preprocessing
df = df.drop(columns=["Id"])

pipeline = ClassificationPipeline_V1(df=df, target_col="Species")
X_train, X_test, y_train, y_test = pipeline.pipeline()

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
```

What this pipeline does:

1. Validates input and target column
2. Performs stratified train/test split
3. Drops high-null feature columns based on threshold
4. Imputes missing values
5. One-hot encodes categorical features
6. Scales features
7. Encodes labels and returns PyTorch tensors

### Download a Kaggle dataset

```python
from pathlib import Path
from utils.download_kaggle_dataset import download_dataset
from paths import RAW_DATA_DIR

dataset = "uciml/iris"
output_path = RAW_DATA_DIR / "iris"

download_dataset(dataset=dataset, download_path=Path(output_path))
```

This downloads and unzips files under data/raw by default.

## Data Directories

- data/raw: source datasets (downloaded or manually added)
- data/processed: transformed data outputs

Path helpers are defined in src/paths.py:

- PROJECT_ROOT
- DATA_DIR
- RAW_DATA_DIR
- PROCESSED_DATA_DIR

## Development

Run linting:

```bash
ruff check src
```

Run tests (if/when tests are added):

```bash
pytest
```

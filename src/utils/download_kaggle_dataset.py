from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

from paths import RAW_DATA_DIR


def download_dataset(dataset: str, download_path: Path = RAW_DATA_DIR) -> Path:
    download_path.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset, path=str(download_path), unzip=True)

    print(f"Dataset downloaded to: {download_path.resolve()}")
    return download_path

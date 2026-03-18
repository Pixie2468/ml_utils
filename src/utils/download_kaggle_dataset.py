from pathlib import Path
import kagglehub

from paths import RAW_DATA_DIR


def download_dataset(dataset: str, download_path: Path = RAW_DATA_DIR) -> Path:
    download_path.mkdir(parents=True, exist_ok=True)
    
    path = kagglehub.dataset_download(dataset, path=str(download_path))
    path = Path(path)

    print(f"Path to dataset files: {path}")
    return path
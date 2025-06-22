"""
PyTorch DataLoader Creation Utilities for Image Classification

This module provides utilities for creating PyTorch DataLoaders from image directories,
with support for various dataset formats, data augmentation, and performance optimization.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any
import logging

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default number of workers (use all available CPU cores, but cap at reasonable limit)
NUM_WORKERS = min(os.cpu_count() or 1, 8)


def create_dataloaders(
    train_dir: Union[str, Path],
    test_dir: Union[str, Path],
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    pin_memory: Optional[bool] = None,
    shuffle_train: bool = True,
    shuffle_test: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = True
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Creates training and testing DataLoaders from image directories.

    Takes in training and testing directory paths and converts them into PyTorch 
    DataLoaders with proper configuration for image classification tasks.

    Args:
        train_dir: Path to training directory containing class subdirectories
        test_dir: Path to testing directory containing class subdirectories  
        transform: torchvision transforms to apply to training and testing data
        batch_size: Number of samples per batch in each DataLoader
        num_workers: Number of worker processes for data loading (default: min(CPU_count, 8))
        pin_memory: Whether to pin memory for faster GPU transfer (auto-detected if None)
        shuffle_train: Whether to shuffle training data (default: True)
        shuffle_test: Whether to shuffle test data (default: False)
        drop_last: Whether to drop the last incomplete batch (default: False)
        persistent_workers: Whether to keep workers alive between epochs (default: True)

    Returns:
        Tuple of (train_dataloader, test_dataloader, class_names)
        where class_names is a list of target class names

    Raises:
        FileNotFoundError: If train_dir or test_dir doesn't exist
        ValueError: If directories are empty or have no valid classes
        RuntimeError: If DataLoader creation fails

    Example:
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ...     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        ...                        std=[0.229, 0.224, 0.225])
        ... ])
        >>> train_dl, test_dl, classes = create_dataloaders(
        ...     train_dir="data/train",
        ...     test_dir="data/test", 
        ...     transform=transform,
        ...     batch_size=32
        ... )
    """
    # Convert to Path objects for better path handling
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    
    # Validate directories exist
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Auto-detect pin_memory based on CUDA availability
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    # Adjust num_workers for Windows compatibility
    if os.name == 'nt' and num_workers > 0:  # Windows
        num_workers = min(num_workers, 4)  # Windows has issues with too many workers
    
    try:
        # Create datasets using ImageFolder
        train_data = datasets.ImageFolder(root=train_dir, transform=transform)
        test_data = datasets.ImageFolder(root=test_dir, transform=transform)
        
        # Validate datasets are not empty
        if len(train_data) == 0:
            raise ValueError(f"No images found in training directory: {train_dir}")
        if len(test_data) == 0:
            raise ValueError(f"No images found in test directory: {test_dir}")
        
        # Get class names and validate consistency
        class_names = train_data.classes
        if len(class_names) == 0:
            raise ValueError("No class directories found")
        
        # Warn if class names don't match between train and test
        if set(train_data.classes) != set(test_data.classes):
            raise ValueError(
                f"Class names differ between train and test sets.\n"
                f"Train classes: {train_data.classes}\n"
                f"Test classes: {test_data.classes}"
            )
        
        # Create DataLoaders
        train_dataloader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers and num_workers > 0
        )

        test_dataloader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers and num_workers > 0
        )
        
        # Log dataset information
        logger.info(f"Created DataLoaders successfully:")
        logger.info(f"  Train samples: {len(train_data)}")
        logger.info(f"  Test samples: {len(test_data)}")
        logger.info(f"  Classes: {len(class_names)} ({class_names})")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Num workers: {num_workers}")
        
        return train_dataloader, test_dataloader, class_names
        
    except Exception as e:
        logger.error(f"Failed to create DataLoaders: {str(e)}")
        raise RuntimeError(f"DataLoader creation failed: {str(e)}")


def create_dataloaders_with_validation(
    data_dir: Union[str, Path],
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    batch_size: int,
    val_split: float = 0.2,
    random_seed: int = 42,
    num_workers: int = NUM_WORKERS,
    **kwargs
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Creates training and validation DataLoaders from a single directory by splitting the data.
    
    Args:
        data_dir: Path to directory containing class subdirectories
        train_transform: Transforms for training data (typically includes augmentation)
        val_transform: Transforms for validation data (typically just normalization)
        batch_size: Number of samples per batch
        val_split: Fraction of data to use for validation (0.0 to 1.0)
        random_seed: Random seed for reproducible splits
        num_workers: Number of worker processes
        **kwargs: Additional arguments passed to DataLoader
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, class_names)
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Create full dataset
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in directory: {data_dir}")
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Create random split
    torch.manual_seed(random_seed)
    train_dataset_split, val_dataset_split = random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create datasets with different transforms
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    
    # Get indices from the split
    train_indices = train_dataset_split.indices
    val_indices = val_dataset_split.indices
    
    # Create subsets with appropriate transforms
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **kwargs
    )
    
    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **kwargs
    )
    
    class_names = full_dataset.classes
    
    logger.info(f"Created train/val split:")
    logger.info(f"  Train samples: {len(train_subset)}")
    logger.info(f"  Val samples: {len(val_subset)}")
    logger.info(f"  Classes: {len(class_names)}")
    
    return train_dataloader, val_dataloader, class_names


def get_class_distribution(dataloader: DataLoader) -> Dict[str, int]:
    """
    Analyzes class distribution in a DataLoader.
    
    Args:
        dataloader: PyTorch DataLoader to analyze
        
    Returns:
        Dictionary mapping class indices to sample counts
    """
    class_counts = {}
    
    for _, labels in dataloader:
        for label in labels:
            label_item = label.item()
            class_counts[label_item] = class_counts.get(label_item, 0) + 1
    
    return class_counts


def print_dataloader_info(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    class_names: List[str]
) -> None:
    """
    Prints detailed information about DataLoaders.
    
    Args:
        train_dataloader: Training DataLoader
        test_dataloader: Test DataLoader  
        class_names: List of class names
    """
    print("=" * 50)
    print("DATALOADER INFORMATION")
    print("=" * 50)
    
    # Basic info
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Batch size: {train_dataloader.batch_size}")
    print(f"Number of workers: {train_dataloader.num_workers}")
    print(f"Pin memory: {train_dataloader.pin_memory}")
    
    # Dataset sizes
    train_dataset_size = len(train_dataloader.dataset)
    test_dataset_size = len(test_dataloader.dataset)
    print(f"\nDataset sizes:")
    print(f"  Train: {train_dataset_size} samples ({len(train_dataloader)} batches)")
    print(f"  Test: {test_dataset_size} samples ({len(test_dataloader)} batches)")
    
    # Sample batch shape
    try:
        sample_batch = next(iter(train_dataloader))
        images, labels = sample_batch
        print(f"\nSample batch shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Label dtype: {labels.dtype}")
    except Exception as e:
        print(f"Could not get sample batch: {e}")
    
    print("=" * 50)


def create_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    train_augmentation: bool = True
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Creates standard training and validation transforms for image classification.
    
    Args:
        image_size: Target image size (height, width)
        mean: Normalization mean values for RGB channels
        std: Normalization std values for RGB channels  
        train_augmentation: Whether to apply data augmentation to training data
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Base transforms for all data
    base_transforms = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    
    # Training transforms with optional augmentation
    if train_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        train_transform = transforms.Compose(base_transforms)
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose(base_transforms)
    
    return train_transform, val_transform
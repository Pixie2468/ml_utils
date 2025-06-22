"""
PyTorch Model Training and Utility Functions

This module provides utility functions for PyTorch model management, including
saving/loading models, plotting training results, device management, and other
common ML workflow utilities.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_model(
    model: torch.nn.Module,
    target_dir: Union[str, Path],
    model_name: str,
    save_state_dict_only: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Saves a PyTorch model to a target directory with optional metadata.

    Args:
        model: PyTorch model to save
        target_dir: Directory path for saving the model
        model_name: Filename for the saved model (should include .pth or .pt extension)
        save_state_dict_only: If True, saves only state_dict (recommended).
                              If False, saves entire model (less portable)
        metadata: Optional dictionary with model metadata (accuracy, epoch, etc.)

    Returns:
        Path object pointing to the saved model file

    Raises:
        ValueError: If model_name doesn't have proper extension
        OSError: If directory creation or file saving fails

    Example:
        >>> metadata = {
        ...     'epoch': 50,
        ...     'train_acc': 0.95,
        ...     'test_acc': 0.92,
        ...     'model_architecture': 'ResNet18'
        ... }
        >>> save_path = save_model(
        ...     model=my_model,
        ...     target_dir="models",
        ...     model_name="best_model.pth",
        ...     metadata=metadata
        ... )
    """
    # Validate model name extension
    if not (model_name.endswith(".pth") or model_name.endswith(".pt")):
        raise ValueError("model_name must end with '.pth' or '.pt'")
    
    # Create target directory
    target_dir_path = Path(target_dir)
    try:
        target_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory {target_dir}: {e}")
    
    # Create model save path
    model_save_path = target_dir_path / model_name
    
    try:
        # Prepare save object
        if save_state_dict_only:
            save_obj = {
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'save_timestamp': datetime.now().isoformat()
            }
            if metadata:
                save_obj.update(metadata)
            
            logger.info(f"Saving model state_dict to: {model_save_path}")
            torch.save(save_obj, model_save_path)
        else:
            logger.info(f"Saving entire model to: {model_save_path}")
            torch.save(model, model_save_path)
            
        logger.info(f"Model saved successfully: {model_save_path}")
        return model_save_path
        
    except Exception as e:
        raise OSError(f"Failed to save model: {e}")


def load_model(
    model: torch.nn.Module,
    model_path: Union[str, Path],
    device: torch.device,
    strict: bool = True
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Loads a PyTorch model from a file.
    
    Args:
        model: Model instance to load state into
        model_path: Path to the saved model file
        device: Device to load the model on
        strict: Whether to strictly enforce that the keys match
        
    Returns:
        Tuple of (loaded_model, metadata_dict)
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different save formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Saved with metadata
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
        elif isinstance(checkpoint, dict):
            # Just state_dict
            model.load_state_dict(checkpoint, strict=strict)
            metadata = {}
        else:
            # Entire model saved (not recommended)
            model = checkpoint
            metadata = {}
        
        model.to(device)
        logger.info(f"Model loaded successfully from: {model_path}")
        
        return model, metadata
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Gets the best available device for PyTorch operations.
    
    Args:
        prefer_mps: Whether to prefer MPS (Apple Silicon) over CPU when CUDA unavailable
        
    Returns:
        torch.device object representing the best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif prefer_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Counts the number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def model_summary(model: torch.nn.Module) -> None:
    """
    Prints a summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model to summarize
    """
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model, trainable_only=False):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
    print("\nModel architecture:")
    print(model)
    print("=" * 70)


def plot_training_curves(
    results: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plots training and validation loss and accuracy curves.
    
    Args:
        results: Dictionary with keys 'train_loss', 'train_acc', 'test_loss', 'test_acc'
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    epochs = range(1, len(results['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    ax1.plot(epochs, results['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, results['test_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, results['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, results['test_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to: {save_path}")
    
    plt.show()


def save_training_results(
    results: Dict[str, List[float]],
    save_dir: Union[str, Path],
    filename: str = "training_results.json"
) -> Path:
    """
    Saves training results to a JSON file.
    
    Args:
        results: Dictionary containing training metrics
        save_dir: Directory to save the results
        filename: Name of the JSON file
        
    Returns:
        Path to the saved file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = save_dir / filename
    
    # Add metadata
    results_with_metadata = {
        **results,
        'save_timestamp': datetime.now().isoformat(),
        'total_epochs': len(results.get('train_loss', [])),
        'best_train_acc': max(results.get('train_acc', [0])),
        'best_test_acc': max(results.get('test_acc', [0])),
        'final_train_loss': results.get('train_loss', [0])[-1],
        'final_test_loss': results.get('test_loss', [0])[-1]
    }
    
    with open(save_path, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    logger.info(f"Training results saved to: {save_path}")
    return save_path


def set_seeds(seed: int = 42) -> None:
    """
    Sets random seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Also set Python and NumPy seeds if available
    import random
    random.seed(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    logger.info(f"Random seeds set to: {seed}")


def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculates the memory footprint of a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    model_size = param_size + buffer_size
    
    return {
        'parameters_mb': param_size / (1024 ** 2),
        'buffers_mb': buffer_size / (1024 ** 2),
        'total_mb': model_size / (1024 ** 2)
    }


def early_stopping_check(
    current_loss: float,
    best_loss: float,
    patience_counter: int,
    patience: int,
    min_delta: float = 0.0
) -> Tuple[bool, int, float]:
    """
    Checks if early stopping criteria are met.
    
    Args:
        current_loss: Current epoch's validation loss
        best_loss: Best validation loss so far
        patience_counter: Current patience counter
        patience: Maximum patience (epochs to wait)
        min_delta: Minimum change to qualify as improvement
        
    Returns:
        Tuple of (should_stop, updated_patience_counter, updated_best_loss)
    """
    if current_loss < (best_loss - min_delta):
        # Improvement found
        return False, 0, current_loss
    else:
        # No improvement
        patience_counter += 1
        should_stop = patience_counter >= patience
        return should_stop, patience_counter, best_loss


def freeze_layers(model: torch.nn.Module, layer_names: Optional[List[str]] = None) -> None:
    """
    Freezes specified layers or all layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze. If None, freezes all layers.
    """
    if layer_names is None:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        logger.info("All model parameters frozen")
    else:
        # Freeze specific layers
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
        logger.info(f"Frozen layers: {layer_names}")


def unfreeze_layers(model: torch.nn.Module, layer_names: Optional[List[str]] = None) -> None:
    """
    Unfreezes specified layers or all layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze. If None, unfreezes all layers.
    """
    if layer_names is None:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        logger.info("All model parameters unfrozen")
    else:
        # Unfreeze specific layers
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
        logger.info(f"Unfrozen layers: {layer_names}")


# Legacy function name for backward compatibility
def save_model_legacy(model, target_dir, model_name):
    """Legacy version of save_model for backward compatibility."""
    return save_model(model, target_dir, model_name, save_state_dict_only=True)
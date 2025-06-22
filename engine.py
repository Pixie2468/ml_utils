"""
PyTorch Model Training and Testing Engine

This module provides utilities for training and evaluating PyTorch models with comprehensive
metrics tracking, proper error handling, and flexible scheduler support.
"""

import torch
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_step(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Performs one training epoch on the given model.
    
    Args:
        model: PyTorch model to train
        train_dataloader: DataLoader containing training data
        loss_fn: Loss function (e.g., nn.CrossEntropyLoss())
        optimizer: Optimizer (e.g., torch.optim.Adam())
        device: Device to run training on (cuda/cpu)
        
    Returns:
        Tuple containing (average_loss, average_accuracy) for the epoch
        
    Raises:
        RuntimeError: If model training fails
        ValueError: If dataloader is empty
    """
    if len(train_dataloader) == 0:
        raise ValueError("Training dataloader is empty")
        
    model.train()
    train_loss, train_acc = 0.0, 0.0
    num_batches = len(train_dataloader)

    try:
        for batch_idx, (X, y) in enumerate(train_dataloader):
            # Move data to device
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Forward pass
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            
            # Calculate accuracy (works for both binary and multi-class)
            with torch.no_grad():
                if y_pred.dim() > 1 and y_pred.size(1) > 1:
                    # Multi-class classification
                    y_pred_class = torch.argmax(y_pred, dim=1)
                else:
                    # Binary classification
                    y_pred_class = (y_pred > 0.5).float().squeeze()
                    
                correct = (y_pred_class == y).sum().item()
                train_acc += correct / len(y)

    except Exception as e:
        logger.error(f"Training failed at batch {batch_idx}: {str(e)}")
        raise RuntimeError(f"Training step failed: {str(e)}")

    # Calculate averages
    avg_train_loss = train_loss / num_batches
    avg_train_acc = train_acc / num_batches

    return avg_train_loss, avg_train_acc


def test_step(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Performs evaluation on the test dataset.
    
    Args:
        model: PyTorch model to evaluate
        test_dataloader: DataLoader containing test data
        loss_fn: Loss function used for evaluation
        device: Device to run evaluation on (cuda/cpu)
        
    Returns:
        Tuple containing (average_loss, average_accuracy) for the test set
        
    Raises:
        ValueError: If dataloader is empty
        RuntimeError: If evaluation fails
    """
    if len(test_dataloader) == 0:
        raise ValueError("Test dataloader is empty")
        
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    num_batches = len(test_dataloader)

    try:
        with torch.inference_mode():
            for batch_idx, (X, y) in enumerate(test_dataloader):
                # Move data to device
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                # Forward pass
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                
                # Accumulate loss
                test_loss += loss.item()
                
                # Calculate accuracy
                if y_pred.dim() > 1 and y_pred.size(1) > 1:
                    # Multi-class classification
                    y_pred_class = torch.argmax(y_pred, dim=1)
                else:
                    # Binary classification
                    y_pred_class = (y_pred > 0.5).float().squeeze()
                    
                correct = (y_pred_class == y).sum().item()
                test_acc += correct / len(y)

    except Exception as e:
        logger.error(f"Evaluation failed at batch {batch_idx}: {str(e)}")
        raise RuntimeError(f"Test step failed: {str(e)}")

    # Calculate averages
    avg_test_loss = test_loss / num_batches
    avg_test_acc = test_acc / num_batches

    return avg_test_loss, avg_test_acc


def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    save_best: bool = False,
    save_path: Optional[str] = None,
    patience: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Complete training loop with testing, optional model saving, and early stopping.

    Args:
        model: PyTorch model to train
        train_dataloader: DataLoader for training data
        test_dataloader: DataLoader for test/validation data
        optimizer: Optimizer for training
        loss_fn: Loss function
        epochs: Number of epochs to train
        device: Device to run training on
        scheduler: Optional learning rate scheduler
        save_best: Whether to save the best model based on test loss
        save_path: Path to save the best model (required if save_best=True)
        patience: Early stopping patience (stops if no improvement for N epochs)
        verbose: Whether to print training progress

    Returns:
        Dictionary containing training history with keys:
        - 'train_loss': List of training losses per epoch
        - 'train_acc': List of training accuracies per epoch  
        - 'test_loss': List of test losses per epoch
        - 'test_acc': List of test accuracies per epoch
        - 'learning_rates': List of learning rates per epoch (if scheduler used)

    Raises:
        ValueError: If save_best=True but save_path is None
        RuntimeError: If training fails
    """
    if save_best and save_path is None:
        raise ValueError("save_path must be provided when save_best=True")
    
    # Initialize results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "learning_rates": []
    }
    
    # Early stopping variables
    best_test_loss = float('inf')
    epochs_without_improvement = 0
    
    logger.info(f"Starting training for {epochs} epochs on device: {device}")
    
    try:
        # Training loop with progress bar
        epoch_iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        
        for epoch in epoch_iterator:
            # Training phase
            train_loss, train_acc = train_step(
                model=model,
                train_dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device
            )

            # Testing phase
            test_loss, test_acc = test_step(
                model=model,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device
            )

            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_loss)
                else:
                    scheduler.step()

            # Store metrics
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            results["learning_rates"].append(current_lr)

            # Save best model
            if save_best and test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                }, save_path)
                epochs_without_improvement = 0
                if verbose:
                    logger.info(f"New best model saved with test loss: {test_loss:.4f}")
            else:
                epochs_without_improvement += 1

            # Early stopping
            if patience and epochs_without_improvement >= patience:
                if verbose:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break

            # Print progress
            if verbose:
                print(
                    f"Epoch: {epoch+1:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Test Loss: {test_loss:.4f} | "
                    f"Test Acc: {test_acc:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

    except Exception as e:
        logger.error(f"Training failed at epoch {epoch}: {str(e)}")
        raise RuntimeError(f"Training loop failed: {str(e)}")

    logger.info("Training completed successfully")
    return results


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device
) -> Dict[str, float]:
    """
    Loads a model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        Dictionary with checkpoint metadata
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return {
            'epoch': checkpoint.get('epoch', 0),
            'test_loss': checkpoint.get('test_loss', 0.0),
            'test_acc': checkpoint.get('test_acc', 0.0)
        }
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")


# Legacy function names for backward compatibility
train_data = train_step
test_data = test_step
training_loop = train_model
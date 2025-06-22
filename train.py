"""
PyTorch Training Script

A comprehensive training script that integrates engine.py, data_setup.py, utils.py, and model.py
for training PyTorch models with command-line arguments similar to ImageNet training.

Usage:
    python train.py --lr 0.01 --epochs 100 --batch-size 32 --model resnet18 --data-path ./data
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time

# Note: Future versions may import from torchaudio_models, torchrec_models, etc.
# for audio and recommendation tasks. Update get_model calls accordingly.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

# Import your custom modules
try:
    from engine import train_model, load_checkpoint
    from utils import (
        save_model, get_device, set_seeds, model_summary, 
        plot_training_curves, save_training_results, calculate_model_size
    )
    from torchvision_model import get_model, get_available_models, list_models, model_info
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Make sure engine.py, utils.py, and model.py are in the same directory as train.py")
    sys.exit(1)


def get_data_transforms(dataset: str, input_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Returns data transforms for training and validation.
    
    Args:
        dataset: Dataset name
        input_size: Input image size
        
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    if dataset.lower() in ['cifar10', 'cifar100']:
        # CIFAR datasets use 32x32 images
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    elif dataset.lower() == 'mnist':
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        
    else:
        # ImageNet-style transforms
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return {'train': train_transform, 'val': val_transform}


def get_datasets_and_loaders(args) -> tuple:
    """
    Creates datasets and data loaders based on arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple of (train_loader, val_loader, num_classes)
    """
    transforms_dict = get_data_transforms(args.dataset, args.input_size)
    
    if args.dataset.lower() == 'cifar10':
        train_dataset = CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transforms_dict['train']
        )
        val_dataset = CIFAR10(
            root=args.data_path,
            train=False,
            download=True,
            transform=transforms_dict['val']
        )
        num_classes = 10
        
    elif args.dataset.lower() == 'cifar100':
        train_dataset = CIFAR100(
            root=args.data_path,
            train=True,
            download=True,
            transform=transforms_dict['train']
        )
        val_dataset = CIFAR100(
            root=args.data_path,
            train=False,
            download=True,
            transform=transforms_dict['val']
        )
        num_classes = 100
        
    elif args.dataset.lower() == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=transforms_dict['train']
        )
        val_dataset = torchvision.datasets.MNIST(
            root=args.data_path,
            train=False,
            download=True,
            transform=transforms_dict['val']
        )
        num_classes = 10
        
    else:
        # Custom dataset using ImageFolder
        train_path = Path(args.data_path) / 'train'
        val_path = Path(args.data_path) / 'val'
        
        if not train_path.exists() or not val_path.exists():
            raise ValueError(f"Dataset paths not found: {train_path}, {val_path}")
        
        train_dataset = ImageFolder(train_path, transform=transforms_dict['train'])
        val_dataset = ImageFolder(val_path, transform=transforms_dict['val'])
        num_classes = len(train_dataset.classes)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if args.device.type == 'cuda' else False,
        drop_last=True  # For consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if args.device.type == 'cuda' else False
    )
    
    return train_loader, val_loader, num_classes


def get_optimizer(model: nn.Module, args) -> optim.Optimizer:
    """
    Creates and returns an optimizer based on arguments.
    
    Args:
        model: PyTorch model
        args: Parsed command line arguments
        
    Returns:
        PyTorch optimizer
    """
    if args.optimizer.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov
        )
    elif args.optimizer.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    elif args.optimizer.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    elif args.optimizer.lower() == 'rmsprop':
        return optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def get_scheduler(optimizer: optim.Optimizer, args) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Creates and returns a learning rate scheduler based on arguments.
    
    Args:
        optimizer: PyTorch optimizer
        args: Parsed command line arguments
        
    Returns:
        PyTorch scheduler or None
    """
    if args.scheduler is None:
        return None
    
    if args.scheduler.lower() == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    elif args.scheduler.lower() == 'multistep':
        milestones = [int(x) for x in args.milestones.split(',')]
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    elif args.scheduler.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr
        )
    elif args.scheduler.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.gamma,
            patience=args.scheduler_patience,
            min_lr=args.min_lr
        )
    elif args.scheduler.lower() == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=args.gamma
        )
    elif args.scheduler.lower() == 'warmup_cosine':
        # Custom warmup + cosine annealing
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return epoch / args.warmup_epochs
            else:
                return 0.5 * (1 + torch.cos(torch.pi * (epoch - args.warmup_epochs) / 
                                           (args.epochs - args.warmup_epochs)))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


def get_loss_function(args) -> nn.Module:
    """
    Creates and returns a loss function based on arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        PyTorch loss function
    """
    if args.loss.lower() == 'crossentropy':
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss.lower() == 'mse':
        return nn.MSELoss()
    elif args.loss.lower() == 'mae':
        return nn.L1Loss()
    elif args.loss.lower() == 'bce':
        return nn.BCEWithLogitsLoss()
    elif args.loss.lower() == 'focal':
        # Simple focal loss implementation
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ce = nn.CrossEntropyLoss(reduction='none')
            
            def forward(self, inputs, targets):
                ce_loss = self.ce(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")


def save_config(args, save_path: Path) -> None:
    """
    Saves training configuration to a JSON file.
    
    Args:
        args: Parsed command line arguments
        save_path: Path to save the configuration
    """
    config = {
        'model': args.model,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'loss_function': args.loss,
        'device': str(args.device),
        'seed': args.seed,
        'input_size': args.input_size,
        'pretrained': args.pretrained,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'data_path': args.data_path,
        'output_dir': args.output_dir,
        'resume': args.resume,
        'early_stopping_patience': args.patience,
        'save_best': args.save_best,
        'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
    }
    
    config_path = save_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")


def print_model_comparison(model_names: list, num_classes: int = 10) -> None:
    """
    Print a comparison table of different models.
    
    Args:
        model_names: List of model names to compare
        num_classes: Number of classes for the models
    """
    print("\nModel Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} {'Parameters':<15} {'Size (MB)':<12} {'Status':<10}")
    print("-" * 80)
    
    for model_name in model_names:
        try:
            model = get_model(model_name, num_classes=num_classes)
            info = model_info(model)
            status = "✓"
        except Exception as e:
            info = {'total_parameters': 0, 'model_size_mb': 0}
            status = "✗"
        
        print(f"{model_name:<20} {info['total_parameters']:>12,} "
              f"{info['model_size_mb']:>9.2f} {status:>7}")
    
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Training Script with Model Factory',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', default='resnet18', type=str,
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model')
    parser.add_argument('--input-size', default=224, type=int,
                        help='Input image size')
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and exit')
    parser.add_argument('--compare-models', nargs='+', type=str,
                        help='Compare multiple models and exit')
    
    # Dataset arguments
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='Dataset name (cifar10, cifar100, or custom)')
    parser.add_argument('--data-path', default='./data', type=str,
                        help='Path to dataset')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loading workers')
    
    # Training arguments
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of training epochs')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum for SGD')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--nesterov', action='store_true',
                        help='Use Nesterov momentum')
    
    # Adam/AdamW specific arguments
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='Beta1 for Adam/AdamW')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='Beta2 for Adam/AdamW')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', default='sgd', type=str,
                        choices=['sgd', 'adam', 'adamw', 'rmsprop'],
                        help='Optimizer type')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', default=None, type=str,
                        choices=['step', 'multistep', 'cosine', 'plateau', 'exponential', 'warmup_cosine'],
                        help='Learning rate scheduler')
    parser.add_argument('--step-size', default=30, type=int,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--milestones', default='60,80', type=str,
                        help='Milestones for MultiStepLR (comma-separated)')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Learning rate decay factor')
    parser.add_argument('--min-lr', default=1e-6, type=float,
                        help='Minimum learning rate')
    parser.add_argument('--scheduler-patience', default=10, type=int,
                        help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--warmup-epochs', default=5, type=int,
                        help='Number of warmup epochs for warmup_cosine scheduler')
    
    # Loss function arguments
    parser.add_argument('--loss', default='crossentropy', type=str,
                        choices=['crossentropy', 'mse', 'mae', 'bce', 'focal'],
                        help='Loss function')
    parser.add_argument('--label-smoothing', default=0.0, type=float,
                        help='Label smoothing for CrossEntropyLoss')
    parser.add_argument('--focal-alpha', default=1.0, type=float,
                        help='Alpha parameter for Focal Loss')
    parser.add_argument('--focal-gamma', default=2.0, type=float,
                        help='Gamma parameter for Focal Loss')
    
    # Device and reproducibility
    parser.add_argument('--device', default=None, type=str,
                        help='Device to use (cuda, mps, cpu)')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')
    
    # Model saving and loading
    parser.add_argument('--output-dir', default='./outputs', type=str,
                        help='Output directory for saving models and results')
    parser.add_argument('--save-best', action='store_true',
                        help='Save the best model based on validation loss')
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--save-frequency', default=10, type=int,
                        help='Save checkpoint every N epochs')
    
    # Early stopping
    parser.add_argument('--patience', default=None, type=int,
                        help='Early stopping patience (epochs)')
    
    # Logging and visualization
    parser.add_argument('--print-freq', default=10, type=int,
                        help='Print frequency during training')
    parser.add_argument('--plot-curves', action='store_true',
                        help='Plot and save training curves')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', default='pytorch-training', type=str,
                        help='W&B project name')
    
    # Model-specific arguments
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate for custom models')
    
    args = parser.parse_args()
    
    # Handle utility arguments
    if args.list_models:
        list_models()
        return
    
    if args.compare_models:
        print_model_comparison(args.compare_models)
        return
    
    # Set device
    if args.device is None:
        args.device = get_device()
    else:
        args.device = torch.device(args.device)
    
    # Set random seeds
    set_seeds(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"{args.model}_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
            )
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
            args.wandb = False
    
    # Save configuration
    save_config(args, output_dir)
    
    print("=" * 80)
    print("PYTORCH TRAINING SCRIPT WITH MODEL FACTORY")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Loss function: {args.loss}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Get data loaders
    print("Loading datasets...")
    train_loader, val_loader, num_classes = get_datasets_and_loaders(args)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Create model using the model factory
    print(f"\nCreating model: {args.model}")
    try:
        model_kwargs = {}
        if args.model in ['custom_cnn', 'deep_cnn', 'mini_vgg']:
            model_kwargs['dropout'] = args.dropout
        elif args.model == 'linear':
            # Calculate input size for linear model
            if args.dataset.lower() in ['cifar10', 'cifar100']:
                model_kwargs['input_size'] = 3 * 32 * 32
            else:
                model_kwargs['input_size'] = 3 * args.input_size * args.input_size
        
        model = get_model(
            model_name=args.model,
            num_classes=num_classes,
            pretrained=args.pretrained,
            input_channels=3,
            **model_kwargs
        )
        model = model.to(args.device)
        
        # Print model information
        if args.verbose:
            info = model_info(model)
            print(f"Model: {info['model_name']}")
            print(f"Total parameters: {info['total_parameters']:,}")
            print(f"Trainable parameters: {info['trainable_parameters']:,}")
            print(f"Model size: {info['model_size_mb']:.2f} MB")
            
    except Exception as e:
        print(f"Error creating model: {e}")
        print("\nAvailable models:")
        list_models()
        return
    
    # Create optimizer
    optimizer = get_optimizer(model, args)
    print(f"Optimizer: {optimizer.__class__.__name__}")
    
    # Create scheduler
    scheduler = get_scheduler(optimizer, args)
    if scheduler:
        print(f"Scheduler: {scheduler.__class__.__name__}")
    
    # Create loss function
    loss_fn = get_loss_function(args)
    print(f"Loss function: {loss_fn.__class__.__name__}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint_info = load_checkpoint(model, optimizer, args.resume, args.device)
        start_epoch = checkpoint_info.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Training
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    
    try:
        results = train_model(
            model=model,
            train_dataloader=train_loader,
            test_dataloader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=args.epochs,
            device=args.device,
            scheduler=scheduler,
            save_best=args.save_best,
            save_path=str(output_dir / 'best_model.pth') if args.save_best else None,
            patience=args.patience,
            verbose=args.verbose
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Log to W&B if enabled
        if args.wandb:
            wandb.log({
                'final_train_acc': results['train_acc'][-1],
                'final_val_acc': results['test_acc'][-1],
                'best_val_acc': max(results['test_acc']),
                'training_time': training_time
            })
        
        # Save final model
        final_model_path = save_model(
            model=model,
            target_dir=output_dir,
            model_name='final_model.pth',
            metadata={
                'epoch': args.epochs,
                'train_acc': results['train_acc'][-1],
                'val_acc': results['test_acc'][-1],
                'train_loss': results['train_loss'][-1],
                'val_loss': results['test_loss'][-1],
                'training_time': training_time,
                'model_architecture': args.model,
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'dataset': args.dataset,
                'config': vars(args)
            }
        )
        
        # Save training results
        results_path = save_training_results(results, output_dir)
        
        # Plot training curves
        if args.plot_curves:
            plot_path = output_dir / 'training_curves.png'
            plot_training_curves(results, save_path=plot_path)
        
        # Print final results
        print("\n" + "=" * 80)
        print("TRAINING RESULTS")
        print("=" * 80)
        print(f"Best training accuracy: {max(results['train_acc']):.4f}")
        print(f"Best validation accuracy: {max(results['test_acc']):.4f}")
        print(f"Final training loss: {results['train_loss'][-1]:.4f}")
        print(f"Final validation loss: {results['test_loss'][-1]:.4f}")
        print(f"Total training time: {training_time:.2f} seconds")
        print(f"Final model saved to: {final_model_path}")
        print(f"Training results saved to: {results_path}")
        print("=" * 80)
        
        if args.wandb:
            wandb.finish()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current state
        interrupt_path = output_dir / 'interrupted_model.pth'
        save_model(model, output_dir, 'interrupted_model.pth')
        print(f"Model saved to: {interrupt_path}")
        
        if args.wandb:
            wandb.finish()
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if args.wandb:
            wandb.finish()
        raise


if __name__ == '__main__':
    main()
"""
Example usage of ml_utils for training a custom model on a dataset.
"""

import torch
from ml_utils.data_setup_image import create_dataloaders, create_transforms
from ml_utils.torchvision_models import get_model
from ml_utils.engine import train_model
from ml_utils.utils import get_device, save_model, plot_training_curves, save_training_results

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define transforms
    train_transform, val_transform = create_transforms(
        image_size=(32, 32),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        train_augmentation=True
    )

    # Create dataloaders (replace with your dataset path)
    try:
        train_dl, test_dl, class_names = create_dataloaders(
            train_dir="custom_dataset/train",
            test_dir="custom_dataset/val",
            transform=train_transform,
            batch_size=32,
            num_workers=2
        )
    except FileNotFoundError:
        print("Dataset not found. Using CIFAR10 as fallback.")
        from torchvision.datasets import CIFAR10
        train_dataset = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root="./data", train=False, download=True, transform=val_transform)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        class_names = train_dataset.classes

    # Get device
    device = get_device()

    # Create model
    model = get_model(
        model_name="custom_cnn",
        num_classes=len(class_names),
        input_channels=3,
        dropout=0.5
    ).to(device)

    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train
    results = train_model(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=5,
        device=device,
        save_best=True,
        save_path="outputs/best_model.pth",
        verbose=True
    )

    # Save model and results
    save_model(
        model=model,
        target_dir="outputs",
        model_name="final_model.pth",
        metadata={
            "num_classes": len(class_names),
            "model_name": "custom_cnn",
            "final_test_acc": results["test_acc"][-1]
        }
    )
    
    # Save and plot results
    save_training_results(results, save_dir="outputs")
    plot_training_curves(results, save_path="outputs/training_curves.png")

if __name__ == "__main__":
    main()
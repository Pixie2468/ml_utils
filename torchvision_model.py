"""
Model Factory and Custom Model Definitions

This module provides a centralized way to create and manage PyTorch models,
including both predefined torchvision models and custom architectures.

Author: Generated model factory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict, Any, List
import warnings


class CustomCNN(nn.Module):
    """
    Simple custom CNN for image classification.
    Suitable for datasets like CIFAR-10/100.
    """
    def __init__(self, num_classes: int = 10, input_channels: int = 3, dropout: float = 0.5):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class DeepCNN(nn.Module):
    """
    Deeper custom CNN with residual-like connections.
    """
    def __init__(self, num_classes: int = 10, input_channels: int = 3, dropout: float = 0.5):
        super(DeepCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual-like blocks
        self.block1 = self._make_block(64, 64, 2)
        self.block2 = self._make_block(64, 128, 2)
        self.block3 = self._make_block(128, 256, 2)
        self.block4 = self._make_block(256, 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_block(self, in_channels: int, out_channels: int, num_layers: int) -> nn.Sequential:
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ])
            else:
                layers.extend([
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ])
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class MiniVGG(nn.Module):
    """
    Simplified VGG-like architecture.
    """
    def __init__(self, num_classes: int = 10, input_channels: int = 3, dropout: float = 0.5):
        super(MiniVGG, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LinearModel(nn.Module):
    """
    Simple linear model for testing or baseline comparison.
    """
    def __init__(self, input_size: int, num_classes: int = 10, hidden_sizes: List[int] = None):
        super(LinearModel, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [512, 256]
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        return self.model(x)


# Registry of custom models
CUSTOM_MODELS = {
    'custom_cnn': CustomCNN,
    'deep_cnn': DeepCNN,
    'mini_vgg': MiniVGG,
    'linear': LinearModel
}


def get_model(model_name: str, 
              num_classes: int, 
              pretrained: bool = False,
              input_channels: int = 3,
              **kwargs) -> nn.Module:
    """
    Creates and returns a model based on the model name.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (only for torchvision models)
        input_channels: Number of input channels (for custom models)
        **kwargs: Additional arguments for custom models
        
    Returns:
        PyTorch model
        
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = model_name.lower()
    
    # Handle pretrained parameter deprecation warning
    if pretrained and hasattr(models, model_name.upper() + '_Weights'):
        weights = 'DEFAULT'
    else:
        weights = None
    
    # Torchvision ResNet models
    if model_name == 'resnet18':
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet34':
        if pretrained:
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        else:
            model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet50':
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet101':
        if pretrained:
            model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet152':
        if pretrained:
            model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            model = models.resnet152(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # VGG models
    elif model_name == 'vgg11':
        if pretrained:
            model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
        else:
            model = models.vgg11(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
    elif model_name == 'vgg16':
        if pretrained:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
            model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
    elif model_name == 'vgg19':
        if pretrained:
            model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        else:
            model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    # DenseNet models
    elif model_name == 'densenet121':
        if pretrained:
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name == 'densenet161':
        if pretrained:
            model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        else:
            model = models.densenet161(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name == 'densenet169':
        if pretrained:
            model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        else:
            model = models.densenet169(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # MobileNet models
    elif model_name == 'mobilenet_v2':
        if pretrained:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'mobilenet_v3_small':
        if pretrained:
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        else:
            model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        
    elif model_name == 'mobilenet_v3_large':
        if pretrained:
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        else:
            model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    # EfficientNet models
    elif model_name == 'efficientnet_b0':
        if pretrained:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'efficientnet_b1':
        if pretrained:
            model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        else:
            model = models.efficientnet_b1(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'efficientnet_b2':
        if pretrained:
            model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        else:
            model = models.efficientnet_b2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Vision Transformer models
    elif model_name == 'vit_b_16':
        if pretrained:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        else:
            model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    elif model_name == 'vit_b_32':
        if pretrained:
            model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        else:
            model = models.vit_b_32(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # Custom models
    elif model_name in CUSTOM_MODELS:
        model_class = CUSTOM_MODELS[model_name]
        
        # Handle special case for linear model
        if model_name == 'linear':
            # Assume square input images for simplicity
            input_size = input_channels * 32 * 32  # Default CIFAR size
            if 'input_size' in kwargs:
                input_size = kwargs.pop('input_size')
            model = model_class(input_size=input_size, num_classes=num_classes, **kwargs)
        else:
            model = model_class(num_classes=num_classes, input_channels=input_channels, **kwargs)
            
        if pretrained:
            warnings.warn(f"Pretrained weights not available for custom model: {model_name}")
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: {get_available_models()}")
    
    return model


def get_available_models() -> Dict[str, List[str]]:
    """
    Returns a dictionary of available models categorized by type.
    
    Returns:
        Dictionary with model categories and their available models
    """
    available_models = {
        'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
        'vgg': ['vgg11', 'vgg16', 'vgg19'],
        'densenet': ['densenet121', 'densenet161', 'densenet169'],
        'mobilenet': ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'],
        'efficientnet': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2'],
        'vision_transformer': ['vit_b_16', 'vit_b_32'],
        'custom': list(CUSTOM_MODELS.keys())
    }
    return available_models


def list_models() -> None:
    """
    Prints all available models in a formatted way.
    """
    models_dict = get_available_models()
    
    print("Available Models:")
    print("=" * 50)
    
    for category, model_list in models_dict.items():
        print(f"\n{category.upper()}:")
        for model in model_list:
            print(f"  - {model}")
    
    print(f"\nTotal models available: {sum(len(models) for models in models_dict.values())}")


def model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Returns information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB (assuming float32)
    param_size = total_params * 4  # 4 bytes per float32
    buffer_size = sum(buf.numel() * 4 for buf in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'model_name': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'model_size_bytes': param_size + buffer_size
    }


if __name__ == '__main__':
    # Example usage and testing
    print("Model Factory Test")
    print("=" * 50)
    
    # List all available models
    list_models()
    
    # Test creating some models
    print("\nTesting model creation:")
    
    test_models = ['resnet18', 'custom_cnn', 'vgg16', 'efficientnet_b0']
    
    for model_name in test_models:
        try:
            model = get_model(model_name, num_classes=10)
            info = model_info(model)
            print(f"\n{model_name}:")
            print(f"  Parameters: {info['total_parameters']:,}")
            print(f"  Size: {info['model_size_mb']:.2f} MB")
        except Exception as e:
            print(f"Error creating {model_name}: {e}")
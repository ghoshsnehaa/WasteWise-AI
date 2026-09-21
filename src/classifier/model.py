"""
WasteWise AI - Computer Vision Model Architecture Module

Implements Transfer Learning model builder using PyTorch torchvision models.
By default loads a pretrained ResNet18 model and adapts the final classification layer to 9 outputs.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def build_waste_classifier(
    num_classes: int = 9,
    architecture: str = "resnet18",
    pretrained: bool = True
) -> nn.Module:
    """
    Builds a computer vision classification model using Transfer Learning.

    Args:
        num_classes: Number of output waste categories (default: 9 for RealWaste).
        architecture: Name of base model architecture ('resnet18').
        pretrained: If True, loads ImageNet pretrained weights.

    Returns:
        PyTorch nn.Module ready for training or evaluation.
    """
    if architecture.lower() == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        # ResNet18 output layer is `fc` (Linear layer with 512 input features)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        return model
    else:
        raise ValueError(f"Unsupported architecture '{architecture}'. Currently supported: 'resnet18'")

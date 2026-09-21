"""
WasteWise AI - RealWaste Dataset Loader & Preprocessing Module

Provides PyTorch Dataset and DataLoaders for the RealWaste 9-category dataset based on
data/dataset_splits.json.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_JSON_PATH = PROJECT_ROOT / "data" / "dataset_splits.json"
CLASS_MAP_PATH = PROJECT_ROOT / "data" / "class_to_idx.json"

# Standard 9 RealWaste categories in fixed alphabetical order
STANDARD_CATEGORIES = [
    "cardboard",
    "food_organics",
    "glass",
    "metal",
    "miscellaneous_trash",
    "paper",
    "plastic",
    "textile_trash",
    "vegetation"
]

# ImageNet normalization statistics (required for ResNet18 pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_class_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Creates and saves the fixed class-to-index mapping (e.g. 'cardboard': 0).
    """
    class_to_idx = {cat: idx for idx, cat in enumerate(STANDARD_CATEGORIES)}
    idx_to_class = {idx: cat for cat, idx in class_to_idx.items()}

    CLASS_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)

    return class_to_idx, idx_to_class


def get_transforms(is_training: bool = False) -> transforms.Compose:
    """
    Returns image transformation pipeline.
    
    Args:
        is_training: If True, applies data augmentations (flips, rotations, color jitter).
                     If False, applies only deterministic resize and normalization.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class RealWasteDataset(Dataset):
    """
    PyTorch Dataset for loading RealWaste images using data/dataset_splits.json.
    """

    def __init__(
        self,
        split_name: str,  # 'train', 'validation', or 'test'
        splits_path: Path = SPLITS_JSON_PATH,
        transform: Optional[transforms.Compose] = None
    ):
        self.splits_path = Path(splits_path)
        self.transform = transform
        self.class_to_idx, _ = get_class_mappings()

        if not self.splits_path.exists():
            raise FileNotFoundError(f"Dataset split manifest not found at: {self.splits_path}")

        with open(self.splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        if split_name not in splits_data:
            raise ValueError(f"Invalid split name '{split_name}'. Available: {list(splits_data.keys())}")

        self.image_rel_paths = splits_data[split_name]
        self.samples = []

        for rel_path in self.image_rel_paths:
            full_path = PROJECT_ROOT / rel_path
            # Deduce category from folder name (e.g. data/raw/realwaste/Glass/sample.jpg -> glass)
            folder_name = full_path.parent.name.lower().replace(" ", "_")
            if folder_name in self.class_to_idx:
                label_idx = self.class_to_idx[folder_name]
                self.samples.append((full_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        return img_tensor, label


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Creates train, validation, and test DataLoaders.
    """
    train_dataset = RealWasteDataset("train", transform=get_transforms(is_training=True))
    val_dataset = RealWasteDataset("validation", transform=get_transforms(is_training=False))
    test_dataset = RealWasteDataset("test", transform=get_transforms(is_training=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_to_idx, _ = get_class_mappings()
    return train_loader, val_loader, test_loader, class_to_idx

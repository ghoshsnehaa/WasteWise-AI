"""
WasteWise AI - RealWaste Dataset Exploration Script

Explores and analyzes the RealWaste image dataset across the 9 target categories:
- Cardboard
- Food Organics
- Glass
- Metal
- Miscellaneous Trash
- Paper
- Plastic
- Textile Trash
- Vegetation

Features:
- Validates dataset folder structure
- Counts total images and category breakdown
- Checks image formats, dimensions, corrupt files, and MD5 duplicates
- Generates a 3x3 visual sample grid (saved to data/dataset_samples.png)
- Creates a reproducible Train (70%) / Validation (15%) / Test (15%) split manifest (saved to data/dataset_splits.json)
"""

import os
import json
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "realwaste"
OUTPUT_SPLITS_PATH = PROJECT_ROOT / "data" / "dataset_splits.json"
OUTPUT_PLOT_PATH = PROJECT_ROOT / "data" / "dataset_samples.png"

# Fixed random seed for reproducible train/val/test splits
RANDOM_SEED = 42

# Official 9 RealWaste Category Mappings (Folder Name Variations -> Standardized ID)
CATEGORY_MAPPING = {
    "cardboard": "cardboard",
    "food organics": "food_organics",
    "food_organics": "food_organics",
    "foodorganics": "food_organics",
    "glass": "glass",
    "metal": "metal",
    "miscellaneous trash": "miscellaneous_trash",
    "miscellaneous_trash": "miscellaneous_trash",
    "miscellaneoustrash": "miscellaneoustrash",
    "paper": "paper",
    "plastic": "plastic",
    "textile trash": "textile_trash",
    "textile_trash": "textile_trash",
    "textiletrash": "textile_trash",
    "vegetation": "vegetation"
}

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

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dataset_directory_exists() -> bool:
    """
    Ensures raw dataset directory and 9 category folders exist.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for cat_name in [
        "Cardboard", "Food Organics", "Glass", "Metal",
        "Miscellaneous Trash", "Paper", "Plastic", "Textile Trash", "Vegetation"
    ]:
        (RAW_DATA_DIR / cat_name).mkdir(exist_ok=True)
    return True


def generate_sample_synthetic_images_if_empty():
    """
    Creates dummy test images for each category if the dataset folder is currently empty.
    This allows testing the exploration pipeline before full dataset download.
    """
    ensure_dataset_directory_exists()
    
    # Check if any image files exist
    existing_files = list(RAW_DATA_DIR.rglob("*.*"))
    image_files = [f for f in existing_files if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    
    if len(image_files) == 0:
        print("[INFO] Dataset folder is empty. Generating representative sample images for testing...")
        
        category_colors = {
            "Cardboard": (180, 140, 100),
            "Food Organics": (100, 180, 80),
            "Glass": (120, 200, 220),
            "Metal": (160, 160, 170),
            "Miscellaneous Trash": (120, 120, 120),
            "Paper": (230, 230, 220),
            "Plastic": (240, 180, 80),
            "Textile Trash": (200, 100, 150),
            "Vegetation": (60, 150, 60)
        }
        
        for folder_name, color in category_colors.items():
            cat_dir = RAW_DATA_DIR / folder_name
            cat_dir.mkdir(exist_ok=True)
            for i in range(1, 11):  # Create 10 sample images per category
                img = Image.new("RGB", (224, 224), color=color)
                draw = ImageDraw.Draw(img)
                draw.text((20, 100), f"{folder_name} #{i}", fill=(255, 255, 255))
                img.save(cat_dir / f"sample_{i:02d}.jpg")
        print(f"[OK] Generated 90 sample test images across 9 categories in {RAW_DATA_DIR}")


def scan_dataset() -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    """
    Scans the RAW_DATA_DIR for images and collects metadata.
    """
    images_metadata = []
    category_counts = {cat: 0 for cat in STANDARD_CATEGORIES}
    corrupted_files = []
    md5_hashes = {}
    duplicates = []

    for root, _, files in os.walk(RAW_DATA_DIR):
        folder_name = Path(root).name.lower()
        cat_id = CATEGORY_MAPPING.get(folder_name)
        
        if not cat_id:
            continue

        for filename in files:
            file_path = Path(root) / filename
            ext = file_path.suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                continue

            try:
                with Image.open(file_path) as img:
                    img.verify()  # Validate image integrity
                
                # Re-open after verify() to read dimensions
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode

                # Compute MD5 to check duplicates
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in md5_hashes:
                    duplicates.append((str(file_path), md5_hashes[file_hash]))
                else:
                    md5_hashes[file_hash] = str(file_path)

                category_counts[cat_id] += 1
                images_metadata.append({
                    "path": str(file_path.relative_to(PROJECT_ROOT)),
                    "category": cat_id,
                    "filename": filename,
                    "extension": ext,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "md5": file_hash
                })

            except Exception as err:
                corrupted_files.append(f"{file_path}: {str(err)}")

    return images_metadata, category_counts, corrupted_files, duplicates


def create_stratified_split(images_metadata: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Creates a 70% Train / 15% Val / 15% Test stratified split using a fixed random seed.
    """
    paths = [item["path"] for item in images_metadata]
    labels = [item["category"] for item in images_metadata]

    # Split 70% Train, 30% Temp (Val + Test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=RANDOM_SEED, stratify=labels
    )

    # Split 30% Temp into equal 15% Val and 15% Test
    val_paths, test_paths, _, _ = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_labels
    )

    splits = {
        "random_seed": RANDOM_SEED,
        "total_images": len(paths),
        "split_ratio": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "train": train_paths,
        "validation": val_paths,
        "test": test_paths
    }

    with open(OUTPUT_SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    return splits


def generate_sample_visualization(images_metadata: List[Dict[str, Any]]):
    """
    Saves a 3x3 visualization grid of representative sample images from each category.
    """
    cat_samples = {}
    for item in images_metadata:
        cat = item["category"]
        if cat not in cat_samples:
            cat_samples[cat] = PROJECT_ROOT / item["path"]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("RealWaste Category Representative Samples", fontsize=14, fontweight="bold")

    for idx, cat_id in enumerate(STANDARD_CATEGORIES):
        ax = axes[idx // 3, idx % 3]
        img_path = cat_samples.get(cat_id)

        if img_path and img_path.exists():
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(cat_id.replace("_", " ").title(), fontsize=10, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No Image", ha="center", va="center")
            ax.set_title(cat_id)
            
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH)
    plt.close()


def run_exploration():
    print("==================================================")
    print("      RealWaste Dataset Exploration Report        ")
    print("==================================================")

    # 1. Ensure directory and generate sample test images if empty
    generate_sample_synthetic_images_if_empty()

    # 2. Scan dataset
    images, category_counts, corrupted, duplicates = scan_dataset()

    print(f"\n[1] General Metrics:")
    print(f"    - Total Images Scanned: {len(images)}")
    print(f"    - Corrupted / Unreadable Files: {len(corrupted)}")
    print(f"    - Duplicate Image Content: {len(duplicates)}")

    if corrupted:
        print("\n    Corrupted Files:")
        for c in corrupted:
            print(f"      - {c}")

    print("\n[2] Category Breakdown (9 RealWaste Categories):")
    for cat, count in category_counts.items():
        print(f"    - {cat:20s}: {count:5d} images")

    # 3. Format & Dimension Analysis
    if images:
        formats = set(item["extension"] for item in images)
        widths = [item["width"] for item in images]
        heights = [item["height"] for item in images]
        print("\n[3] Image Properties:")
        print(f"    - File Formats Found: {', '.join(formats)}")
        print(f"    - Dimension Range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)} px")

    # 4. Train / Val / Test Split
    if len(images) >= 9:
        splits = create_stratified_split(images)
        print("\n[4] Train / Validation / Test Stratified Split Plan:")
        print(f"    - Random Seed: {splits['random_seed']}")
        print(f"    - Train Set       (70%): {len(splits['train'])} images")
        print(f"    - Validation Set  (15%): {len(splits['validation'])} images")
        print(f"    - Test Set        (15%): {len(splits['test'])} images")
        print(f"    - Split manifest saved to: data/dataset_splits.json")

    # 5. Visual Grid
    generate_sample_visualization(images)
    print(f"\n[5] Visualization Grid:")
    print(f"    - Sample grid saved to: data/dataset_samples.png")

    print("\n==================================================")
    print("  Dataset Exploration & Split Completed! [OK]     ")
    print("==================================================")


if __name__ == "__main__":
    run_exploration()

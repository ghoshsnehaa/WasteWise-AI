"""
WasteWise AI - Model Training Script

Trains ResNet18 transfer learning baseline on the RealWaste dataset.
Saves the best model checkpoint based on validation accuracy and records training history.
"""

import sys
import argparse
import json
import time
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.classifier.dataset import create_dataloaders
from src.classifier.model import build_waste_classifier

MODEL_SAVE_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_SAVE_DIR / "resnet18_best.pth"
HISTORY_PATH = MODEL_SAVE_DIR / "training_history.json"


def set_seed(seed: int = 42):
    """
    Fixes random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device, max_batches=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, max_batches=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 WasteWise Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 batch test pass without full training")
    args = parser.parse_args()

    set_seed(args.seed)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training device selected: {device}")

    # Load DataLoaders
    train_loader, val_loader, _, class_to_idx = create_dataloaders(batch_size=args.batch_size)
    print(f"[INFO] Dataset splits loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} validation.")

    # Build Model
    model = build_waste_classifier(num_classes=len(class_to_idx), pretrained=True)
    model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.dry_run:
        print("[INFO] Executing Dry Run (1 batch forward/backward pass)...")
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device, max_batches=1)
        v_loss, v_acc = validate(model, val_loader, criterion, device, max_batches=1)
        print(f"[SUCCESS] Dry Run Complete! Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")
        return

    print("==================================================")
    print(f"   Starting Training ({args.epochs} Epochs on {device})   ")
    print("==================================================")

    start_time = time.time()
    best_val_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "epoch_time_sec": round(epoch_time, 2)
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch}/{args.epochs}] ({epoch_time:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
        )

        # Save Best Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  --> Saved new best model checkpoint to {BEST_MODEL_PATH} (Val Acc: {val_acc*100:.2f}%)")

    total_time = time.time() - start_time

    # Save History Manifest
    history_manifest = {
        "random_seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": str(device),
        "total_training_time_sec": round(total_time, 2),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "checkpoint_path": str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "epoch_history": history
    }

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history_manifest, f, indent=2)

    print("==================================================")
    print("           Training Run Completed!                ")
    print(f" Total Time     : {total_time:.2f} seconds ({total_time/60:.2f} mins)")
    print(f" Best Epoch     : Epoch {best_epoch}")
    print(f" Best Val Acc   : {best_val_acc*100:.2f}%")
    print(f" Checkpoint Path: {BEST_MODEL_PATH}")
    print("==================================================")


if __name__ == "__main__":
    main()

"""
WasteWise AI - Model Evaluation Script

Evaluates trained ResNet18 baseline model on the held-out test set (713 images).
Calculates accuracy, per-class metrics, confusion matrix, saves evaluation results JSON,
and generates confusion_matrix.png visualization.
"""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from src.classifier.dataset import RealWasteDataset, get_transforms, get_class_mappings
from src.classifier.model import build_waste_classifier

MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_best.pth"
RESULTS_JSON_PATH = PROJECT_ROOT / "models" / "test_evaluation_results.json"
CONFUSION_MATRIX_PNG_PATH = PROJECT_ROOT / "models" / "confusion_matrix.png"


def plot_and_save_confusion_matrix(cm, class_names, save_path=CONFUSION_MATRIX_PNG_PATH):
    """
    Renders and saves a confusion matrix visualization image.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[c.replace("_", " ").title() for c in class_names],
        yticklabels=[c.replace("_", " ").title() for c in class_names],
        title="RealWaste Test Set - Confusion Matrix",
        ylabel="True Category",
        xlabel="Predicted Category"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating test set on device: {device}")

    class_to_idx, idx_to_class = get_class_mappings()
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # 1. Load Test Dataset
    test_dataset = RealWasteDataset("test", transform=get_transforms(is_training=False))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"[INFO] Loaded {len(test_dataset)} test samples.")

    # 2. Load Model Checkpoint
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")

    model = build_waste_classifier(num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    eval_time = time.time() - start_time

    # 3. Calculate Metrics
    report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    report_str = classification_report(all_targets, all_preds, target_names=class_names)
    cm = confusion_matrix(all_targets, all_preds)

    # 4. Save Confusion Matrix Plot
    plot_and_save_confusion_matrix(cm, class_names)

    # 5. Save JSON Results
    results_payload = {
        "model_checkpoint": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "test_sample_count": len(test_dataset),
        "evaluation_time_sec": round(eval_time, 2),
        "overall_accuracy": round(report_dict["accuracy"], 4),
        "macro_avg": {
            "precision": round(report_dict["macro avg"]["precision"], 4),
            "recall": round(report_dict["macro avg"]["recall"], 4),
            "f1_score": round(report_dict["macro avg"]["f1-score"], 4)
        },
        "weighted_avg": {
            "precision": round(report_dict["weighted avg"]["precision"], 4),
            "recall": round(report_dict["weighted avg"]["recall"], 4),
            "f1_score": round(report_dict["weighted avg"]["f1-score"], 4)
        },
        "per_class": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall": round(report_dict[cls]["recall"], 4),
                "f1_score": round(report_dict[cls]["f1-score"], 4),
                "support": report_dict[cls]["support"]
            }
            for cls in class_names
        },
        "confusion_matrix": cm.tolist()
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("\n==================================================")
    print("      WasteWise AI - Test Set Evaluation          ")
    print("==================================================")
    print(f"Overall Accuracy : {report_dict['accuracy']*100:.2f}%")
    print(f"Evaluation Time  : {eval_time:.2f} seconds")
    print("\nDetailed Classification Report:")
    print(report_str)
    print("\nConfusion Matrix:")
    print(cm)
    print("==================================================")
    print(f"Results saved to: {RESULTS_JSON_PATH}")
    print(f"Confusion matrix plot saved to: {CONFUSION_MATRIX_PNG_PATH}")


if __name__ == "__main__":
    evaluate()

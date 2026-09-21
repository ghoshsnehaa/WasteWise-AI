"""
WasteWise AI - End-to-End Pipeline Wrapper Module

Coordinates the complete workflow:
Image File ➔ ResNet18 Inference ➔ Confidence Gate ➔ RAG Knowledge Retrieval ➔ Prompt Builder ➔ IBM Granite LLM
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Union, Optional

from PIL import Image
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.model import build_waste_classifier
from src.classifier.dataset import get_transforms, get_class_mappings
from src.utils.confidence import evaluate_confidence
from src.rag.retriever import fetch_guidance
from src.rag.prompt_builder import build_rag_prompt
from src.rag.granite_client import IBMGraniteClient

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "resnet18_best.pth"


class WasteWisePipeline:
    """
    End-to-End WasteWise AI pipeline coordinator.
    """

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        threshold: Optional[float] = None,
        force_mock_granite: bool = False
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.class_to_idx, self.idx_to_class = get_class_mappings()
        self.num_classes = len(self.class_to_idx)
        self.transform = get_transforms(is_training=False)

        # Load Vision Classifier Model
        self.model = None
        if self.checkpoint_path.exists():
            try:
                self.model = build_waste_classifier(num_classes=self.num_classes, pretrained=False)
                self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
            except Exception as err:
                print(f"[WARNING] Failed to load model weights from {self.checkpoint_path}: {err}")
                self.model = None

        # Load IBM Granite Client
        self.granite_client = IBMGraniteClient(force_mock=force_mock_granite)

    def analyze_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Runs the full pipeline on a single waste image.

        Args:
            image_path: Path to input image.

        Returns:
            Dict containing predicted category, confidence, reliability, retrieved knowledge, and Granite guidance.
        """
        img_path = Path(image_path)

        # 1. Validate File Existence
        if not img_path.exists():
            return {
                "status": "error_file_not_found",
                "is_reliable": False,
                "error_message": f"Image file not found at: {img_path}",
                "user_message": "Error: Selected image file could not be found.",
                "granite_called": False,
                "granite_response": None
            }

        # 2. Open & Validate Image Integrity
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as err:
            return {
                "status": "error_invalid_image",
                "is_reliable": False,
                "error_message": f"Corrupt or invalid image file: {err}",
                "user_message": "Error: Unable to read image file. Please upload a valid JPEG/PNG image.",
                "granite_called": False,
                "granite_response": None
            }

        # 3. Validate Checkpoint Availability
        if self.model is None:
            return {
                "status": "error_missing_checkpoint",
                "is_reliable": False,
                "error_message": f"Model weights missing at: {self.checkpoint_path}",
                "user_message": "Error: Trained model weights missing. Please ensure models/resnet18_best.pth exists.",
                "granite_called": False,
                "granite_response": None
            }

        # 4. Perform ResNet18 Inference
        try:
            img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(img_tensor)
                probabilities = F.softmax(logits, dim=1)[0]
                top_prob, top_idx = torch.max(probabilities, dim=0)

            confidence_score = float(top_prob.item())
            predicted_cat_id = self.idx_to_class[int(top_idx.item())]
        except Exception as err:
            return {
                "status": "error_inference_failed",
                "is_reliable": False,
                "error_message": f"Inference failed: {err}",
                "user_message": "Error processing image through neural network classifier.",
                "granite_called": False,
                "granite_response": None
            }

        # 5. Evaluate Confidence Gate
        kwargs = {} if self.threshold is None else {"threshold": self.threshold}
        conf_eval = evaluate_confidence(confidence_score, **kwargs)

        # 6. Retrieve Knowledge & Build Prompt Package
        knowledge_entry = fetch_guidance(predicted_cat_id)
        prompt_package = build_rag_prompt(
            predicted_category=predicted_cat_id,
            confidence_score=confidence_score,
            confidence_result=conf_eval,
            knowledge_entry=knowledge_entry,
            threshold=self.threshold
        )

        # 7. Generate Response via IBM Granite Client
        granite_result = self.granite_client.generate_response(prompt_package)

        display_name = (
            knowledge_entry.get("display_name", predicted_cat_id.replace("_", " ").title())
            if knowledge_entry else predicted_cat_id.title()
        )

        return {
            "status": granite_result.get("status", "unknown"),
            "image_path": str(img_path),
            "predicted_category": predicted_cat_id,
            "display_name": display_name,
            "confidence": round(confidence_score, 4),
            "confidence_percentage": round(confidence_score * 100, 2),
            "confidence_status": conf_eval.get("status"),
            "is_reliable": conf_eval.get("is_reliable", False),
            "retrieved_knowledge": knowledge_entry,
            "granite_called": granite_result.get("granite_called", False),
            "is_live_service": granite_result.get("is_live_service", False),
            "granite_response": granite_result.get("response_text"),
            "user_message": granite_result.get("response_text")
        }


def run_cli_demo(image_path: str, threshold: Optional[float] = None):
    """
    CLI demonstration printer.
    """
    pipeline = WasteWisePipeline(threshold=threshold)
    result = pipeline.analyze_image(image_path)

    print("\n==================================================")
    print("           WasteWise AI Analysis Result           ")
    print("==================================================")

    if result["status"].startswith("error"):
        print(f"Status           : ERROR ({result['status']})")
        print(f"Message          : {result['user_message']}")
        print("==================================================\n")
        return

    print(f"Image Path       : {result['image_path']}")
    print(f"Prediction       : {result['display_name']} ({result['predicted_category']})")
    print(f"Confidence       : {result['confidence_percentage']}%")
    print(f"Confidence Status: {result['confidence_status'].upper()}")
    print(f"Is Reliable      : {result['is_reliable']}")
    print(f"Granite Called   : {'Yes' if result['granite_called'] else 'No'} (Live: {result['is_live_service']})")
    print("--------------------------------------------------")
    print("Guidance / Explanation:\n")
    print(result['granite_response'])
    print("==================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WasteWise AI End-to-End Pipeline CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to waste image file")
    parser.add_argument("--threshold", type=float, default=None, help="Optional confidence threshold override")
    args = parser.parse_args()

    run_cli_demo(args.image, threshold=args.threshold)

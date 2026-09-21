"""
Tests for WasteWise AI End-to-End Pipeline
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import WasteWisePipeline


def test_pipeline_nonexistent_image():
    pipeline = WasteWisePipeline(force_mock_granite=True)
    res = pipeline.analyze_image("nonexistent_path_xyz.jpg")
    assert res["status"] == "error_file_not_found"
    assert res["is_reliable"] is False
    assert res["granite_called"] is False
    print("[OK] Test 1 Passed: Nonexistent image path handled safely.")


def test_pipeline_high_confidence_flow():
    # Use real image from dataset
    real_image_path = PROJECT_ROOT / "data" / "raw" / "realwaste" / "Plastic" / "Plastic_1.jpg"
    if not real_image_path.exists():
        print("[SKIP] Test 2 Skipped: Plastic_1.jpg not found.")
        return

    pipeline = WasteWisePipeline(force_mock_granite=True)
    res = pipeline.analyze_image(real_image_path)

    assert res["status"] == "success"
    assert res["predicted_category"] == "plastic"
    assert res["confidence"] > 0.0
    assert res["is_reliable"] is True
    assert res["granite_called"] is True
    assert "Plastic" in res["granite_response"]
    print(f"[OK] Test 2 Passed: High-confidence image flow succeeded (Confidence: {res['confidence_percentage']}%).")


def test_pipeline_low_confidence_suppression():
    real_image_path = PROJECT_ROOT / "data" / "raw" / "realwaste" / "Plastic" / "Plastic_1.jpg"
    if not real_image_path.exists():
        print("[SKIP] Test 3 Skipped: Plastic_1.jpg not found.")
        return

    # Set threshold unnaturally high (0.9999) to force low_confidence branch
    pipeline = WasteWisePipeline(threshold=0.9999, force_mock_granite=True)
    res = pipeline.analyze_image(real_image_path)

    assert res["is_reliable"] is False
    assert res["granite_called"] is False
    assert "uncertain" in res["user_message"].lower()
    print("[OK] Test 3 Passed: Low confidence threshold correctly suppressed Granite call.")


def test_pipeline_missing_checkpoint_handling():
    fake_checkpoint = PROJECT_ROOT / "models" / "nonexistent_model.pth"
    pipeline = WasteWisePipeline(checkpoint_path=fake_checkpoint, force_mock_granite=True)
    
    real_image_path = PROJECT_ROOT / "data" / "raw" / "realwaste" / "Plastic" / "Plastic_1.jpg"
    res = pipeline.analyze_image(real_image_path)
    assert res["status"] == "error_missing_checkpoint"
    assert res["is_reliable"] is False
    assert res["granite_called"] is False
    print("[OK] Test 4 Passed: Missing checkpoint handled safely.")


if __name__ == "__main__":
    print("--- Running End-to-End Pipeline Tests ---")
    test_pipeline_nonexistent_image()
    test_pipeline_high_confidence_flow()
    test_pipeline_low_confidence_suppression()
    test_pipeline_missing_checkpoint_handling()
    print("[SUCCESS] All pipeline tests passed successfully!")

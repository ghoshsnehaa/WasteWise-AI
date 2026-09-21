"""
Tests for WasteWise AI Confidence & Uncertainty Evaluation Utility
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.confidence import evaluate_confidence, is_valid_confidence


def test_confidence_evaluation():
    threshold = 0.70

    # 1. High confidence value (e.g., 0.85)
    res_high = evaluate_confidence(0.85, threshold=threshold)
    assert res_high["status"] == "high_confidence"
    assert res_high["is_reliable"] is True
    print("[OK] Test 1 Passed: High confidence score (0.85) returned 'high_confidence'.")

    # 2. Low confidence value (e.g., 0.45)
    res_low = evaluate_confidence(0.45, threshold=threshold)
    assert res_low["status"] == "low_confidence"
    assert res_low["is_reliable"] is False
    print("[OK] Test 2 Passed: Low confidence score (0.45) returned 'low_confidence'.")

    # 3. Value exactly at threshold (0.70)
    res_exact = evaluate_confidence(0.70, threshold=threshold)
    assert res_exact["status"] == "high_confidence"
    assert res_exact["is_reliable"] is True
    print("[OK] Test 3 Passed: Score exactly at threshold (0.70) returned 'high_confidence'.")

    # 4. Boundary value 0 (0.0)
    res_zero = evaluate_confidence(0.0, threshold=threshold)
    assert res_zero["status"] == "low_confidence"
    assert res_zero["is_reliable"] is False
    print("[OK] Test 4 Passed: Boundary score 0.0 returned 'low_confidence'.")

    # 5. Boundary value 1 (1.0)
    res_one = evaluate_confidence(1.0, threshold=threshold)
    assert res_one["status"] == "high_confidence"
    assert res_one["is_reliable"] is True
    print("[OK] Test 5 Passed: Boundary score 1.0 returned 'high_confidence'.")


def test_invalid_confidence_values():
    # 6. Negative score (-0.1)
    res_neg = evaluate_confidence(-0.1)
    assert res_neg["status"] == "invalid_input"
    assert res_neg["is_reliable"] is False
    print("[OK] Test 6 Passed: Negative score (-0.1) safely handled as 'invalid_input'.")

    # 7. Score greater than 1 (1.1)
    res_over = evaluate_confidence(1.1)
    assert res_over["status"] == "invalid_input"
    assert res_over["is_reliable"] is False
    print("[OK] Test 7 Passed: Over-range score (1.1) safely handled as 'invalid_input'.")

    # 8. Non-numeric score ("high")
    res_str = evaluate_confidence("high")
    assert res_str["status"] == "invalid_input"
    print("[OK] Test 8 Passed: String input handled safely as 'invalid_input'.")


if __name__ == "__main__":
    print("--- Running Confidence Evaluation Tests ---")
    test_confidence_evaluation()
    test_invalid_confidence_values()
    print("[SUCCESS] All confidence tests passed successfully!")

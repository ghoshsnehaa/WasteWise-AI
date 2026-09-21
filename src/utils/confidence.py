"""
WasteWise AI - Confidence & Uncertainty Evaluation Module

Provides utilities to evaluate model prediction confidence against a configurable
threshold and flag low-confidence predictions for Responsible AI uncertainty handling.
"""

from typing import Dict, Any

# Default initial threshold used during development.
# NOTE: The final production threshold will be tuned after evaluating our model on real images.
DEFAULT_THRESHOLD = 0.70


def is_valid_confidence(score: Any) -> bool:
    """
    Validates whether an input is a valid confidence score (number between 0.0 and 1.0).
    """
    if isinstance(score, bool):  # Booleans inherit from int in Python
        return False
    if not isinstance(score, (int, float)):
        return False
    return 0.0 <= score <= 1.0


def evaluate_confidence(
    score: float,
    threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, Any]:
    """
    Evaluates a model prediction confidence score against a threshold.

    Args:
        score: Model prediction confidence score (float between 0.0 and 1.0).
        threshold: Minimum score required to accept prediction (default: 0.70).

    Returns:
        Dict containing:
          - 'status': 'high_confidence', 'low_confidence', or 'invalid_input'
          - 'is_reliable': bool (True if high confidence, False otherwise)
          - 'score': float or original input
          - 'threshold': float or original threshold
          - 'message': User-friendly description
    """
    # 1. Input Validation
    if not is_valid_confidence(score) or not is_valid_confidence(threshold):
        return {
            "status": "invalid_input",
            "is_reliable": False,
            "score": score,
            "threshold": threshold,
            "message": f"Invalid score ({score}) or threshold ({threshold}). Expected a float between 0.0 and 1.0."
        }

    score_val = float(score)
    threshold_val = float(threshold)

    # 2. Threshold Comparison
    if score_val >= threshold_val:
        return {
            "status": "high_confidence",
            "is_reliable": True,
            "score": score_val,
            "threshold": threshold_val,
            "message": f"Confidence ({score_val:.2f}) meets or exceeds threshold ({threshold_val:.2f})."
        }
    else:
        return {
            "status": "low_confidence",
            "is_reliable": False,
            "score": score_val,
            "threshold": threshold_val,
            "message": f"Confidence ({score_val:.2f}) is below threshold ({threshold_val:.2f}). Prompt for clearer image."
        }

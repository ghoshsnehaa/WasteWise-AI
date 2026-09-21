"""
Integration Tests for IBM Granite LLM Client
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.granite_client import IBMGraniteClient, generate_waste_guidance
from src.rag.prompt_builder import build_rag_prompt


def test_high_confidence_granite_flow():
    # 1. High confidence test (82% confidence, 70% threshold)
    res = generate_waste_guidance("plastic", 0.82, threshold=0.70)
    assert res["status"] == "success"
    assert res["is_reliable"] is True
    assert res["granite_called"] is True
    assert "Plastic" in res["response_text"] or "plastic" in res["response_text"]
    assert "is_live_service" in res
    print("[OK] Test 1 Passed: High-confidence flow successfully invoked Granite client.")


def test_low_confidence_suppression():
    # 2. Low confidence test (45% confidence, 70% threshold)
    # MUST NOT call Granite for disposal advice
    res = generate_waste_guidance("plastic", 0.45, threshold=0.70)
    assert res["status"] == "low_confidence_uncertainty"
    assert res["is_reliable"] is False
    assert res["granite_called"] is False  # Verified: Granite was NOT called
    assert "uncertain" in res["response_text"].lower()
    print("[OK] Test 2 Passed: Low-confidence prediction correctly suppressed Granite call.")


def test_unknown_category_flow():
    # 3. Unknown category test
    res = generate_waste_guidance("unknown_item", 0.90)
    assert res["status"] == "unknown_category"
    assert res["is_reliable"] is False
    assert res["granite_called"] is False
    assert "unknown_item" in res["response_text"]
    print("[OK] Test 3 Passed: Unknown category correctly suppressed Granite call.")


def test_explicit_mock_mode():
    # 4. Explicit mock client initialization
    client = IBMGraniteClient(force_mock=True)
    assert client.is_live is False
    
    prompt_pkg = build_rag_prompt("glass", 0.88, threshold=0.70)
    res = client.generate_response(prompt_pkg)
    assert res["status"] == "success"
    assert res["is_live_service"] is False
    print("[OK] Test 4 Passed: Explicit mock mode operates safely.")


if __name__ == "__main__":
    print("--- Running IBM Granite Client Integration Tests ---")
    test_high_confidence_granite_flow()
    test_low_confidence_suppression()
    test_unknown_category_flow()
    test_explicit_mock_mode()
    print("[SUCCESS] All IBM Granite client integration tests passed successfully!")

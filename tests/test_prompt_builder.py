"""
Tests for WasteWise AI RAG Prompt Builder Module
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.prompt_builder import build_rag_prompt


def test_high_confidence_prompt_building():
    # 1. Test plastic with 82% confidence (Threshold 70%)
    res = build_rag_prompt("plastic", 0.82, threshold=0.70)
    assert res["status"] == "success"
    assert res["is_reliable"] is True
    assert res["display_name"] == "Plastic"
    assert res["llm_prompt"] is not None
    assert "[RETRIEVED FACTUAL KNOWLEDGE BASE]" in res["llm_prompt"]
    assert "Plastic" in res["llm_prompt"]
    print("[OK] Test 1 Passed: High confidence (82%) generated Granite-ready prompt.")


def test_low_confidence_uncertainty_handling():
    # 2. Test plastic with 45% confidence (Threshold 70%)
    res = build_rag_prompt("plastic", 0.45, threshold=0.70)
    assert res["status"] == "low_confidence_uncertainty"
    assert res["is_reliable"] is False
    assert res["llm_prompt"] is None  # Should NOT generate LLM prompt on low confidence
    assert "uncertain" in res["user_message"].lower()
    print("[OK] Test 2 Passed: Low confidence (45%) safely triggered uncertainty response.")


def test_unknown_category_handling():
    # 3. Test unknown category string
    res = build_rag_prompt("unknown_item", 0.90)
    assert res["status"] == "unknown_category"
    assert res["is_reliable"] is False
    assert res["llm_prompt"] is None
    assert "unknown_item" in res["user_message"]
    print("[OK] Test 3 Passed: Unknown category string safely handled.")


def test_missing_knowledge_handling():
    # 4. Test passing explicitly empty knowledge_entry
    res = build_rag_prompt("plastic", 0.85, knowledge_entry=None)
    # Since plastic exists in categories.json, fetch_guidance will load it
    assert res["status"] == "success"

    # Test passing empty dict knowledge entry
    res_empty = build_rag_prompt("plastic", 0.85, knowledge_entry={})
    assert res_empty["status"] == "unknown_category"
    print("[OK] Test 4 Passed: Missing knowledge handling verified.")


if __name__ == "__main__":
    print("--- Running RAG Prompt Builder Tests ---")
    test_high_confidence_prompt_building()
    test_low_confidence_uncertainty_handling()
    test_unknown_category_handling()
    test_missing_knowledge_handling()
    print("[SUCCESS] All RAG prompt builder tests passed successfully!")

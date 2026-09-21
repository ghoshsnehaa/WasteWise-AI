"""
Tests for WasteWise AI Knowledge Base Retriever
"""

import sys
from pathlib import Path

# Add project root to Python sys.path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import KnowledgeBaseRetriever, fetch_guidance


def test_valid_category_retrieval():
    retriever = KnowledgeBaseRetriever()

    # Test 1: Retrieve 'plastic'
    plastic_info = retriever.get_category_guidance("plastic")
    assert plastic_info is not None, "Plastic guidance should not be None"
    assert plastic_info["display_name"] == "Plastic"
    assert "Dry Waste" in plastic_info["general_disposal_category"]
    print("[OK] Test 1 Passed: Successfully retrieved 'plastic' category guidance.")

    # Test 2: Retrieve 'glass'
    glass_info = retriever.get_category_guidance("glass")
    assert glass_info is not None, "Glass guidance should not be None"
    assert glass_info["display_name"] == "Glass"
    assert "guidance" in glass_info
    print("[OK] Test 2 Passed: Successfully retrieved 'glass' category guidance.")

    # Test 3: Retrieve 'food_organics'
    food_info = fetch_guidance("food_organics")
    assert food_info is not None, "Food organics guidance should not be None"
    assert food_info["display_name"] == "Food Organics"
    print("[OK] Test 3 Passed: Successfully retrieved 'food_organics' using fetch_guidance().")


def test_invalid_category_handling():
    retriever = KnowledgeBaseRetriever()

    # Test 4: Unknown category ID 'unknown_garbage'
    invalid_result = retriever.get_category_guidance("unknown_garbage")
    assert invalid_result is None, "Invalid category should return None safely"
    print("[OK] Test 4 Passed: Safely handled unknown category 'unknown_garbage' (returned None).")

    # Test 5: Empty input
    empty_result = retriever.get_category_guidance("")
    assert empty_result is None, "Empty category string should return None safely"
    print("[OK] Test 5 Passed: Safely handled empty string input.")


if __name__ == "__main__":
    print("--- Running Knowledge Base Retriever Tests ---")
    test_valid_category_retrieval()
    test_invalid_category_handling()
    print("[SUCCESS] All tests passed successfully!")

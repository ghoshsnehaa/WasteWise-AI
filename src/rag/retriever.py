"""
WasteWise AI - Knowledge Base Retriever Module

This module provides a simple, deterministic retriever that loads waste disposal
guidance from data/knowledge_base/categories.json for any of the 9 RealWaste categories.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

# Automatically determine the project root directory
# Path(__file__) = src/rag/retriever.py
# .parent = src/rag
# .parent.parent = src
# .parent.parent.parent = WasteWise-AI (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_KB_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "categories.json"


class KnowledgeBaseRetriever:
    """
    Loads and searches the WasteWise AI knowledge base.
    """

    def __init__(self, json_path: Path = DEFAULT_KB_PATH):
        self.json_path = Path(json_path)
        self.categories_index: Dict[str, Dict[str, Any]] = {}
        self._load_knowledge_base()

    def _load_knowledge_base(self) -> None:
        """
        Reads the categories.json file and indexes all categories by ID.
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found at: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Index each category by its lowercased ID for fast and flexible lookups
        for category in data.get("categories", []):
            cat_id = category.get("id", "").strip().lower()
            if cat_id:
                self.categories_index[cat_id] = category

    def get_category_guidance(self, category_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches disposal guidance for a given category ID.

        Args:
            category_id: Category string (e.g. 'plastic', 'glass', 'food_organics')

        Returns:
            The category dictionary if found, or None if category is invalid/unknown.
        """
        if not isinstance(category_id, str):
            return None

        # Clean and normalize input (e.g. "  Plastic  " -> "plastic")
        clean_id = category_id.strip().lower()

        # Safely return the category data or None if not found
        return self.categories_index.get(clean_id, None)


# Helper function for quick single-call usage
def fetch_guidance(category_id: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to get guidance directly without manually instantiating the class.
    """
    retriever = KnowledgeBaseRetriever()
    return retriever.get_category_guidance(category_id)

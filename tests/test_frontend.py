"""
Tests for WasteWise AI Streamlit Web Frontend
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_frontend_module_import():
    import web.app as web_app
    assert hasattr(web_app, "main")
    assert hasattr(web_app, "load_pipeline")
    print("[OK] Test 1 Passed: Streamlit frontend web.app imported successfully.")


def test_root_app_entrypoint_import():
    import app
    assert hasattr(app, "main")
    print("[OK] Test 2 Passed: Root app.py entrypoint imported successfully.")


if __name__ == "__main__":
    print("--- Running Frontend Tests ---")
    test_frontend_module_import()
    test_root_app_entrypoint_import()
    print("[SUCCESS] All frontend tests passed successfully!")

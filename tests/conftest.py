"""
Shared pytest setup: makes src/ and app/ importable, and mocks the
streamlit module so app.py can be imported without a real Streamlit
runtime (app.py calls st.set_page_config() and other Streamlit
functions at module import time).
"""
import os
import sys
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
APP_DIR = os.path.join(PROJECT_ROOT, "app")

for path in (SRC_DIR, APP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

sys.modules.setdefault("streamlit", MagicMock())
# tests/conftest.py
import sys
import os

# Add the project root directory (parent of 'tests' and 'src') to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# You can add fixtures or other pytest hooks here later if needed 
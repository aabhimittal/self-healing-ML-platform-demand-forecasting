"""Shared import bootstrap for the stdlib test runner.

``conftest.py`` handles the path for pytest; this module does the same when the
tests are executed by ``tests/run_tests.py`` without pytest installed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

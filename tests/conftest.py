# conftest.py — shared pytest configuration for HoudiniMLPose tests
import sys
import os

# Ensure source modules and fixtures are importable
SRC = os.path.join(os.path.dirname(__file__), "..", "src", "PoseExtractorHDA", "python")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

for p in (SRC, FIXTURES):
    if p not in sys.path:
        sys.path.insert(0, p)

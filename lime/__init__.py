"""Project-vendored LIME compatibility layer used only when the external lime package is unavailable.

This exposes the LimeTabularExplainer API needed by Phase 3B. It is intentionally small and
records its implementation provenance in the Phase 3B metadata.
"""
__version__ = "vendored-compat-0.2.0.1"
from .lime_tabular import LimeTabularExplainer

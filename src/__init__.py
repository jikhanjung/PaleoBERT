"""
PaleoBERT source package.

Modules:
    normalization: Text normalization with character-level alignment maps
"""

from .normalization import normalize_text, project_span, create_inverse_map

__all__ = ["normalize_text", "project_span", "create_inverse_map"]

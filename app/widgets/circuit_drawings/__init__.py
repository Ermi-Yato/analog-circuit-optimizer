"""
Circuit schematic drawings using schemdraw.

Public API
----------
get_drawing(circuit_id) -> matplotlib.figure.Figure | None
    Returns a dark-themed schematic figure for the given circuit ID,
    or None if no drawing is registered for that ID.
"""
from .drawings import get_drawing

__all__ = ["get_drawing"]

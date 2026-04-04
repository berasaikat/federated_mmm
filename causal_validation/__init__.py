"""
Causal validation: geo data, matching, synthetic control, and incrementality audits.
"""

from .audit import run_incrementality_audit
from .geo_loader import load_geo_data
from .geo_matcher import GeoMatcher
from .synthetic_control import estimate_incrementality, fit_synthetic_control

__all__ = [
    "run_incrementality_audit",
    "load_geo_data",
    "GeoMatcher",
    "fit_synthetic_control",
    "estimate_incrementality",
]

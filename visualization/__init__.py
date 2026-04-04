"""
Diagnostic plots for federated rounds, privacy budgets, surprise, and audit results.
"""

from .audit_chart import plot_audit_results
from .posterior_plots import plot_posterior_evolution
from .privacy_plots import plot_budget_consumption
from .surprise_heatmap import plot_surprise_heatmap

__all__ = [
    "plot_posterior_evolution",
    "plot_budget_consumption",
    "plot_surprise_heatmap",
    "plot_audit_results",
]

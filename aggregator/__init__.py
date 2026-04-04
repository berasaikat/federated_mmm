"""
Federated aggregation: training loop, rounds, posterior pooling, and Flower hooks.
"""

from .convergence import check_convergence, compute_convergence_metrics
from .fed_avg_posterior import fedavg_posterior
from .federated_loop import run_federated_training
from .flower_strategy import FederatedMMMStrategy
from .hierarchical import hierarchical_pool
from .round_manager import RoundManager
from .simulate import run_simulation

__all__ = [
    "run_federated_training",
    "RoundManager",
    "FederatedMMMStrategy",
    "fedavg_posterior",
    "hierarchical_pool",
    "check_convergence",
    "compute_convergence_metrics",
    "run_simulation",
]

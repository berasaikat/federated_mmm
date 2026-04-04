"""
Local Bayesian MMM training for federated participants.
Exposes LocalTrainer, MMMClient, and inference utilities.
"""

from .flower_client import MMMClient
from .inference import run_mcmc
from .local_trainer import LocalTrainer
from .mmm_model import mmm_numpyro
from .posterior import (
    deserialize_posterior_summary,
    extract_posterior_summary,
    serialize_posterior_summary,
)

__all__ = [
    "LocalTrainer",
    "MMMClient",
    "mmm_numpyro",
    "run_mcmc",
    "extract_posterior_summary",
    "serialize_posterior_summary",
    "deserialize_posterior_summary",
]

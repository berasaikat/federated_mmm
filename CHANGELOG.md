# Changelog

## [0.1.0] — 2026-04-04

### Added
- Federated Bayesian MMM with NumPyro/JAX local training
- LLM-guided prior elicitation via Claude API with KL surprise feedback loop
- Differential privacy via Gaussian mechanism with per-participant budget tracking
- Hierarchical posterior aggregation (FedAvg + shrinkage estimator)
- Flower (flwr) federated simulation integration
- Synthetic control incrementality audit with LLM-assisted geo matching
- Full experiment logging (JSONL rounds, priors, audits)
- CLI via run.py (generate-data, train, validate, visualize, report)
- 40+ smoke tests across all phases
- Integration and regression math test suites
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ExperimentLogger:
    def __init__(self, experiment_id: str, output_dir: str):
        """
        Initializes the Experiment Logging conduit to meticulously track Federated operations and Causal validations.
        
        Args:
            experiment_id: A unique categorical identifier or UUID tracking the holistic operation sequence.
            output_dir: The root target file directory path where sub-environments resolve statically.
        """
        self.experiment_id = str(experiment_id)
        self.base_dir = Path(output_dir) / self.experiment_id
        self.logs_dir = self.base_dir / "logs"
        
        # Instantiate structural filesystem constraints securely
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Map endpoints for specific append-only JSONL files
        self.rounds_log_path = self.logs_dir / "rounds.jsonl"
        self.priors_log_path = self.logs_dir / "priors.jsonl"
        self.audits_log_path = self.logs_dir / "audits.jsonl"
        
        # Establish deterministic tracking parameters caching the aggregate session counts
        self.summary = {
            "experiment_id": self.experiment_id,
            "datetime_start": datetime.utcnow().isoformat(),
            "metrics": {
                "total_rounds_logged": 0,
                "total_priors_elicited": 0,
                "total_audits_logged": 0,
                "epsilon_spent_cumulated": 0.0
            }
        }
        
        logger.info(f"Experiment tracking actively initialized: resolving locally to {self.logs_dir.resolve()}")

    def _append_jsonl(self, filepath: Path, payload: Dict[str, Any]):
        """Helper encoding internal Python Dict arrays robustly into single-line NDJSON syntax."""
        with open(filepath, "a") as f:
            # Ensures isolated Numpy constraints successfully collapse uniformly against base JSON bounds `default=float`
            f.write(json.dumps(payload, default=float) + "\n")

    def log_round(self, round_num: int, global_summary: Dict[str, Any], surprise_scores: Dict[str, float], epsilon_spent: float):
        """
        Tracks boundaries capturing mathematical outcomes bridging identical federated rounds sequentially.
        """
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "round_num": round_num,
            "global_summary": global_summary,
            "surprise_scores": surprise_scores,
            "epsilon_spent": epsilon_spent
        }
        self._append_jsonl(self.rounds_log_path, payload)
        
        # Increment tracking cache
        self.summary["metrics"]["total_rounds_logged"] += 1
        self.summary["metrics"]["epsilon_spent_cumulated"] += epsilon_spent

    def log_priors(self, round_num: int, participant_id: str, priors_dict: Dict[str, Any]):
        """
        Logs explicitly established generative inferences structured via LLMs mapping exactly to Bayesian MCMC operations.
        """
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "round_num": round_num,
            "participant_id": str(participant_id),
            "priors": priors_dict
        }
        self._append_jsonl(self.priors_log_path, payload)
        
        self.summary["metrics"]["total_priors_elicited"] += 1

    def log_audit(self, audit_result: Dict[str, Any]):
        """
        Logs validations mathematically binding pure causal constraints explicitly derived off true observational divergences.
        """
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "audit_result": audit_result
        }
        self._append_jsonl(self.audits_log_path, payload)
        
        self.summary["metrics"]["total_audits_logged"] += 1

    def save_summary(self):
        """
        Finalizes tracking structurally dumping session accumulations inside an isolated `experiment_summary.json` file.
        """
        self.summary["datetime_end"] = datetime.utcnow().isoformat()
        
        target_path = self.base_dir / "experiment_summary.json"
        with open(target_path, "w") as f:
            json.dump(self.summary, f, indent=4, default=float)
            
        logger.info(f"Analytical pipeline closed successfully: Experiment summary persisted to {target_path.resolve()}")

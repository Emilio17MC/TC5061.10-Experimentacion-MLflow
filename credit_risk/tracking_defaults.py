"""Shared tracking defaults, safe to use before MLflow is imported."""

from pathlib import Path

from credit_risk import config as configuration

EXPERIMENT_NAME = "riesgo-impago-logistica"


def local_storage() -> Path:
    return configuration.PROJECT_ROOT / ".mlflow"


def default_tracking_uri() -> str:
    return f"sqlite:///{local_storage() / 'mlflow.db'}"

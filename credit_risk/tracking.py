"""Persist one executable training run without changing the training API."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.entities import Run
from sklearn.pipeline import Pipeline
from sqlalchemy.engine import make_url

from credit_risk import TrainingConfig, TrainingResult, run_training
from credit_risk.provenance import MODEL_DEPENDENCIES, input_fingerprints, log_provenance
from credit_risk.tracking_defaults import EXPERIMENT_NAME, default_tracking_uri, local_storage


@dataclass(frozen=True)
class TrackedTraining:
    result: TrainingResult
    run_id: str
    model_uri: str


@contextmanager
def _run_preserving_error(experiment_id: str) -> Iterator[Run]:
    # MLflow finalization can itself fail when the backend becomes unavailable.
    primary_error: Exception | None = None
    try:
        with mlflow.start_run(experiment_id=experiment_id) as run:
            try:
                yield run
            except Exception as error:
                primary_error = error
                raise
    except Exception as error:
        if primary_error is not None and error is not primary_error:
            primary_error.add_note(f"También falló el cierre de la corrida MLflow: {error!r}")
            raise primary_error
        raise


def _log_portable_pipeline(pipeline: Pipeline) -> str:
    # Invented values, never a row sampled from the source or holdout.
    example = pd.DataFrame([{
        "funded_amnt": 5000.0, "interest_rate": 12.0, "annual_income": 60000.0,
        "dept_paym_income_ratio": 15.0, "num_30+_delinq_in_2yrs": 0.0,
        "num_inq_in_6mths": 1.0, "num_open_credit_lines": 8.0,
        "num_derogatory_pub_rec": 0.0, "total_credit_revolving_bal": 2500.0,
        "used_credit_share": 30.0, "tot_num_credit_lines": 15.0,
        "loan_term_months": 36.0, "emp_length": "5 years", "grade": "B2",
        "home_ownership_status": "RENT", "verification_status": "Verified",
        "loan_purpose": "debt_consolidation",
    }])
    signature = mlflow.models.infer_signature(example, pipeline.predict(example))
    with TemporaryDirectory(prefix="credit-risk-code-") as temporary:
        source = Path(__file__).resolve().parent
        package = Path(temporary) / "credit_risk"
        # Only Python source: no datasets, credentials or interpreter caches.
        for module in source.rglob("*.py"):
            destination = package / module.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            copyfile(module, destination)
        model = mlflow.sklearn.log_model(
            pipeline, name="logistic_pipeline", serialization_format="cloudpickle",
            code_paths=[str(package)], input_example=example, signature=signature,
            pip_requirements=[
                f"{dependency}=={version(dependency)}"
                for dependency in MODEL_DEPENDENCIES
            ],
        )
    return model.model_uri


def track_training(
    config: TrainingConfig, *,
    experiment_name: str | None = None, tracking_uri: str | None = None,
) -> TrackedTraining:
    """Run once and persist mandatory evidence; propagate incomplete-run errors."""
    artifact_location = None
    if tracking_uri is None:
        storage = local_storage()
        storage.mkdir(parents=True, exist_ok=True)
        tracking_uri = default_tracking_uri()
        artifact_location = (storage / "artifacts").as_uri()
    elif tracking_uri.startswith("sqlite:"):
        database = make_url(tracking_uri).database
        if database:
            store = Path(database).resolve()
            artifact_location = (store.parent / f"{store.stem}-artifacts").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    name = experiment_name if experiment_name is not None else EXPERIMENT_NAME
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        name, artifact_location=artifact_location,
    )

    with _run_preserving_error(experiment_id) as run:
        for name, value in {
            "C": config.regularization, "class_weight": config.class_weight or "none",
            "max_iter": config.max_iterations, "seed": config.seed, "solver": "lbfgs",
            "sample_cap": config.sample_cap, "test_fraction": config.test_fraction,
            "threshold": config.threshold, "dummy_strategy": "prior",
        }.items():
            mlflow.log_param(name, value)

        fingerprints = input_fingerprints(config)
        result = run_training(config)
        log_provenance(result, fingerprints)
        metrics = {
            f"test_{model}_{name}": value
            for model, evaluation in [
                ("logistic", result.primary_evaluation), ("dummy", result.dummy_evaluation),
            ]
            for name, value in evaluation.metrics.items()
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_artifact(str(result.figure_path), artifact_path="evaluation")
        with TemporaryDirectory(prefix="credit-risk-metrics-") as temporary:
            metrics_path = Path(temporary) / "metrics.json"
            metrics_path.write_text(
                json.dumps({"metrics": metrics, "threshold": config.threshold}, allow_nan=False),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        model_uri = _log_portable_pipeline(result.model)
    return TrackedTraining(result=result, run_id=run.info.run_id, model_uri=model_uri)

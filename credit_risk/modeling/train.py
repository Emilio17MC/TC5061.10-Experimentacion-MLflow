"""One training operation hiding loading, splitting, fitting and evaluation."""

from dataclasses import dataclass
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from credit_risk.config import TrainingConfig
from credit_risk.dataset import DataQualitySummary, load_historical_loan_sample
from credit_risk.features import build_preprocessor, clean_raw_features
from credit_risk.modeling.evaluate import Evaluation, evaluate
from credit_risk.plots import save_evaluation_figure


@dataclass(frozen=True)
class TrainingResult:
    model: Pipeline
    primary_evaluation: Evaluation
    dummy_evaluation: Evaluation
    seed: int
    training_rows: int
    test_rows: int
    data_quality: DataQualitySummary
    figure_path: Path


def run_training(config: TrainingConfig) -> TrainingResult:
    """Train the initial logistic tracer on valid historical-loan records."""
    sample = load_historical_loan_sample(config)
    split_error = (
        "No es posible la división estratificada con ambos estados de pago en "
        "entrenamiento y prueba. Revise los conteos tras la auditoría, "
        "aumente sample_cap o ajuste test_fraction."
    )
    try:
        train_features, test_features, train_status, test_status = train_test_split(
            sample.features, sample.payment_status, test_size=config.test_fraction,
            stratify=sample.payment_status, random_state=config.seed,
        )
    except ValueError as error:
        raise ValueError(split_error) from error
    if train_status.nunique() != 2 or test_status.nunique() != 2:
        raise ValueError(split_error)
    model = make_pipeline(
        FunctionTransformer(clean_raw_features, validate=False),
        build_preprocessor(),
        LogisticRegression(
            C=config.regularization, class_weight=config.class_weight,
            max_iter=config.max_iterations, solver="lbfgs",
            random_state=config.seed,
        ),
    )
    model.fit(train_features, train_status)
    dummy = DummyClassifier(strategy="prior", random_state=config.seed)
    dummy.fit(train_features, train_status)
    primary_evaluation = evaluate(model, test_features, test_status, config.threshold)
    dummy_evaluation = evaluate(dummy, test_features, test_status, config.threshold)
    figure_path = save_evaluation_figure(
        test_status, primary_evaluation, dummy_evaluation, config.figure_path,
    )
    return TrainingResult(
        model=model,
        primary_evaluation=primary_evaluation,
        dummy_evaluation=dummy_evaluation,
        seed=config.seed,
        training_rows=len(train_features),
        test_rows=len(test_features),
        data_quality=sample.data_quality,
        figure_path=figure_path,
    )

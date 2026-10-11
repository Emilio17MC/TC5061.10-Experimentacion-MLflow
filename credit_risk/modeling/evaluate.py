"""Structured model evaluation without file writes or experiment tracking."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Evaluation:
    metrics: dict[str, float]
    probabilities: NDArray[np.float64]
    predictions: NDArray[np.int64]
    threshold: float

    @property
    def accuracy(self) -> float:
        return self.metrics["accuracy"]


def evaluate(
    model: Pipeline | DummyClassifier, features: pd.DataFrame, payment_status: pd.Series,
    threshold: float,
) -> Evaluation:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(np.int64)
    return Evaluation(
        metrics={
            "AP": float(average_precision_score(payment_status, probabilities)),
            "ROC_AUC": float(roc_auc_score(payment_status, probabilities)),
            "precision": float(precision_score(payment_status, predictions, zero_division=0)),
            "recall": float(recall_score(payment_status, predictions, zero_division=0)),
            "F1": float(f1_score(payment_status, predictions, zero_division=0)),
            "accuracy": float(accuracy_score(payment_status, predictions)),
        },
        probabilities=probabilities,
        predictions=predictions,
        threshold=threshold,
    )

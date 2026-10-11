"""Evaluation visualization from precomputed results, without model inference."""

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from credit_risk.modeling.evaluate import Evaluation


def save_evaluation_figure(
    payment_status: pd.Series, primary: Evaluation, dummy: Evaluation, destination: Path,
) -> Path:
    figure = Figure(figsize=(16, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    try:
        ConfusionMatrixDisplay.from_predictions(
            payment_status, primary.predictions, labels=[0, 1],
            display_labels=["Préstamo pagado", "Préstamo impagado"],
            ax=axes[0], colorbar=False, cmap="Blues", values_format="d",
        )
        axes[0].set(
            title=f"Matriz de confusión — umbral {primary.threshold:g}",
            xlabel="Estado de pago predicho", ylabel="Estado de pago observado",
        )
        for name, evaluation in [("Logística", primary), ("Dummy", dummy)]:
            RocCurveDisplay.from_predictions(
                payment_status, evaluation.probabilities, ax=axes[1], name=name,
            )
            PrecisionRecallDisplay.from_predictions(
                payment_status, evaluation.probabilities, ax=axes[2], name=name,
            )
        axes[1].set(
            title="Curva ROC", xlabel="Tasa de falsos positivos",
            ylabel="Tasa de verdaderos positivos",
        )
        axes[2].set(
            title="Curva precisión-recall", xlabel="Recall de préstamos impagados",
            ylabel="Precisión de préstamos impagados",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, format="png", dpi=150, bbox_inches="tight")
    finally:
        figure.clear()
    return destination

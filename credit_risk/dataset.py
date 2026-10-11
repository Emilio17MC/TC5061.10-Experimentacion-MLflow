"""Validated, reproducible historical-loan samples for training and exploration."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from credit_risk.config import TrainingConfig
from credit_risk.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass(frozen=True)
class DataQualitySummary:
    source_rows: int
    sampled_rows: int
    duplicate_rows_removed: int
    contradictory_rows_removed: int

    @property
    def retained_rows(self) -> int:
        return self.sampled_rows - self.duplicate_rows_removed - self.contradictory_rows_removed


@dataclass(frozen=True)
class HistoricalLoanSample:
    features: pd.DataFrame
    payment_status: pd.Series
    data_quality: DataQualitySummary


def load_historical_loan_sample(config: TrainingConfig) -> HistoricalLoanSample:
    """Load a validated stratified sample without fitting or transforming records."""
    features, payment_status = load_records(config)
    source_rows = len(features)
    if len(features) > config.sample_cap:
        # A stratified split requires both outcomes in the discarded partition.
        # Sampling directly also handles a one-row remainder or a rare outcome.
        if len(features) - config.sample_cap == 1 or payment_status.value_counts().min() < 2:
            indices = resample(
                features.index, replace=False, n_samples=config.sample_cap,
                stratify=payment_status, random_state=config.seed,
            )
        else:
            indices, _ = train_test_split(
                features.index, train_size=config.sample_cap,
                stratify=payment_status, random_state=config.seed,
            )
        indices = sorted(indices)
        features = features.loc[indices]
        payment_status = payment_status.loc[indices]
        if payment_status.nunique() != 2:
            raise ValueError("La muestra debe incluir ambos estados de pago; aumente sample_cap.")
    # Group by every original feature, retaining missing-valued groups too.
    groups = features.groupby(list(features.columns), dropna=False, sort=False).ngroup()
    contradictory = payment_status.groupby(groups).transform("nunique") > 1
    duplicates = features.duplicated()
    quality = DataQualitySummary(
        source_rows=source_rows, sampled_rows=len(features),
        duplicate_rows_removed=int((duplicates & ~contradictory).sum()),
        contradictory_rows_removed=int(contradictory.sum()),
    )
    keep = ~contradictory & ~duplicates
    return HistoricalLoanSample(
        features=features.loc[keep].copy(),
        payment_status=payment_status.loc[keep].copy(),
        data_quality=quality,
    )


def load_records(config: TrainingConfig) -> tuple[pd.DataFrame, pd.Series]:
    features = _read_csv(config.feature_path)
    missing = sorted(set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(features.columns))
    if missing:
        raise ValueError(f"Faltan columnas en {config.feature_path}: {', '.join(missing)}")
    status_frame = _read_csv(config.payment_status_path)
    if list(status_frame.columns) != ["y"]:
        raise ValueError(
            f"{config.payment_status_path} debe contener una única columna 'y'."
        )
    if len(features) != len(status_frame):
        raise ValueError(
            f"Cantidad de filas incompatible: features={len(features)}, "
            f"estado de pago={len(status_frame)}. Revise la alineación de los CSV."
        )
    payment_status = status_frame["y"]
    if not payment_status.isin([0, 1]).all():
        raise ValueError("El estado de pago solo admite 0 (pagado) y 1 (impagado), sin faltantes.")
    if payment_status.nunique() != 2:
        raise ValueError("Los datos deben contener ambos estados de pago (0 y 1).")
    return features, payment_status


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(f"CSV {path} vacío o inválido; revise encabezado y filas.") from error
    if frame.empty:
        raise ValueError(f"CSV {path} vacío; se requieren préstamos históricos.")
    return frame

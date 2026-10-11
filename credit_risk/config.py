"""Immutable inputs to a training run."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TrainingConfig:
    feature_path: Path = PROJECT_ROOT / "data" / "X.csv"
    payment_status_path: Path = PROJECT_ROOT / "data" / "target.csv"
    seed: int = 42
    sample_cap: int = 100_000
    test_fraction: float = 0.2
    regularization: float = 1.0
    max_iterations: int = 1000
    threshold: float = 0.5
    figure_path: Path = PROJECT_ROOT / "reports" / "figures" / "evaluation_baseline.png"
    class_weight: Literal["balanced"] | None = "balanced"

    def __post_init__(self) -> None:
        if self.class_weight is not None and (
            type(self.class_weight) is not str or self.class_weight != "balanced"
        ):
            raise ValueError("class_weight debe ser 'balanced' o None (sin ponderación).")
        if (type(self.regularization) not in (int, float)
                or not isfinite(self.regularization) or self.regularization <= 0):
            raise ValueError("C (regularization) debe ser un número finito mayor que cero.")
        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            raise ValueError("max_iter (max_iterations) debe ser un entero mayor que cero.")
        if type(self.seed) is not int or not 0 <= self.seed <= 4294967295:
            raise ValueError("seed debe ser un entero entre 0 y 4294967295, incluidos.")
        if type(self.sample_cap) is not int or self.sample_cap < 2:
            raise ValueError("sample_cap debe ser un entero mayor o igual a 2.")
        for source in [self.feature_path, self.payment_status_path]:
            same_file = (
                self.figure_path.exists() and source.exists()
                and self.figure_path.samefile(source)
            )
            if self.figure_path.resolve() == source.resolve() or same_file:
                raise ValueError("La figura no puede sobrescribir un CSV de entrada; elija otra ruta.")

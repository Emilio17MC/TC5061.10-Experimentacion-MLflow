"""Thin executable adapter for the training interface."""

import argparse
from dataclasses import replace
from pathlib import Path

from credit_risk import TrainingConfig
from credit_risk.tracking_defaults import EXPERIMENT_NAME, default_tracking_uri


def main(
    config: TrainingConfig | None = None, *,
    experiment_name: str | None = None, tracking_uri: str | None = None,
) -> None:
    # Delay MLflow initialization until after help and argument validation.
    from credit_risk.tracking import track_training

    tracked = track_training(
        config if config is not None else TrainingConfig(),
        experiment_name=experiment_name, tracking_uri=tracking_uri,
    )
    result = tracked.result
    print(f"Semilla: {result.seed}")
    quality = result.data_quality
    print(f"Registros fuente: {quality.source_rows}")
    print(f"Muestra antes de auditoría: {quality.sampled_rows}")
    print(f"Copias exactas eliminadas: {quality.duplicate_rows_removed}")
    print(f"Registros contradictorios eliminados: {quality.contradictory_rows_removed}")
    print(f"Muestra retenida: {quality.retained_rows}")
    print(f"Entrenamiento: {result.training_rows}")
    print(f"Prueba: {result.test_rows}")
    print(f"{'Modelo':<12}" + " ".join(f"{name:>10}" for name in result.primary_evaluation.metrics))
    for name, evaluation in [
        ("Logística", result.primary_evaluation), ("Dummy", result.dummy_evaluation),
    ]:
        print(f"{name:<12}" + " ".join(f"{value:10.4f}" for value in evaluation.metrics.values()))
    print(f"Evidencia: {result.figure_path}")
    print(f"Corrida MLflow: {tracked.run_id}")
    print(f"Modelo MLflow: {tracked.model_uri}")


def _cli() -> None:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        description="Entrenar y evaluar el riesgo de impago de préstamos históricos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--C", dest="regularization", type=float, default=defaults.regularization,
        help="Inverso de la intensidad de regularización; finito y mayor que cero.",
    )
    parser.add_argument(
        "--max-iter", dest="max_iterations", type=int, default=defaults.max_iterations,
        help="Límite de iteraciones; entero mayor que cero (no garantiza convergencia).",
    )
    parser.add_argument(
        "--seed", type=int, default=defaults.seed,
        help="Semilla única para muestreo, división y modelos; entero de 0 a 4294967295.",
    )
    parser.add_argument(
        "--class-weight", choices=["balanced", "none"], default=defaults.class_weight,
        help="Ponderación de los estados de pago; 'none' desactiva la ponderación.",
    )
    parser.add_argument(
        "--feature-path", type=Path, default=defaults.feature_path,
        help="CSV de características de préstamos históricos.",
    )
    parser.add_argument(
        "--payment-status-path", type=Path, default=defaults.payment_status_path,
        help="CSV con el estado de pago observado, alineado con las características.",
    )
    parser.add_argument(
        "--figure-path", type=Path, default=defaults.figure_path,
        help="Destino de la figura; no puede sobrescribir un CSV de entrada.",
    )
    parser.add_argument(
        "--experiment-name",
        help=f"Experimento MLflow; si se omite: {EXPERIMENT_NAME}.",
    )
    parser.add_argument(
        "--tracking-uri",
        help=f"Backend MLflow; si se omite: {default_tracking_uri()}.",
    )
    arguments = parser.parse_args()
    for option in ["experiment_name", "tracking_uri"]:
        value = getattr(arguments, option)
        if value is not None and not value.strip():
            parser.error(f"--{option.replace('_', '-')} requiere un valor no vacío")
    experiment_name = arguments.experiment_name
    tracking_uri = arguments.tracking_uri
    del arguments.experiment_name, arguments.tracking_uri
    if arguments.class_weight == "none":
        arguments.class_weight = None
    try:
        config = replace(defaults, **vars(arguments))
    except ValueError as error:
        parser.error(str(error))
    main(config, experiment_name=experiment_name, tracking_uri=tracking_uri)


if __name__ == "__main__":
    _cli()

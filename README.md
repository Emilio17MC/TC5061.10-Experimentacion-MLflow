# Entrenamiento de riesgo de impago

Baseline reproducible que compara una regresión logística con un modelo Dummy
sobre préstamos históricos. Usa el notebook para explorar los datos y `train.py`
para entrenar y evaluar. El resultado es experimental: no es una decisión de
crédito ni demuestra desempeño en producción.

## Inicio rápido

Requisitos: Python 3.13 o posterior.

### Prerrequisito: dataset de Kaggle

Antes de entrenar, obtén los datos del conjunto [Lending Club Loan Data Cleared](https://www.kaggle.com/datasets/db0boy/lending-club-loan-data-cleared/).
Coloca los archivos de entrada como `data/X.csv` y `data/target.csv`. En este
proyecto suman aproximadamente 434 MB y no se incluyen en Git, deben descargarse
por separado.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install --no-deps -e .
make train
```

`make train` ejecuta el baseline predeterminado. Para consultar las opciones de
entrenamiento, ejecuta `python train.py --help`.

## Comandos habituales

| Comando | Acción |
| --- | --- |
| `make train` | Entrena, evalúa y registra una corrida |
| `make test` | Ejecuta las pruebas; usa datos temporales, no requiere los CSV reales |
| `make typecheck` | Verifica los tipos con mypy |

Ejemplo de configuración desde la terminal:

```bash
make train ARGS='--C 0.1 --class-weight none --max-iter 2000 --seed 17'
```

## Resultados y MLflow

El entrenamiento guarda la figura comparativa en
`reports/figures/evaluation_baseline.png` y registra cada corrida en MLflow, con
base local en `.mlflow/mlflow.db`. Para abrir la interfaz, desde la raíz del
proyecto y con el entorno activado:

```bash
mlflow server --backend-store-uri "sqlite:///$(pwd)/.mlflow/mlflow.db" \
  --host 127.0.0.1 --port 5000
```

Después visita <http://127.0.0.1:5000>. No se suben CSV ni registros individuales
a MLflow. Carga únicamente modelos de confianza.

## Explorar y consultar evidencia

- [Notebook de exploración](notebooks/1.0-eaml-historical-loan-exploration.ipynb)
- [Validación del baseline con datos reales](docs/baseline-validation.md)
- [Recorrido de MLflow validado con datos reales](docs/mlflow-validation.md)

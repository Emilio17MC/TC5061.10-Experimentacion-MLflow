# MLflow: recorrido integrado verificado con datos reales

**Resultado:** una corrida real terminó en `FINISHED`; sus parámetros, doce
métricas, procedencia, evidencia y modelo se recuperaron mediante MLflow público.
El mismo modelo funcionó fuera del repositorio sobre una entrada sintética.
El usuario confirmó visualmente todos los elementos en la interfaz local.

Ejecución real del **8 de octubre de 2026**; regresión y cierre el **9 de octubre**,
en macOS arm64, Python 3.13.11 y MLflow 3.17.0.
Es una configuración, no una comparación de cinco entrenamientos ni una
evaluación de utilidad en producción. Los CSV y el almacenamiento quedan locales.

## Repetir el entrenamiento

Instalar según [Entorno y pruebas](../README.md#entorno-y-pruebas). Se usó el
entorno existente: `uv pip check --python .venv/bin/python` verificó **123 paquetes
compatibles**. No se afirma una nueva instalación limpia en esta validación.

Desde la raíz del proyecto, con `data/X.csv` y `data/target.csv` disponibles:

```bash
mkdir -p .cache/mlflow-validation
shasum -a 256 data/X.csv data/target.csv
.venv/bin/python train.py --experiment-name validacion-integrada-real \
  --tracking-uri "sqlite:///$(pwd)/.mlflow/ticket05.db" \
  --figure-path .cache/mlflow-validation/evaluation.png
shasum -a 256 data/X.csv data/target.csv
```

La ejecución terminó con código **0**. Los SHA-256 antes y después fueron iguales:

| Entrada | SHA-256 |
| --- | --- |
| `data/X.csv` | `516e88bc4cce5cab7b137c9e6541f5a088d9dc5e19f3eabd2e13fa312734d22c` |
| `data/target.csv` | `f7ad3252f7856b57f0814f635854c3921afddf29c9505ec0300fb0d841f8d229` |

## Identidad y configuración efectiva

| Elemento | Valor observado |
| --- | --- |
| Backend | `.mlflow/ticket05.db`, SQLite absoluto desde la raíz |
| Artefactos | `.mlflow/ticket05-artifacts/`, ubicación absoluta registrada |
| Experimento | `validacion-integrada-real`, ID `1` en esta base |
| Corrida | `48eb0ac36037411fb7c26fd928b9c354` |
| Modelo | `models:/m-56511e6c93a7419ca7a32e7ab4601418` |
| Parámetros | `C=1.0`, `class_weight=balanced`, `max_iter=1000`, `seed=42`, `solver=lbfgs` |
| Muestreo/evaluación | `sample_cap=100000`, `test_fraction=0.2`, `threshold=0.5`, `dummy_strategy=prior` |
| Git registrado | `a520603184be76f7a5f4329af23dfb082119bee8`, `status=available`, `dirty=true` |

El árbol estaba modificado durante la validación; el commit no identifica por sí
solo todo ese estado. El informe y la prueba añadida se versionan posteriormente.
Repetir el comando crea IDs nuevos: sustituir los de este informe al consultar.
Los IDs no reconstruyen una base que no se distribuye con Git.

## Resultados persistidos

Los siguientes valores coinciden exactamente con
[el baseline previamente reproducido](baseline-validation.md). Se contrastaron
las métricas de la corrida con `evaluation/metrics.json` y la tabla de consola
(esta última redondeada a cuatro decimales):

| Sufijo de métrica | `test_logistic_…` | `test_dummy_…` |
| --- | ---: | ---: |
| `AP` | 0.26453428060771705 | 0.13075 |
| `ROC_AUC` | 0.7141822993052955 | 0.5 |
| `precision` | 0.2170666013712047 | 0.0 |
| `recall` | 0.6780114722753346 | 0.0 |
| `F1` | 0.3288509691180562 | 0.0 |
| `accuracy` | 0.63815 | 0.86925 |

La consola muestra accuracy logística `0.6381`; no es un valor persistido
diferente. La diferencia de redondeo frente al notebook está explicada en el
informe del baseline. No se alteró entrenamiento para acercarse a la referencia.

Los conteos de tags y `provenance/metadata.json` coincidieron con la consola:
**2,139,643** registros fuente; **100,000** muestreados; **0** copias exactas y
**0** registros contradictorios eliminados; **100,000** retenidos;
**80,000** de entrenamiento y **20,000** de prueba. La auditoría se limita a la muestra.

Versiones comprobadas contra el entorno y los tags `software.*`:

| Software | Versión |
| --- | --- |
| Python | 3.13.11 |
| MLflow | 3.17.0 |
| scikit-learn | 1.9.1 |
| NumPy | 2.5.3 |
| pandas | 3.0.6 |
| cloudpickle | 3.1.2 |
| Matplotlib | 3.11.2 |
| SciPy | 1.18.1 |
| joblib | 1.6.0 |
| threadpoolctl | 3.7.0 |

Se descargaron y comprobaron:

- `evaluation/evaluation.png`: **117,022 bytes**, idénticos a la figura local.
- `evaluation/metrics.json`: las doce métricas y umbral `0.5`.
- `provenance/metadata.json`: huellas, siete conteos, versiones y Git,
  coherentes con los tags; no incluye filas individuales.
- Modelo vinculado: `get_logged_model()` confirma `source_run_id` de esta corrida;
  firma, ejemplo sintético de 17 características y seis dependencias fijadas
  presentes. No hay CSV en el paquete descargado.

## Consultar y recargar con MLflow público

Desde la raíz, usando únicamente modelos de confianza:

```python
import json
from pathlib import Path
import mlflow
import mlflow.sklearn

tracking_uri = f"sqlite:///{Path.cwd() / '.mlflow' / 'ticket05.db'}"
run_id = "48eb0ac36037411fb7c26fd928b9c354"
model_uri = "models:/m-56511e6c93a7419ca7a32e7ab4601418"
mlflow.set_tracking_uri(tracking_uri)
client = mlflow.MlflowClient(tracking_uri=tracking_uri)
run = client.get_run(run_id)
assert run.info.status == "FINISHED"
print(run.data.params, run.data.metrics)
for artifact in ["evaluation/metrics.json", "provenance/metadata.json"]:
    print(json.loads(Path(client.download_artifacts(run_id, artifact)).read_text()))
model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
description = mlflow.models.Model.load(model_path)
probe = description.load_input_example(model_path)  # Ejemplo inventado, no fila fuente.
model = mlflow.sklearn.load_model(model_uri)
print(model.predict(probe), model.predict_proba(probe))
```

### Portabilidad realmente comprobada

Además de esa recarga, el harness manual lanzó un proceso desde un directorio
temporal ajeno con `python -I -S`, sin directorio actual, `PYTHONPATH` ni startup
`.pth`. Añadió solo el directorio de dependencias (`sysconfig.get_path("purelib")`)
y confirmó que `credit_risk` no era importable antes de cargar el modelo.
Después verificó que `credit_risk.features` provenía de `code/` del artefacto,
no del checkout ni de la instalación editable. No se usaron registros reales
como entradas de esta prueba de inferencia.

Para el ejemplo sintético registrado: estado de pago predicho **0**,
probabilidades **[0.693040228841975, 0.30695977115802503]**. Coincidieron con
la recarga del padre: decisiones exactas; probabilidades con `rtol=atol=1e-12`.
Esto verifica portabilidad en el entorno actual, no igualdad entre plataformas.
La prueba `test_registered_pipeline_loads_without_the_checkout_or_editable_package`
reproduce automáticamente el aislamiento con datos sintéticos, tanto para
destino predeterminado como para experimento/backend seleccionados.

## Interfaz: verificación HTTP y confirmación visual

Desde la raíz:

```bash
.venv/bin/mlflow server \
  --backend-store-uri "sqlite:///$(pwd)/.mlflow/ticket05.db" \
  --host 127.0.0.1 --port 5055 --workers 1 --no-serve-artifacts
```

Abrir <http://127.0.0.1:5055>, seleccionar **validacion-integrada-real** y la
corrida anterior. En su detalle, consultar parámetros, tabla de métricas,
artefactos `evaluation` y `provenance`, y modelo vinculado. La figura y los JSON
son descargables; el modelo también se recupera con la URI indicada.
El servidor debe conservar acceso a las rutas absolutas de artefactos locales.
El uso del mismo backend sigue la
[documentación del servidor MLflow](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).

**Verificado por el agente:** servidor iniciado sobre esa base; raíz HTTP con
respuesta 200; URL de la corrida abierta en el navegador; artefactos descargados
por SDK. **Verificado visualmente por el usuario:** parámetros, doce métricas,
figura, ambos JSON y modelo vinculado; respondió «Sí, veo todos esos elementos».
El agente no dispone de control visual del navegador: no se atribuye esa
confirmación a una navegación automatizada ni se publican capturas inexistentes.

El sandbox denegó inicialmente el bind del puerto; se repitió con autorización.
La disponibilidad HTTP se comprobó directamente, no a partir del código de
salida del primer intento. No se aprovisionó infraestructura remota.

## Pruebas y límites de la evidencia

La prueba enfocada de portabilidad integrada pasó: **2 passed, 169 deselected**,
25.34 s. `.venv/bin/python -m mypy` pasó sobre **12 archivos**.
`.venv/bin/python -m pytest -q` terminó con **171 passed, 1 warning**, 208.77 s:
regresión de entrenamiento/datos, CLI Python/Make, persistencia, procedencia,
portabilidad y contratos de fallo; la advertencia es la deprecación
SQLAlchemy/MLflow ya conocida. Las pruebas usan datos y almacenes temporales:
no necesitan distribuir los CSV reales. El snippet Python de consulta/recarga
de este informe también se ejecutó con éxito sobre la corrida real.

Se conservaron avisos del entorno: caché Matplotlib fuera del sandbox (usó
temporal), advertencia de seguridad `cloudpickle`, `pip` ausente en el entorno
uv (no quedó fijado en `conda.yaml`), y deprecación Starlette al iniciar el
servidor. El modelo sí conserva las seis versiones explícitas en `requirements.txt`;
no se promete un lockfile transitorio completo ni se silencian las advertencias.

Solo se versionan este informe, la guía y la prueba sintética. La base,
artefactos, logs y harness manual quedan en `.mlflow/` y `.cache/`, excluidos
de Git; las entradas `data/*.csv` tampoco se versionan. La validación no incluye
cinco configuraciones comparativas, selección académica, PDF, Canvas, Registry
ni despliegue. Verificar datos faltantes o backend inaccesible no equivale a
afirmar una corrida exitosa o una transición `FAILED` que no pudo persistirse.

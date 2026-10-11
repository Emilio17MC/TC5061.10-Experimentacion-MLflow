# Baseline reproducido con datos reales

**Resultado:** la migración reproduce las seis métricas del notebook original,
y dos corridas predeterminadas devuelven métricas, probabilidades y decisiones
idénticas en el entorno validado. No se detectaron desviaciones materiales.

Validación realizada el **6 de octubre de 2026**, en macOS arm64 con Python
3.13.11 y las versiones directas de `requirements.txt` y `requirements-dev.txt`.
No implica igualdad bit a bit en otras plataformas o futuras resoluciones de
dependencias transitivas.

## Datos y configuración

Los CSV son entradas locales, alineadas por posición y no versionadas.
Estos identificadores permiten comprobar que se usan los mismos archivos:

| Archivo | SHA-256 |
| --- | --- |
| `data/X.csv` | `516e88bc4cce5cab7b137c9e6541f5a088d9dc5e19f3eabd2e13fa312734d22c` |
| `data/target.csv` | `f7ad3252f7856b57f0814f635854c3921afddf29c9505ec0300fb0d841f8d229` |

- Fuente: **2,139,643 préstamos históricos**.
- Semilla 42, muestra estratificada de 100,000; 0 copias exactas y 0 registros
  contradictorios excluidos. Auditoría limitada a la muestra.
- Entrenamiento: 80,000; prueba: 20,000; ambos estados de pago presentes.
- Logística: `C=1`, pesos balanceados, `lbfgs`, máximo 1,000 iteraciones;
  umbral 0.5. Dummy: probabilidades previas del entrenamiento.
- Imputación, escalamiento y codificación se ajustan solo con entrenamiento.
  El subgrado crediticio incorpora la evaluación previa del proveedor.

Los SHA-256 de ambos CSV se comprobaron antes y después de las dos corridas:
no cambiaron. No se serializaron modelos, métricas JSON ni experimentos MLflow.

## Comparación con el notebook original

Referencia: tabla de la celda 58 del notebook raíz conservado en el commit
`3ef9f09`. Esa tabla mostraba las métricas mediante `DataFrame.round(4)`.

| Métrica | Logística original | Logística migrada (valor) | Dummy migrado (valor) |
| --- | ---: | ---: | ---: |
| AP | 0.2645 | 0.26453428060771705 | 0.13075 |
| ROC-AUC | 0.7142 | 0.7141822993052955 | 0.5 |
| Precisión | 0.2171 | 0.2170666013712047 | 0.0 |
| Recall | 0.6780 | 0.6780114722753346 | 0.0 |
| F1 | 0.3289 | 0.3288509691180562 | 0.0 |
| Accuracy | 0.6382 | 0.63815 | 0.86925 |

El Dummy original mostraba AP 0.1308, ROC-AUC 0.5000, precisión/recall/F1
0.0000 y accuracy 0.8692, iguales al redondeo de los valores migrados.

### Diferencia de presentación investigada

La consola usa `f'{value:.4f}'` y presenta accuracy logística **0.6381**.
El notebook usaba pandas y presentaba **0.6382**. Sobre el mismo valor
`0.63815` se verificó explícitamente:

```python
import pandas as pd
value = 0.63815
print(f'{value:.4f}')                 # 0.6381
print(pd.Series([value]).round(4)[0]) # 0.6382
```

La representación binaria y las distintas operaciones de redondeo explican
la discrepancia de una unidad en el cuarto decimal: no prueba un cambio en
predicciones. Todas las métricas migradas reproducen las cifras originales
cuando se aplica el mismo redondeo de pandas. No fue necesario alterar modelo,
umbral ni preprocesamiento para acercar resultados a la referencia.

## Cómo repetir la comprobación

Instalar el entorno según el README. La instalación limpia se verificó con
`uv venv`, `uv pip install -r requirements.txt -r requirements-dev.txt` y
`uv pip install --no-deps -e .`; `uv pip check` confirmó 53 paquetes compatibles,
incluido el proyecto. No se instaló MLflow.

Se ejecutaron tanto el script Python por ruta absoluta como `make -f` desde un
directorio temporal ajeno al proyecto. Make se verificó también usando el
intérprete del entorno limpio mediante `PYTHON=/ruta/absoluta/al/python`.
Ambos producen la tabla anterior y `reports/figures/evaluation_baseline.png`.
La figura real contiene matriz de confusión, ROC y precisión-recall; se verificó
que no está vacía (117,022 bytes en esta ejecución). Es regenerable y no se versiona.

Para verificar repetibilidad con la interfaz pública, sin escribir otro modelo
ni un archivo de métricas:

```python
import numpy as np
from credit_risk import TrainingConfig, run_training

config = TrainingConfig()
first = run_training(config)
second = run_training(config)
for a, b in [
    (first.primary_evaluation, second.primary_evaluation),
    (first.dummy_evaluation, second.dummy_evaluation),
]:
    assert a.metrics == b.metrics
    np.testing.assert_array_equal(a.probabilities, b.probabilities)
    np.testing.assert_array_equal(a.predictions, b.predictions)
print(first.primary_evaluation.metrics)
```

El harness manual comprobó además probabilidades del modelo para 128 registros
fuente mediante `model.predict_proba`, iguales entre corridas. La suite sintética
comprueba la misma propiedad con datos temporales, sin depender de estos CSV.
La igualdad verificada es entre corridas de la aplicación; no se conservaron
probabilidades originales del notebook para afirmar igualdad con ellas.

## Límites

La división es aleatoria, no temporal. Accuracy alta del Dummy no significa
que identifique préstamos impagados: su recall es cero. El baseline no demuestra
utilidad en producción, causalidad ni una evaluación crediticia independiente
del proveedor. La auditoría no cubre filas fuera de la muestra ni identifica
prestatarios repetidos. Argumentos CLI, seguimiento y registro de modelos con
MLflow siguen fuera del paso 1.

## Verificación automatizada

En el entorno limpio, desde fuera del proyecto:

- `make -f /ruta/al/proyecto/Makefile test PYTHON=/ruta/al/python`:
  **71 passed en 12.16 s**. El recipe ejecuta `python -m pytest -q`.
- El recipe `typecheck` ejecuta `python -m mypy`: sin errores en 9 archivos.
- Las pruebas cubren Python y Make desde otro directorio, repetibilidad de
  ambas evaluaciones e inferencia cruda, contratos de datos/calidad, generación
  de evidencia y ejecución secuencial del notebook sobre CSV sintéticos.

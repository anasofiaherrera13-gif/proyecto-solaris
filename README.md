# Proyecto SOLARIS — Predicción de riesgo de mora

Proyecto de ciencia de datos desarrollado para **FrescaMar S.A.S.** con el objetivo de estimar el riesgo de mora de distribuidores con crédito a partir de información histórica de cartera.

El repositorio organiza el flujo completo de trabajo: auditoría y exploración de datos, preparación reproducible, selección de algoritmos, entrenamiento del modelo final, evaluación y serialización del pipeline.

## 1. Descripción del problema

FrescaMar S.A.S. cuenta con información histórica de aproximadamente **15.000 distribuidores** con variables relacionadas con perfil del cliente, ingresos, saldo, cuotas, ciudad, nivel educativo y score externo.

El objetivo es construir un modelo que permita distinguir entre clientes:

- **Al día** → clase `0`
- **Moroso** → clase `1`

Los registros con estado **Cancelado** no se utilizan en la clasificación binaria final. Después de este filtro, el conjunto utilizado para modelado contiene **12.300 registros**.

El propósito del modelo es apoyar el análisis de riesgo de crédito y priorizar distribuidores que podrían requerir seguimiento adicional.

## 2. Solución propuesta

El proyecto implementa un flujo reproducible compuesto por tres notebooks principales:

1. `01_eda_auditoria.ipynb`: exploración, auditoría de calidad y análisis inicial.
2. `02_model_selection.ipynb`: limpieza determinística, generación del dataset procesado y comparación de algoritmos.
3. `03_modeling_pipeline.ipynb`: entrenamiento, optimización, evaluación y serialización del modelo final.

El modelo seleccionado es **Random Forest**, entrenado mediante un `Pipeline` de scikit-learn que integra el preprocesamiento y el modelo. La optimización se realiza con `GridSearchCV` y validación cruzada de 5 folds utilizando **ROC-AUC** como métrica principal.

La separación final de datos utiliza un esquema **80/20 estratificado** con `random_state=42`.

### Métricas del pipeline final

| Métrica | Resultado |
|---|---:|
| ROC-AUC CV | 0.9621 |
| ROC-AUC Train | 0.9980 |
| ROC-AUC Test | 0.9588 |
| Diferencia Train-Test | 0.0391 |
| Precision — Moroso | 0.9076 |
| Recall — Moroso | 0.6278 |
| F1 — Moroso | 0.7422 |
| Accuracy | 0.9362 |

La diferencia entre ROC-AUC de train y test se mantiene por debajo del umbral de alerta definido en el proyecto, por lo que el pipeline presenta un comportamiento estable en esta validación.

Sin embargo, el **recall de la clase Moroso (0.6278)** todavía se encuentra por debajo del objetivo de negocio definido previamente de `0.75`. Por esta razón, el modelo debe considerarse un prototipo validado y reproducible, pero todavía requiere ajustes antes de una eventual implementación en producción.

## 3. Instalación

### Requisitos

- Python 3.12
- Git

Clonar el repositorio:

```bash
git clone https://github.com/anasofiaherrera13-gif/proyecto-solaris.git
cd proyecto-solaris
```

Crear un entorno virtual.

En Windows PowerShell:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

Actualizar `pip` e instalar las dependencias:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Comprobar el entorno:

```powershell
python -m pip check
```

## 4. Uso

### Ejecutar el pipeline final

Desde la raíz del repositorio:

```powershell
python src\pipeline.py
```

La ejecución entrena y optimiza el modelo y genera los artefactos principales en:

```text
models/artifacts/
reports/
reports/figures/
```

El modelo serializado actual se guarda como:

```text
models/artifacts/solaris_rf_v1_20260916.joblib
```

También se generan:

```text
reports/metricas_s08.json
reports/figures/confusion_matrix_s08.png
```

### Ejecutar los notebooks

Abrir Jupyter desde la raíz del proyecto:

```powershell
jupyter lab
```

y ejecutar los notebooks en este orden:

```text
notebooks/01_eda_auditoria.ipynb
notebooks/02_model_selection.ipynb
notebooks/03_modeling_pipeline.ipynb
```

### Dataset crudo

El dataset utilizado en el proyecto es ficticio y se incluye en el repositorio para facilitar la reproducibilidad completa del flujo.

Se encuentra en:

```text
data/raw/SOLARIS_FrescaMar_cartera_credito.csv
```

El notebook 02 genera:

```text
data/processed/solaris_datos_limpios.csv
```

El notebook 03 y `src/pipeline.py` utilizan este archivo procesado como entrada.

## 5. Resultados y contexto

Random Forest fue seleccionado después de comparar distintas alternativas de clasificación. La elección consideró no solamente desempeño predictivo, sino también estabilidad, facilidad de integración con el pipeline y comportamiento durante la validación.

El resultado final alcanza un **ROC-AUC Test de 0.9588**, lo que muestra una buena capacidad de discriminación entre clientes al día y morosos dentro del conjunto de prueba utilizado.

La precisión para la clase Moroso es alta (`0.9076`), mientras que el recall (`0.6278`) evidencia que todavía existen clientes morosos que el modelo no identifica. Este punto representa la principal oportunidad de mejora del prototipo.

Posibles líneas de trabajo posteriores incluyen ajuste del umbral de clasificación, técnicas de balanceo, optimización orientada a recall y análisis adicional del costo de falsos negativos.

## 6. Estructura del repositorio

```text
proyecto-solaris/
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── solaris_datos_limpios.csv
│   └── external/
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_eda_auditoria.ipynb
│   ├── 02_model_selection.ipynb
│   └── 03_modeling_pipeline.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── features.py
│   └── pipeline.py
│
├── models/
│   └── artifacts/
│       └── solaris_rf_v1_20260916.joblib
│
├── reports/
│   ├── figures/
│   ├── M2_MatrizEvaluacion_Algoritmos.csv
│   ├── metricas_s08.json
│   └── Perfil_Final.html
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Reproducibilidad

El proyecto utiliza una semilla fija:

```text
random_state = 42
```

El flujo final puede reproducirse desde terminal mediante:

```powershell
python src\pipeline.py
```

La ejecución desde `03_modeling_pipeline.ipynb` utiliza la misma lógica implementada en `src/pipeline.py`, evitando duplicación de código entre el notebook y el script principal.

---

**Proyecto SOLARIS**  
Ana Sofía Herrera Sánchez

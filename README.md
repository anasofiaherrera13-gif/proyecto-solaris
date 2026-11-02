# Proyecto SOLARIS — Sistema de apoyo para priorización de riesgo de mora

**Cliente:** FrescaMar S.A.S.  
**Proyecto académico:** QUANTUM ANALYTICS — Universidad de La Sabana  
**Autora:** Ana Sofía Herrera Sánchez  
**Repositorio:** https://github.com/anasofiaherrera13-gif/proyecto-solaris

---

## 1. Descripción general

**SOLARIS** es un proyecto de ciencia de datos desarrollado para apoyar al área de **Crédito y Cartera de FrescaMar S.A.S.** en la priorización de distribuidores que requieren revisión antes de aprobar, mantener o modificar un cupo de crédito.

El problema de negocio no consiste únicamente en predecir mora. FrescaMar necesita una forma más consistente de decidir **dónde concentrar primero la revisión humana**, utilizando señales históricas de comportamiento de pago, situación financiera y score externo.

La solución final:

- estima la probabilidad de mora de cada distribuidor;
- permite priorizar casos con mayor riesgo y exposición económica;
- incorpora explicabilidad para apoyar la revisión;
- incluye un dashboard interactivo en Streamlit;
- documenta limitaciones técnicas, de equidad y de gobernanza;
- mantiene la decisión final en manos del analista.

> **Recomendación final:** DESPLEGAR CON CONDICIONES, únicamente como herramienta de apoyo a la decisión. SOLARIS no debe utilizarse como decisor autónomo para negar, restringir o retirar crédito.

---

## 2. Evolución del proyecto: Módulos 1 a 5

### Módulo 1 — Discovery y definición del problema

Se formalizó el problema de negocio, los stakeholders, las métricas de éxito, las restricciones y los riesgos iniciales.

Principales hallazgos:

- 15.000 registros originales.
- 11 variables disponibles.
- 0 identificadores duplicados.
- 7,61 % de valores faltantes en `score_externo`.
- 31 % de registros con `ingreso_mensual = 0` y saldo activo.
- 847 registros, equivalentes al 5,65 %, con edades fuera del rango plausible.
- inconsistencias de formato en `ciudad`.

Metas preliminares definidas:

- ROC-AUC ≥ 0,80.
- Recall de Moroso ≥ 0,75.
- Precisión de Moroso ≥ 0,60.
- F1 ≥ 0,65.
- adopción operativa del score ≥ 80 % durante los primeros tres meses del piloto.

### Módulo 2 — Selección de modelos y pipeline

Se evaluaron cuatro algoritmos:

- Regresión Logística.
- Random Forest.
- XGBoost.
- SVM RBF.

Aunque XGBoost obtuvo un AUC ligeramente superior en una comparación inicial, **Random Forest** fue seleccionado por ofrecer el mejor equilibrio entre desempeño, robustez, interpretabilidad, estabilidad y facilidad de implementación.

El pipeline integra:

- `ColumnTransformer`;
- imputación por mediana para variables numéricas;
- `StandardScaler`;
- imputación por moda para categóricas;
- `OneHotEncoder(handle_unknown="ignore")`;
- `RandomForestClassifier`;
- optimización con `GridSearchCV`;
- auditoría de data leakage;
- serialización completa del pipeline.

### Módulo 3 — Reproducibilidad y control de versiones

Se consolidó una estructura profesional del repositorio y se documentó la evolución mediante Git y GitHub.

Se validó que:

- el notebook final y `src/pipeline.py` producen los mismos resultados;
- las rutas del proyecto son relativas;
- se utiliza `random_state=42`;
- el pipeline puede ejecutarse desde terminal;
- las métricas y el modelo final quedan serializados;
- el historial de commits refleja el desarrollo progresivo del proyecto.

### Módulo 4 — Evaluación crítica, interpretabilidad y ética

Se profundizó la evaluación mediante:

- comparación contra baseline;
- matriz de confusión;
- validación cruzada;
- interpretabilidad global y local con SHAP;
- análisis de fairness;
- análisis de riesgos;
- evaluación social, económica y ética;
- definición de salvaguardas.

La recomendación resultante fue:

> **Desplegar con condiciones, únicamente como sistema de apoyo a la decisión.**

### Módulo 5 — Dashboard, UX y comunicación ejecutiva

Se construyó un dashboard final en **Streamlit** orientado a la toma de decisiones.

El Dashboard v2 incluye cinco vistas funcionales:

1. **Identificar riesgo**
2. **Comparar ciudades**
3. **Priorizar exposición**
4. **Revisar distribuidor**
5. **Evaluar modelo**

El diseño incorpora:

- jerarquía de decisión;
- test de los 5 segundos;
- regla de los 3 clics;
- filtros con propósito;
- etiquetas descriptivas;
- colores semánticos;
- contraste WCAG básico;
- mensajes de acción;
- revisión humana explícita.

---

## 3. Datos utilizados

### Dataset original

```text
data/raw/SOLARIS_FrescaMar_cartera_credito.csv
```

Volumen original:

```text
15.000 registros × 11 variables
```

Variables originales:

```text
id_cliente
edad
ingreso_mensual
nivel_educativo
ciudad
fecha_vinculacion
saldo_actual
cuotas_pagadas
cuotas_totales
estado_cuenta
score_externo
```

### Población modelada

El problema se formula como clasificación binaria:

```text
Al día  → 0
Moroso  → 1
```

Los registros con estado `Cancelado` se excluyen.

Población final utilizada para modelado:

```text
12.300 registros
```

---

## 4. Variables utilizadas por el modelo

### Variables numéricas

```text
edad
ingreso_mensual
saldo_actual
cuotas_pagadas
cuotas_totales
score_externo
```

### Variables categóricas

```text
nivel_educativo
ciudad
```

### Variables excluidas

**`estado_cuenta`**  
Es la fuente utilizada para construir el target. Incluirla produciría data leakage directo.

**`id_cliente`**  
Es un identificador único sin valor predictivo generalizable.

**`fecha_vinculacion`**  
Representa la fecha de incorporación del distribuidor, no una fecha de scoring ni la fecha del evento de mora. Por esta razón no se utiliza como predictor ni como base para una validación temporal definitiva.

---

## 5. Modelo final

Modelo seleccionado:

```text
RandomForestClassifier
```

Configuración final:

```text
n_estimators = 400
max_depth = 20
min_samples_split = 2
min_samples_leaf = 2
random_state = 42
```

Artefacto serializado:

```text
models/artifacts/solaris_rf_v1_20260916.joblib
```

El preprocesamiento y el clasificador están encapsulados en un único `sklearn.Pipeline`.

---

## 6. Estrategia de validación

La versión final utiliza una separación **80/20 estratificada** con:

```text
random_state = 42
stratify = y
shuffle = True
```

Esto conserva la distribución de la clase Moroso entre entrenamiento y prueba.

### Aclaración sobre la validación temporal

Durante el Módulo 1 se planteó inicialmente una validación temporal utilizando `fecha_vinculacion`.

Posteriormente se comprobó que esta variable no representa una fecha de scoring ni una fecha del evento de mora, por lo que no permite construir una validación out-of-time metodológicamente correcta.

Una futura versión productiva debe registrar una verdadera `fecha_scoring` o fecha del evento y construir un holdout temporal independiente.

---

## 7. Métricas finales validadas

Las cifras finales del Proyecto SOLARIS son:

| Métrica | Resultado |
|---|---:|
| ROC-AUC CV original | 0.9621 |
| ROC-AUC CV reproducida | 0.9615 ± 0.0070 |
| ROC-AUC Train | 0.9980 |
| ROC-AUC Test | 0.9588 |
| Brecha Train-Test | 0.0391 |
| Accuracy | 0.9362 |
| Precisión Moroso | 0.9076 |
| Recall Moroso | 0.6278 |
| F1 Moroso | 0.7422 |

### Lectura de negocio

En el conjunto de prueba hubo:

```text
360 morosos reales
226 identificados correctamente
134 no detectados
```

Esto significa que SOLARIS identifica aproximadamente **63 de cada 100 casos reales de mora**, pero todavía deja pasar aproximadamente **37 de cada 100**.

La meta de negocio definida es identificar al menos:

```text
75 de cada 100 morosos
```

Por esta razón, SOLARIS presenta suficiente capacidad para **priorizar revisiones**, pero todavía no debe tomar decisiones adversas de crédito de manera automática.

---

## 8. Comparación contra baseline

La Regresión Logística se utilizó como baseline interpretable.

| Modelo | ROC-AUC CV | ROC-AUC Test | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Regresión Logística | 0.9225 | 0.9241 | 0.8230 | 0.5167 | 0.6348 |
| Random Forest SOLARIS | 0.9615 | 0.9588 | 0.9076 | 0.6278 | 0.7422 |

Random Forest supera al baseline de forma consistente en las métricas principales.

---

## 9. Interpretabilidad

El análisis SHAP identifica como principales factores:

1. `saldo_actual`
2. `cuotas_pagadas`
3. `score_externo`
4. `cuotas_totales`

Interpretación de negocio:

- un saldo pendiente alto incrementa el riesgo estimado;
- un bajo avance en el pago de cuotas aumenta la probabilidad de mora;
- `score_externo` aporta señal adicional;
- el origen y composición de `score_externo` deben documentarse.

La explicación SHAP permite comprender una predicción, pero **no convierte una correlación en causalidad**.

---

## 10. Fairness y riesgos

Se analizaron diferencias por:

- rango de edad;
- nivel educativo;
- ciudad.

La principal señal encontrada aparece en ciudad.

```text
Medellín ≈ 0.7333
Cartagena ≈ 0.4800
Brecha ≈ 0.2533
```

La diferencia aproximada de 25,33 puntos porcentuales **no demuestra discriminación causal**, pero sí obliga a investigar y monitorear el comportamiento del sistema por ciudad.

Otros riesgos identificados:

- recall por debajo de la meta;
- opacidad de `score_externo`;
- validación temporal limitada;
- automation bias;
- posibles proxies geográficos o socioeconómicos;
- cambios futuros en la distribución de datos.

---

## 11. Salvaguardas de uso responsable

SOLARIS debe operar con las siguientes condiciones:

- revisión humana obligatoria para decisiones adversas;
- prohibición de rechazo automático basado únicamente en el score;
- explicación disponible para los casos prioritarios;
- monitoreo de desempeño y fairness;
- trazabilidad de `score_externo`;
- acceso a la información por roles;
- uso de los datos únicamente para el propósito crediticio declarado;
- mecanismo de corrección de datos erróneos;
- monitoreo de drift antes de ampliar el despliegue.

---

## 12. Dashboard SOLARIS

El Módulo 5 incluye un dashboard interactivo desarrollado en **Streamlit**.

Ruta:

```text
dashboard/
```

Archivo principal:

```text
dashboard/app.py
```

Datos de predicción:

```text
dashboard/data/solaris_predicciones.csv
```

### Vista 1 — Identificar riesgo

Responde:

> ¿Cuántos distribuidores requieren atención y cuáles deberían revisarse primero?

### Vista 2 — Comparar ciudades

Responde:

> ¿Dónde se concentra mayor riesgo?

### Vista 3 — Priorizar exposición

Responde:

> ¿Dónde se encuentra el mayor saldo expuesto entre los casos prioritarios?

### Vista 4 — Revisar distribuidor

Permite revisar un caso individual utilizando:

- probabilidad estimada;
- saldo;
- cuotas pagadas;
- score externo;
- perfil;
- acción sugerida.

### Vista 5 — Evaluar modelo

Comunica si SOLARIS puede apoyar la priorización y bajo qué condiciones.

### Filtros principales

- ciudad;
- umbral para definir prioridad.

---

## 13. Instalación desde cero

### Requisitos

```text
Python 3.12
Git
```

### Windows PowerShell

Clonar el repositorio:

```powershell
git clone https://github.com/anasofiaherrera13-gif/proyecto-solaris.git
cd proyecto-solaris
```

Crear y activar el entorno:

```powershell
py -3.12 -m venv venv_solaris
.\venv_solaris\Scripts\Activate.ps1
```

Instalar dependencias:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r dashboard\requirements.txt
```

Comprobar el entorno:

```powershell
python -m pip check
```

---

## 14. Ejecutar el pipeline final

Desde la raíz del repositorio:

```powershell
python src\pipeline.py
```

Los principales artefactos se encuentran en:

```text
models/artifacts/
reports/
reports/figures/
```

Entre ellos:

```text
models/artifacts/solaris_rf_v1_20260916.joblib
reports/metricas_s08.json
reports/figures/confusion_matrix_s08.png
```

---

## 15. Ejecutar los notebooks

Desde la raíz:

```powershell
jupyter lab
```

Orden recomendado:

```text
notebooks/01_eda_auditoria.ipynb
notebooks/02_model_selection.ipynb
notebooks/03_modeling_pipeline.ipynb
Modulo_4/M4_PCD2_HerreraSanchez_Validacion.ipynb
```

---

## 16. Ejecutar el dashboard

Con el entorno activado:

```powershell
cd dashboard
streamlit run app.py
```

La aplicación se abre localmente en:

```text
http://localhost:8501
```

---

## 17. Estructura general del repositorio

```text
proyecto-solaris/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
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
│
├── reports/
│   ├── figures/
│   └── metricas_s08.json
│
├── Modulo_4/
│   ├── M4_PCD2_HerreraSanchez_Validacion.ipynb
│   ├── M4_PCD2_HerreraSanchez_EvaluacionCritica.pdf
│   ├── M4_PCD2_HerreraSanchez_ImpactoEtico.pdf
│   ├── README_ENTREGA.md
│   ├── figures/
│   └── tables/
│
└── dashboard/
    ├── app.py
    ├── generar_predicciones.py
    ├── requirements.txt
    ├── data/
    │   └── solaris_predicciones.csv
    └── utils/
        └── graficos.py
```

La estructura puede contener archivos adicionales correspondientes a evidencias y entregables de cada módulo.

---

## 18. Entregables finales

### Informe ejecutivo final

```text
M5_PCD2_HerreraSanchez_InformeEjecutivo.pdf
```

Documento integrador del trabajo desarrollado durante los Módulos 1 a 5.

### Presentación ejecutiva final

```text
M5_PCD2_HerreraSanchez_PresentacionFinal.pdf
```

Incluye la narrativa ejecutiva y el apéndice técnico.

### Dashboard

```text
dashboard/
```

Prototipo interactivo funcional desarrollado en Streamlit.

### Repositorio GitHub

```text
https://github.com/anasofiaherrera13-gif/proyecto-solaris
```

---

## 19. Hoja de ruta recomendada

### Fase 1 — Piloto controlado

- integrar SOLARIS al flujo de Crédito y Cartera;
- mantener la decisión final en el analista;
- lograr que al menos 80 % de las nuevas decisiones de crédito consulten SOLARIS durante los primeros tres meses.

### Fase 2 — Mejora

- recalibrar el umbral;
- acercar la detección a la meta de 75 de cada 100 morosos;
- monitorear diferencias por ciudad;
- documentar la procedencia de `score_externo`;
- definir con negocio el costo relativo de los errores.

### Fase 3 — Validación para escala

- registrar una verdadera fecha de scoring;
- construir un holdout out-of-time;
- monitorear drift;
- establecer con Finanzas una línea base económica;
- comparar el proceso apoyado por SOLARIS contra el proceso actual;
- decidir si se amplía el uso.

---

## 20. Limitaciones

SOLARIS **no afirma haber demostrado todavía una reducción monetaria específica de la cartera vencida**.

La cuantificación económica final requiere:

- establecer una línea base con Finanzas;
- ejecutar un piloto;
- observar resultados reales;
- comparar el proceso con SOLARIS frente a la política actual.

La ausencia de una alerta tampoco significa que un distribuidor sea necesariamente seguro.

---

## 21. Reproducibilidad

El proyecto utiliza:

```text
random_state = 42
```

El pipeline principal puede ejecutarse desde terminal mediante:

```powershell
python src\pipeline.py
```

El notebook final y el script principal utilizan la misma lógica de modelado para reducir duplicación e inconsistencias.

El dataset académico es ficticio y puede mantenerse en el repositorio para facilitar la reproducción del flujo.

---

## 22. Recomendación final

**SOLARIS presenta capacidad suficiente para apoyar la priorización interna de Crédito y Cartera, pero no para automatizar decisiones adversas de crédito en su estado actual.**

> # DESPLEGAR CON CONDICIONES

Condiciones mínimas:

1. mantener revisión humana;
2. mejorar o justificar el recall;
3. monitorear fairness geográfica;
4. documentar `score_externo`;
5. construir una validación temporal real;
6. monitorear desempeño después del despliegue.

---

## 23. Autoría

**Ana Sofía Herrera Sánchez**  
Proyecto SOLARIS — FrescaMar S.A.S.  
QUANTUM ANALYTICS  
Universidad de La Sabana  
2026
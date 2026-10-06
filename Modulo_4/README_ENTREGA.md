# Entrega Módulo 4 - Proyecto SOLARIS

Archivos principales:

- `M4_PCD2_HerreraSanchez_Validacion.ipynb`: notebook maestro reproducible.
- `M4_PCD2_HerreraSanchez_EvaluacionCritica.pdf`: informe principal de evaluación crítica / Actividad 1.
- `M4_PCD2_HerreraSanchez_ImpactoEtico.pdf`: Actividad 2, impacto social, económico y ético.

## Ejecución

Desde la raíz del repositorio:

```bash
pip install -r requirements.txt
jupyter lab
```

Abrir `Modulo_4/M4_PCD2_HerreraSanchez_Validacion.ipynb` y ejecutar **Restart Kernel + Run All**. El notebook carga `data/processed/solaris_datos_limpios.csv` y `models/artifacts/solaris_rf_v1_20260916.joblib` mediante rutas relativas y genera tablas en `Modulo_4/tables/` y figuras en `Modulo_4/figures/`.

## Recomendación de despliegue

**Desplegar con condiciones, únicamente como apoyo a la decisión.** El modelo no debe actuar como decisor autónomo hasta mejorar/justificar recall, fairness geográfica, trazabilidad de `score_externo` y validación temporal.

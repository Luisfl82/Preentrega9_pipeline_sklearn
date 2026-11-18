# Preentrega 9 - Pipeline de preprocesamiento robusto (scikit-learn)

## Dataset
Clientes de una empresa de telecomunicaciones (dataset de ejemplo generado con `generar_dataset.py`, con valores nulos a propósito). Variable objetivo: `churn` (1 = abandonó). Variables numéricas: `antiguedad_meses`, `cargo_mensual`. Categóricas: `contrato`, `servicio_internet`.

## Qué hace `pipeline_ml.py`
1. Carga el CSV e identifica columnas numéricas y categóricas.
2. Hace `train_test_split` **antes** de ajustar el preprocesamiento (evita *data leakage*).
3. `ColumnTransformer`:
   - Numéricas: `SimpleImputer(median)` + `StandardScaler`.
   - Categóricas: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`.
4. `Pipeline` final: preprocesador + `LogisticRegression`.
5. Entrena, muestra `classification_report` y prueba `.predict()` con datos nuevos (incluso con categoría desconocida y nulos).
6. Exporta el pipeline entrenado con `joblib` (`modelo_pipeline.joblib`).

## Cómo ejecutarlo
1. `pip install -r requirements.txt`
2. `python generar_dataset.py`
3. `python pipeline_ml.py`

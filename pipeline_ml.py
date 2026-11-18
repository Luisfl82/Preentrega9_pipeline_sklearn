"""Preentrega 9 - Pipeline de preprocesamiento robusto con scikit-learn."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1) Carga y exploración
df = pd.read_csv("data/clientes.csv")
print(df.info())
TARGET = "churn"
X = df.drop(columns=[TARGET])
y = df[TARGET]
columnas_numericas = X.select_dtypes(include="number").columns.tolist()
columnas_categoricas = X.select_dtypes(exclude="number").columns.tolist()
print("Numéricas:", columnas_numericas)
print("Categóricas:", columnas_categoricas)

# 2) Split ANTES de ajustar cualquier preprocesamiento (evita data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Transformers por tipo de columna
pipe_numerico = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler()),
])
pipe_categorico = Pipeline([
    ("imputador", SimpleImputer(strategy="most_frequent")),
    ("codificador", OneHotEncoder(handle_unknown="ignore")),
])
preprocesador = ColumnTransformer([
    ("num", pipe_numerico, columnas_numericas),
    ("cat", pipe_categorico, columnas_categoricas),
])

# 4) Pipeline final: preprocesador + modelo
modelo = Pipeline([
    ("preprocesador", preprocesador),
    ("clasificador", LogisticRegression(max_iter=1000)),
])

# 5) Entrenamiento y reporte
modelo.fit(X_train, y_train)
print("\n=== Reporte de métricas (test) ===")
print(classification_report(y_test, modelo.predict(X_test)))

# 6) Prueba de .predict() con datos nuevos (incluye una categoría desconocida y un nulo)
nuevos = pd.DataFrame({
    "antiguedad_meses": [5, None],
    "cargo_mensual": [95.0, 40.0],
    "contrato": ["Mensual", "Trimestral"],   # "Trimestral" no existe en el entrenamiento
    "servicio_internet": ["Fibra", "DSL"],
})
print("Predicción para clientes nuevos:", modelo.predict(nuevos))

# 7) Persistencia con joblib
joblib.dump(modelo, "modelo_pipeline.joblib")
print("Pipeline guardado en modelo_pipeline.joblib")

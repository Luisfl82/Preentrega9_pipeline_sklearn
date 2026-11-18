"""Dataset de ejemplo tipo churn, con nulos y categorías, en data/clientes.csv"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(21)
n = 1000
tenure = rng.integers(1, 72, n).astype(float)
contrato = rng.choice(["Mensual", "Anual", "Bianual"], n, p=[0.55, 0.25, 0.2])
cargo = rng.normal(70, 25, n).clip(20, 130).round(2)
internet = rng.choice(["DSL", "Fibra", "Ninguno"], n, p=[0.35, 0.45, 0.2])
logit = -1.0 - 0.04 * tenure + 0.02 * cargo + (contrato == "Mensual") * 1.2
churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame({"antiguedad_meses": tenure, "cargo_mensual": cargo,
                   "contrato": contrato, "servicio_internet": internet, "churn": churn})
df.loc[rng.choice(n, 40, replace=False), "antiguedad_meses"] = np.nan
df.loc[rng.choice(n, 30, replace=False), "cargo_mensual"] = np.nan
df.loc[rng.choice(n, 25, replace=False), "contrato"] = None
df.to_csv("data/clientes.csv", index=False)
print("Creado data/clientes.csv")

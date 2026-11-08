#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carga un modelo registrado en MLflow y marca transacciones atípicas en 2023
Uso: python probar_modelo.py valor_esperado_ensamble 1
Equipo DSA
"""
import os
import sys
import numpy as np
import mlflow
import mlflow.pyfunc
from datos_pereira import cargar_datos, sigma_robusto, FEATURES

nombre = sys.argv[1] if len(sys.argv) > 1 else "valor_esperado_ensamble"
version = sys.argv[2] if len(sys.argv) > 2 else "1"

# defina el servidor donde están registrados los modelos
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://34.204.40.23:8050"))
modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{version}")

# Cargue los datos y calcule el valor esperado y el puntaje de atipicidad
train, val, test = cargar_datos("pereira.csv")
sigma = sigma_robusto(val["y"].values - modelo.predict(val[FEATURES]))
test = test.copy()
test["pred_log"] = modelo.predict(test[FEATURES])
test["valor_esperado"] = np.exp(test["pred_log"])
test["z"] = (test["y"] - test["pred_log"]) / sigma
test["atipica"] = test["z"].abs() > 3.5

print(test[["valor", "valor_esperado", "z", "atipica"]].head(10))
print("Marcadas como atípicas:", int(test["atipica"].sum()), "de", len(test))
test.sort_values("z", key=abs, ascending=False).head(50).to_csv("top_atipicas.csv", index=False)

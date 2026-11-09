#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paso previo al tablero: trae cada modelo registrado en MLflow, puntua las
compraventas de validacion (2022) y prueba (2023) y guarda todo en data/.

Requiere en la misma carpeta: datos_pereira.py (el del equipo) y pereira.csv.

Uso (PowerShell, en la carpeta del proyecto):
    $env:MLFLOW_TRACKING_URI = "http://IP_DEL_SERVIDOR:8050"
    python preparar_datos.py --csv pereira.csv
    python preparar_datos.py --csv pereira.csv --modelos rf xgb --version-ensamble 2
"""
import argparse
import json
import os

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from datos_pereira import FEATURES, cargar_datos, sigma_robusto

# clave -> (nombre para mostrar, nombre en el Model Registry de MLflow)
MODELOS = {
    "rf": ("Random Forest", "valor_esperado_random_forest"),
    "xgb": ("XGBoost", "valor_esperado_xgboost"),
    "ensamble": ("Ensamble (stacking)", "valor_esperado_ensamble"),
}
# metricas que los scripts de entrenamiento registran en cada run
METRICAS = ["rmse_log", "mae_log", "r2_log", "sigma_robusto",
            "roc_auc_sintetico", "auc_pr_sintetico", "tiempo_entrenamiento_s"]
COLUMNAS_SALIDA = ["matricula", "documento_justificativo", "year_radica", "mes_radica",
                   "tipo_predio_zona", "valor", "y", "rezago", "anot_folio", "doc_tipo",
                   "doc_entidad"]
DATA_DIR = os.environ.get("RADAR_DATA", "data")


def resolver_version(client, nombre, version):
    """'latest' -> la version mas alta registrada; si no, la version pedida."""
    if version and str(version).lower() != "latest":
        return str(version)
    versiones = client.search_model_versions(f"name='{nombre}'")
    if not versiones:
        raise RuntimeError(f"No hay versiones registradas de '{nombre}' en MLflow")
    return str(max(int(v.version) for v in versiones))


def metricas_del_modelo(client, nombre, version):
    """Metricas del run que genero esa version del modelo."""
    mv = client.get_model_version(nombre, version)
    run = client.get_run(mv.run_id)
    return {m: run.data.metrics.get(m) for m in METRICAS}


def puntuar(nombre, version, val, test):
    """Carga el modelo desde MLflow y calcula valor esperado y z (igual que probar_modelo.py)."""
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{version}")
    df = pd.concat([val, test], ignore_index=True)
    pred = np.asarray(modelo.predict(df[FEATURES]), dtype=float)
    sigma = float(sigma_robusto(val["y"].values - pred[:len(val)]))   # sigma sobre validacion
    out = df[[c for c in COLUMNAS_SALIDA if c in df.columns]].copy()
    out["pred_log"] = pred
    out["valor_esperado"] = np.exp(pred)
    out["z"] = (df["y"].values - pred) / sigma
    return out, sigma


def indice_valoracion(todo):
    """Mediana de valor por anio y tipo de predio, base 2015 = 100 (nominal)."""
    t = todo[todo["year_radica"] >= 2015].copy()
    t["grupo"] = np.where(t["tipo_predio_zona"].astype(str).str.upper().str.contains("RUR"),
                          "Rural", "Urbano")
    m = t.groupby(["year_radica", "grupo"])["valor"].median().reset_index(name="mediana")
    base = m[m["year_radica"] == 2015].set_index("grupo")["mediana"]
    m["indice"] = 100 * m["mediana"] / m["grupo"].map(base)
    return m.rename(columns={"year_radica": "anio"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="pereira.csv")
    ap.add_argument("--modelos", nargs="+", default=list(MODELOS), choices=list(MODELOS))
    for k in MODELOS:
        ap.add_argument(f"--version-{k}", default="latest")
    args = ap.parse_args()

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise SystemExit('Falta MLFLOW_TRACKING_URI. En PowerShell: $env:MLFLOW_TRACKING_URI = "http://IP:8050"')
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Cargando y preparando datos...")
    train, val, test = cargar_datos(args.csv)
    with open(args.csv, "rb") as f:
        n_anotaciones = sum(1 for _ in f) - 1
    todo = pd.concat([train, val, test], ignore_index=True)

    indice_valoracion(todo).to_parquet(os.path.join(DATA_DIR, "indice.parquet"), index=False)
    resumen = {"n_anotaciones": int(n_anotaciones), "n_unicas": int(len(todo)),
               "n_train": int(len(train)), "modelos": {}}

    for k in args.modelos:
        etiqueta, nombre = MODELOS[k]
        version = resolver_version(client, nombre, getattr(args, f"version_{k}"))
        print(f"Puntuando {etiqueta} ({nombre} v{version})...")
        scored, sigma = puntuar(nombre, version, val, test)
        scored.to_parquet(os.path.join(DATA_DIR, f"scored_{k}.parquet"), index=False)
        resumen["modelos"][k] = {"etiqueta": etiqueta, "registro": nombre, "version": version,
                                 "sigma": sigma, "metricas": metricas_del_modelo(client, nombre, version)}

    with open(os.path.join(DATA_DIR, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    print(f"Listo. Archivos en '{DATA_DIR}/'. Ahora: python app.py")


if __name__ == "__main__":
    main()

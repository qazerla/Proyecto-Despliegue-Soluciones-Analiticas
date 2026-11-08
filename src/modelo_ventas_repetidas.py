import os
import time
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

AUTOR = "Valeria"
RUTA_DATOS = "pereira.csv"
COD_COMPRAVENTA = 125
AÑO_CORTE = 2021

mlflow.set_tracking_uri(os.getenv("MLFLOW_URI", "http://34.227.194.203:8050"))
mlflow.set_experiment("experimento-dsa")


def leer_csv(ruta):
    with open(ruta, encoding="utf-8-sig") as f:
        primera = f.readline()
    sep = max(["\t", ";", ","], key=primera.count)
    df = pd.read_csv(ruta, sep=sep, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    return df


def extraer_documento(df):
    doc = df["documento_justificativo"].fillna("")
    df["doc_tipo"] = doc.str.extract(r"^\d*-?\s*([A-ZÁÉÍÓÚÑ]+)")[0].fillna("OTRO")
    fecha = doc.str.extract(r"(\d{1,2})/(\d{1,2})/(\d{2,4})")
    anio = pd.to_numeric(fecha[2], errors="coerce")
    anio = anio.where(anio >= 100, anio + 2000)
    df["rezago_doc"] = (df["year_radica"] - anio).clip(-5, 30)
    entidad = doc.str.split(";", n=1).str[1].fillna("").str.strip()
    top = entidad.value_counts().head(10).index
    df["doc_entidad"] = entidad.where(entidad.isin(top), "OTRA")
    return df


def construir_pares(ruta):
    df = leer_csv(ruta)
    df = df[df["cod_natujur"] == COD_COMPRAVENTA].copy()
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df[df["valor"] >= 1000]

    # Se descartan valores repetidos en varios folios (pertenecen al documento)
    llave = ["documento_justificativo", "valor"]
    df["veces_valor"] = df.groupby(llave)["valor"].transform("size")
    df = df[df["veces_valor"] == 1].copy()

    df = extraer_documento(df)
    df["tipo_predio_zona"] = df["tipo_predio_zona"].fillna("SIN INFORMACION")
    df["log_valor"] = np.log(df["valor"])

    df = df.sort_values(["matricula", "year_radica", "num_anotacion"])
    g = df.groupby("matricula")
    df["log_valor_prev"] = g["log_valor"].shift(1)
    df["year_prev"] = g["year_radica"].shift(1)
    df["ventas_folio"] = g["matricula"].transform("size")

    pares = df.dropna(subset=["log_valor_prev"]).copy()
    pares["años_entre"] = pares["year_radica"] - pares["year_prev"]
    tope = pares["valor"].quantile(0.999)
    return pares[pares["valor"] <= tope]


NUMERICAS = ["log_valor_prev", "años_entre", "rezago_doc", "ventas_folio"]
CATEGORICAS = ["tipo_predio_zona", "doc_tipo", "doc_entidad"]

pares = construir_pares(RUTA_DATOS)
train = pares[pares["year_radica"] <= AÑO_CORTE]
valid = pares[pares["year_radica"] > AÑO_CORTE]
print(f"Pares de ventas: {len(pares):,} (train {len(train):,} | valid {len(valid):,})")

mae_sinmod = mean_absolute_error(valid["log_valor"], valid["log_valor_prev"])
print(f"MAE sin modelo (precio no cambia): {mae_sinmod:.3f}")

preprocesador = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("num", "passthrough", NUMERICAS),
    ]
)

EXPERIMENTOS = [
    ("pares-ridge", Ridge, {"alpha": 1.0}),
    ("pares-rf-100-prof8", RandomForestRegressor, {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5}),
    ("pares-gb-200-prof3", GradientBoostingRegressor, {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1}),
]

for nombre, clase, params in EXPERIMENTOS:
    extra = {"random_state": 42} if clase is not Ridge else {}
    if clase is RandomForestRegressor:
        extra["n_jobs"] = -1
    modelo = Pipeline([("prep", preprocesador), ("reg", clase(**params, **extra))])

    with mlflow.start_run(run_name=f"{AUTOR}-{nombre}"):
        mlflow.set_tag("autor", AUTOR)
        mlflow.set_tag("enfoque", "ventas_repetidas")
        mlflow.set_tag("tipo_modelo", clase.__name__)

        inicio = time.time()
        modelo.fit(train[NUMERICAS + CATEGORICAS], train["log_valor"])
        tiempo_entrenamiento = time.time() - inicio

        pred = modelo.predict(valid[NUMERICAS + CATEGORICAS])

        mae = mean_absolute_error(valid["log_valor"], pred)
        rmse = mean_squared_error(valid["log_valor"], pred) ** 0.5
        r2 = r2_score(valid["log_valor"], pred)

        mlflow.log_params(params)
        mlflow.log_param("año_corte", AÑO_CORTE)
        mlflow.log_param("pares_train", len(train))
        mlflow.log_param("pares_valid", len(valid))
        mlflow.log_metric("mae_log", mae)
        mlflow.log_metric("rmse_log", rmse)
        mlflow.log_metric("r2_log", r2)
        mlflow.log_metric("tiempo_entrenamiento_s", tiempo_entrenamiento)
        mlflow.sklearn.log_model(
            modelo, name="modelo", skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(f"{nombre} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f} | {tiempo_entrenamiento:.1f} s")
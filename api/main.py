import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from datos_pereira import FEATURES, parse_doc, sigma_robusto

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://34.227.194.203:8050")
DATA_DIR = os.environ.get("RADAR_DATA", "/app/radar_data")

MODELOS_REGISTRO = {
    "rf": ("Random Forest", "valor_esperado_random_forest"),
    "xgb": ("XGBoost", "valor_esperado_xgboost"),
    "ensamble": ("Ensamble (stacking)", "valor_esperado_ensamble"),
}

UMBRAL_REZAGO_ANIOS = 5
UMBRAL_ANOTACIONES_FOLIO = 40
UMBRAL_UNIDAD = 1_000
REDONDA_MIN, REDONDA_MULTIPLO = 1e10, 1e9
RAZON_EXTREMA = 10

app = FastAPI(
    title="API Radar de Transacciones Atípicas - ORIP 290 Pereira",
    version="1.0.0",
    description="Servicio de inferencia de valor esperado y detección de compraventas atípicas.",
)


class TransaccionEntrada(BaseModel):
    modelo: str = Field(default="rf", description="Clave del modelo: rf, xgb o ensamble")
    umbral_z: float = Field(default=3.5, ge=0.5, le=15.0)
    matricula: str = Field(default="290-00000")
    valor: float = Field(..., gt=0, description="Valor registrado de la compraventa en COP")
    year_radica: int = Field(default=2023, ge=2000, le=2030)
    mes_radica: int = Field(default=6, ge=1, le=12)
    tipo_predio_zona: str = Field(default="URBANO")
    categoria_ruralidad_2024: str = Field(default="NA")
    documento_justificativo: str = Field(default="00-ESCRITURA 100 DE 01/05/2023; NOTARIA PRIMERA")
    estado_folio: str = Field(default="ACTIVO")
    predios_nuevos: str = Field(default="0")
    dinamica_2024: str = Field(default="1")
    cat_comuna: str = Field(default="NA")
    antiguedad_folio: float = Field(default=10.0)
    num_anotacion: int = Field(default=10)
    count_a: int = Field(default=1)
    count_de: int = Field(default=1)
    anot_folio: int = Field(default=12)
    n_folios_doc: int = Field(default=1)
    folios_derivados_flag: float = Field(default=0.0)
    rezago: Optional[float] = Field(default=None)


class PrediccionLoteEntrada(BaseModel):
    modelo: str = Field(default="rf")
    umbral_z: float = Field(default=3.5, ge=0.5, le=15.0)
    registros: List[Dict[str, Any]]


@lru_cache(maxsize=1)
def _cargar_resumen_local() -> Dict[str, Any]:
    ruta = os.path.join(DATA_DIR, "resumen.json")
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    return {"modelos": {}}


@lru_cache(maxsize=6)
def _cargar_modelo_mlflow(clave: str):
    if clave not in MODELOS_REGISTRO:
        raise ValueError(f"Modelo no soportado: {clave}")
    _, nombre_reg = MODELOS_REGISTRO[clave]
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    versiones = client.search_model_versions(f"name='{nombre_reg}'")
    if not versiones:
        raise RuntimeError(f"No hay versiones registradas para '{nombre_reg}' en {MLFLOW_URI}")
    version = str(max(int(v.version) for v in versiones))
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre_reg}/{version}")
    return modelo, nombre_reg, version


def _preparar_features_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    defaults = {
        "matricula": "290-00000",
        "valor": 100_000_000.0,
        "year_radica": 2023.0,
        "mes_radica": 6.0,
        "tipo_predio_zona": "URBANO",
        "categoria_ruralidad_2024": "NA",
        "documento_justificativo": "00-ESCRITURA 100 DE 01/05/2023; NOTARIA PRIMERA",
        "estado_folio": "ACTIVO",
        "predios_nuevos": "0",
        "dinamica_2024": "1",
        "cat_comuna": "NA",
        "antiguedad_folio": 10.0,
        "num_anotacion": 10.0,
        "count_a": 1.0,
        "count_de": 1.0,
        "anot_folio": 12.0,
        "n_folios_doc": 1.0,
        "folios_derivados_flag": 0.0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)

    parsed = df["documento_justificativo"].apply(parse_doc)
    for c in ["doc_tipo", "doc_year", "doc_entidad"]:
        if c not in df.columns:
            df[c] = parsed[c]
        else:
            df[c] = df[c].fillna(parsed[c])

    if "rezago" not in df.columns:
        df["rezago"] = pd.to_numeric(df["year_radica"], errors="coerce") - pd.to_numeric(
            df["doc_year"], errors="coerce"
        )
    else:
        df["rezago"] = pd.to_numeric(df["rezago"], errors="coerce").fillna(
            pd.to_numeric(df["year_radica"], errors="coerce")
            - pd.to_numeric(df["doc_year"], errors="coerce")
        )

    cat_cols = [
        "tipo_predio_zona",
        "categoria_ruralidad_2024",
        "doc_tipo",
        "doc_entidad",
        "estado_folio",
        "predios_nuevos",
        "dinamica_2024",
        "cat_comuna",
    ]
    num_cols = [
        "year_radica",
        "mes_radica",
        "rezago",
        "antiguedad_folio",
        "num_anotacion",
        "count_a",
        "count_de",
        "anot_folio",
        "n_folios_doc",
        "folios_derivados_flag",
    ]
    for c in cat_cols:
        df[c] = df[c].astype(str).replace("", "NA").fillna("NA")
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    return df


def _evaluar_reglas(valor: float, valor_esperado: float, rezago: float, anot_folio: float) -> tuple[List[str], str]:
    razon = valor / valor_esperado if valor_esperado > 0 else np.nan
    activas = []
    if valor < UMBRAL_UNIDAD:
        activas.append("Posible error de unidad (< $1.000)")
    if pd.notna(razon) and (razon > RAZON_EXTREMA or razon < 1.0 / RAZON_EXTREMA):
        activas.append("Valor 10× mayor o menor que el esperado")
    if valor >= REDONDA_MIN and (valor % REDONDA_MULTIPLO == 0):
        activas.append("Cifra redonda extrema")
    if pd.notna(rezago) and rezago >= UMBRAL_REZAGO_ANIOS:
        activas.append(f"Rezago ≥ {UMBRAL_REZAGO_ANIOS} años entre documento y radicación")
    if pd.notna(anot_folio) and anot_folio >= UMBRAL_ANOTACIONES_FOLIO:
        activas.append(f"Folio con {UMBRAL_ANOTACIONES_FOLIO}+ anotaciones")
    senal_principal = activas[0] if activas else "Sin señal de regla"
    return activas, senal_principal


def _obtener_sigma(clave: str) -> float:
    resumen = _cargar_resumen_local()
    info = resumen.get("modelos", {}).get(clave, {})
    if info.get("sigma"):
        return float(info["sigma"])
    pq_path = os.path.join(DATA_DIR, f"scored_{clave}.parquet")
    if os.path.exists(pq_path):
        df_pq = pd.read_parquet(pq_path)
        if {"y", "pred_log"}.issubset(df_pq.columns):
            return float(sigma_robusto(df_pq["y"].values - df_pq["pred_log"].values))
    return 0.95


@app.get("/health")
def health():
    mlflow_ok = False
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        MlflowClient().search_experiments(max_results=1)
        mlflow_ok = True
    except Exception:
        mlflow_ok = False
    return {
        "status": "ok",
        "mlflow_uri": MLFLOW_URI,
        "mlflow_reachable": mlflow_ok,
        "data_dir": DATA_DIR,
    }


@app.get("/metrics")
def metrics():
    resumen = _cargar_resumen_local()
    return {
        "n_anotaciones": resumen.get("n_anotaciones"),
        "n_unicas": resumen.get("n_unicas"),
        "n_train": resumen.get("n_train"),
        "modelos": resumen.get("modelos", {}),
    }


@app.post("/predict")
def predict_single(payload: TransaccionEntrada):
    clave = payload.modelo.lower().strip()
    if clave not in MODELOS_REGISTRO:
        raise HTTPException(status_code=400, detail=f"Modelo inválido '{clave}'. Use: rf, xgb o ensamble.")

    try:
        modelo, nombre_reg, version = _cargar_modelo_mlflow(clave)
        df_prep = _preparar_features_df([payload.model_dump()])
        pred_log = float(np.asarray(modelo.predict(df_prep[FEATURES]), dtype=float)[0])
        origen = f"mlflow:{nombre_reg}:v{version}"
    except Exception as exc:
        pq_path = os.path.join(DATA_DIR, f"scored_{clave}.parquet")
        if not os.path.exists(pq_path):
            raise HTTPException(status_code=503, detail=f"Error cargando modelo: {exc}") from exc
        df_pq = pd.read_parquet(pq_path)
        med_log = float(df_pq["pred_log"].median())
        pred_log = med_log
        origen = f"fallback_parquet:{clave} (mlflow no disponible)"

    valor_esperado = float(np.exp(pred_log))
    sigma = _obtener_sigma(clave)
    y_log = float(np.log(payload.valor))
    z = float((y_log - pred_log) / sigma) if sigma > 0 else 0.0
    atipica = bool(abs(z) > payload.umbral_z)
    razon = float(payload.valor / valor_esperado) if valor_esperado > 0 else None
    reglas_activas, senal = _evaluar_reglas(
        payload.valor, valor_esperado, float(payload.rezago or 0.0), float(payload.anot_folio)
    )

    return {
        "modelo": clave,
        "origen_modelo": origen,
        "matricula": payload.matricula,
        "valor_registrado": payload.valor,
        "valor_esperado": valor_esperado,
        "pred_log": pred_log,
        "razon_registrado_esperado": razon,
        "sigma_robusto": sigma,
        "z": z,
        "umbral_z": payload.umbral_z,
        "atipica": atipica,
        "sentido": "sobrevalorada" if z > 0 else "subvalorada",
        "senal_revision": senal,
        "reglas_activas": reglas_activas,
    }


@app.post("/predict_batch")
def predict_batch(payload: PrediccionLoteEntrada):
    clave = payload.modelo.lower().strip()
    if clave not in MODELOS_REGISTRO:
        raise HTTPException(status_code=400, detail=f"Modelo inválido '{clave}'.")
    if not payload.registros:
        return {"modelo": clave, "total": 0, "resultados": []}

    try:
        modelo, nombre_reg, version = _cargar_modelo_mlflow(clave)
        df_prep = _preparar_features_df(payload.registros)
        preds_log = np.asarray(modelo.predict(df_prep[FEATURES]), dtype=float)
        origen = f"mlflow:{nombre_reg}:v{version}"
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"No se pudo puntuar el lote en MLflow: {exc}") from exc

    sigma = _obtener_sigma(clave)
    resultados = []
    for idx, row in df_prep.iterrows():
        val = float(row["valor"])
        p_log = float(preds_log[idx])
        v_esp = float(np.exp(p_log))
        z_val = float((np.log(val) - p_log) / sigma) if (val > 0 and sigma > 0) else 0.0
        _, senal = _evaluar_reglas(val, v_esp, float(row.get("rezago", 0.0)), float(row.get("anot_folio", 0.0)))
        resultados.append({
            "matricula": row.get("matricula", ""),
            "valor_registrado": val,
            "valor_esperado": v_esp,
            "razon": float(val / v_esp) if v_esp > 0 else None,
            "z": z_val,
            "atipica": bool(abs(z_val) > payload.umbral_z),
            "senal_revision": senal,
        })

    return {"modelo": clave, "origen_modelo": origen, "total": len(resultados), "resultados": resultados}
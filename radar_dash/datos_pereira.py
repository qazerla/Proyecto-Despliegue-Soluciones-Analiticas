#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preparación de datos compartida por los tres modelos (RF, XGBoost y ensamble).
Objetivo: predecir log(valor) de las compraventas (cod_natujur = 125) de Pereira.
Equipo DSA
"""
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42

# variables explicativas (ninguna se calcula a partir de 'valor')
CAT = ["tipo_predio_zona", "categoria_ruralidad_2024", "doc_tipo", "doc_entidad",
       "estado_folio", "predios_nuevos", "dinamica_2024", "cat_comuna"]
NUM = ["year_radica", "mes_radica", "rezago", "antiguedad_folio", "num_anotacion",
       "count_a", "count_de", "anot_folio", "n_folios_doc", "folios_derivados_flag"]
FEATURES = CAT + NUM


# Formato real: "00-ESCRITURA 1789 DE 25/08/2016; NOTARIA SEXTA DE PEREIRA"
#               "00-ESCRITURA 1768 DEL 18/08/21; PEREIRA"
RX_DOC = re.compile(
    r"^\s*(?:\d+\s*-\s*)?([A-ZÁÉÍÓÚÑ\. ]+?)\s+(?:NO\.?\s*)?([\w\-]+)\s+DEL?\s+"
    r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\s*;?\s*(.*)$")
RX_FECHA = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")


def _anio(txt):
    """Año de 2 o 4 dígitos -> 4 dígitos (2 dígitos: <=26 es 20xx, si no 19xx)."""
    a = int(txt)
    if len(txt) == 4:
        return a
    return 2000 + a if a <= 26 else 1900 + a


def parse_doc(s):
    """Extrae tipo de documento, año y entidad de documento_justificativo."""
    s = str(s).upper().strip() if pd.notna(s) else ""
    m = RX_DOC.match(s)
    if m:
        tipo, anio, entidad = m.group(1).strip(" .-"), _anio(m.group(5)), m.group(6)
    else:  # formato inesperado: se intenta rescatar al menos tipo y año
        f = RX_FECHA.search(s)
        anio = _anio(f.group(3)) if f else np.nan
        t = re.match(r"^\s*(?:\d+\s*-\s*)?([A-ZÁÉÍÓÚÑ\. ]+)", s)
        tipo = t.group(1).strip(" .-") if t else ""
        entidad = s[f.end():].strip(" ;-,:") if f else ""
    entidad = re.sub(r"\s+DE\s+PEREIRA$", "", entidad.strip(" ;-,:"))
    return pd.Series({"doc_tipo": tipo[:30] or "NA",
                      "doc_year": anio,
                      "doc_entidad": entidad[:40] or "NA"})


def cargar_datos(ruta="../pereira.csv"):
    """Lee el CSV, deja una fila por (documento, valor) y divide por año:
    entrenamiento <= 2021, validación = 2022, prueba = 2023."""
    df = pd.read_csv(ruta, sep=";", encoding="utf-8-sig", low_memory=False)
    for c in ["year_radica", "num_anotacion", "count_a", "count_de", "valor", "cod_natujur"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # columnas opcionales: si no vienen en el CSV se crean constantes
    for c in ["categoria_ruralidad_2024", "numero_catastral", "folios_derivados",
              "fecha_radica_texto", "fecha_apertura_texto"]:
        if c not in df.columns:
            df[c] = np.nan

    # fechas: mes de radicación y antigüedad del folio (años)
    f_rad = pd.to_datetime(df["fecha_radica_texto"], errors="coerce")
    f_ape = pd.to_datetime(df["fecha_apertura_texto"], errors="coerce")
    df["mes_radica"] = f_rad.dt.month
    df["antiguedad_folio"] = (f_rad - f_ape).dt.days / 365.25

    # variables a nivel de folio (con todas las anotaciones)
    df["anot_folio"] = df.groupby("matricula")["pk"].transform("size")
    df["folios_derivados_flag"] = df["folios_derivados"].notna().astype(float)

    # solo compraventas con valor positivo
    df = df[(df["cod_natujur"] == 125) & (df["valor"] > 0)].copy()

    # parseo del documento (sobre documentos únicos para ir rápido)
    docs = df["documento_justificativo"].drop_duplicates()
    parsed = docs.apply(parse_doc)
    parsed.index = docs
    df = df.join(parsed, on="documento_justificativo")
    df["rezago"] = df["year_radica"] - df["doc_year"]

    # zona catastral (primeros 11 dígitos del número catastral)
    cat = df["numero_catastral"].astype(str).str.replace(r"\D", "", regex=True)
    df["cat_comuna"] = np.where(cat.str.len() >= 11, cat.str[:11], "NA")

    # el valor pertenece al documento: una fila por (documento, valor)
    df = df.sort_values("pk")
    g = df.groupby(["documento_justificativo", "valor"], dropna=False)
    out = g.first().reset_index()
    out["n_folios_doc"] = g["matricula"].nunique().values

    for c in CAT:
        out[c] = out[c].astype(str).fillna("NA")
    for c in NUM:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    # categorías poco frecuentes -> OTRA
    for c in ["doc_tipo", "doc_entidad", "cat_comuna"]:
        top = out[c].value_counts().head(60).index
        out[c] = out[c].where(out[c].isin(top), "OTRA")

    out["y"] = np.log(out["valor"])
    out = out.reset_index(drop=True)

    train = out[out["year_radica"] <= 2021]
    val = out[out["year_radica"] == 2022]
    test = out[out["year_radica"] == 2023]

    # recorte de extremos en entrenamiento (0,5 % - 99,5 %)
    lo, hi = train["y"].quantile([0.005, 0.995])
    train = train[(train["y"] >= lo) & (train["y"] <= hi)]
    return train, val, test


def preprocesador():
    """Codifica categóricas (ordinal) e imputa numéricas con la mediana."""
    return ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1), CAT),
         ("num", SimpleImputer(strategy="median"), NUM)],
        verbose_feature_names_out=False)


def sigma_robusto(residuos):
    """Desviación robusta (MAD) para estandarizar residuos."""
    return 1.4826 * np.median(np.abs(residuos - np.median(residuos)))


def validacion_sintetica(y_real, y_pred, sigma, frac=0.05):
    """Inyecta errores de digitación (x10, x1000, /10, /1000) en una fracción
    de la prueba y mide qué tan bien |residuo|/sigma los detecta."""
    rng = np.random.default_rng(SEED)
    n = len(y_real)
    etiqueta = np.zeros(n, dtype=int)
    idx = rng.choice(n, int(n * frac), replace=False)
    etiqueta[idx] = 1
    y_mod = np.array(y_real, dtype=float).copy()
    y_mod[idx] += np.log(rng.choice([10, 1000, 0.1, 0.001], size=len(idx)))
    puntaje = np.abs(y_mod - y_pred) / sigma
    return roc_auc_score(etiqueta, puntaje), average_precision_score(etiqueta, puntaje)

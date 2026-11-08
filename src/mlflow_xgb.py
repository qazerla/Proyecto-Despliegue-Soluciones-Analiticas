#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost - valor esperado de compraventas en Pereira
Equipo DSA
"""
import os
import time

# Importe las funciones de preparación de datos del proyecto
from datos_pereira import (cargar_datos, preprocesador, sigma_robusto,
                           validacion_sintetica, FEATURES)

# Importe MLFlow para registrar los experimentos, el regresor XGBoost y las métricas
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# defina el servidor para llevar el registro de modelos y artefactos
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://34.227.194.203:8050"))
# registre el experimento
experiment = mlflow.set_experiment("experimento-dsa")

# Cargue los datos: entrenamiento (<=2021), validación (2022) y prueba (2023)
train, val, test = cargar_datos("../pereira.csv")

with mlflow.start_run(run_name="xgboost", experiment_id=experiment.experiment_id):
    # defina los parámetros del modelo
    n_estimators = 400
    max_depth = 6
    learning_rate = 0.05
    subsample = 0.8
    colsample_bytree = 0.8
    # Cree el modelo con los parámetros definidos y entrénelo
    xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                       learning_rate=learning_rate, subsample=subsample,
                       colsample_bytree=colsample_bytree, tree_method="hist",
                       random_state=42, n_jobs=-1)
    modelo = Pipeline([("prep", preprocesador()), ("model", xgb)])
    t0 = time.time()
    modelo.fit(train[FEATURES], train["y"])
    tiempo = time.time() - t0

    # Realice predicciones de validación y prueba
    pred_val = modelo.predict(val[FEATURES])
    pred_test = modelo.predict(test[FEATURES])
    sigma = sigma_robusto(val["y"].values - pred_val)
    auc, auc_pr = validacion_sintetica(test["y"].values, pred_test, sigma)

    # Registre los parámetros
    mlflow.log_param("num_trees", n_estimators)
    mlflow.log_param("maxdepth", max_depth)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("subsample", subsample)
    mlflow.log_param("colsample_bytree", colsample_bytree)
    mlflow.log_param("n_train", len(train))

    # Registre el modelo
    mlflow.sklearn.log_model(modelo, name="xgboost-model",
                             serialization_format="cloudpickle",
                             signature=infer_signature(test[FEATURES].head(50), pred_test[:50]),
                             input_example=test[FEATURES].head(5),
                             registered_model_name="valor_esperado_xgboost")

    # Cree y registre las métricas de interés
    rmse = mean_squared_error(test["y"], pred_test) ** 0.5
    mlflow.log_metric("rmse_log", rmse)
    mlflow.log_metric("mae_log", mean_absolute_error(test["y"], pred_test))
    mlflow.log_metric("r2_log", r2_score(test["y"], pred_test))
    mlflow.log_metric("sigma_robusto", sigma)
    mlflow.log_metric("roc_auc_sintetico", auc)
    mlflow.log_metric("auc_pr_sintetico", auc_pr)
    mlflow.log_metric("tiempo_entrenamiento_s", tiempo)
    print("RMSE (log):", rmse, "| AUC-PR sintético:", auc_pr)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest - valor esperado de compraventas en Pereira
Equipo DSA
"""
import os
import time

# Importe las funciones de preparación de datos del proyecto
from datos_pereira import (cargar_datos, preprocesador, sigma_robusto,
                           validacion_sintetica, FEATURES)

# Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y las métricas
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# defina el servidor para llevar el registro de modelos y artefactos
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://34.227.194.203:8050"))
# registre el experimento
experiment = mlflow.set_experiment("experimento-dsa")

# Cargue los datos: entrenamiento (<=2021), validación (2022) y prueba (2023)
train, val, test = cargar_datos("../pereira.csv")

with mlflow.start_run(run_name="random-forest", experiment_id=experiment.experiment_id):
    # defina los parámetros del modelo
    n_estimators = 200
    min_samples_leaf = 3
    max_features = 0.6
    # Cree el modelo con los parámetros definidos y entrénelo
    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                               max_features=max_features, random_state=42, n_jobs=-1)
    modelo = Pipeline([("prep", preprocesador()), ("model", rf)])
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
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("max_feat", max_features)
    mlflow.log_param("n_train", len(train))

    # Registre el modelo
    mlflow.sklearn.log_model(modelo, name="random-forest-model",
                             serialization_format="cloudpickle",
                             signature=infer_signature(test[FEATURES].head(50), pred_test[:50]),
                             input_example=test[FEATURES].head(5),
                             registered_model_name="valor_esperado_random_forest")

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

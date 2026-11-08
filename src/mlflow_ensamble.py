#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensamble (Stacking: RF + XGBoost + HistGradientBoosting -> Ridge)
Equipo DSA
"""
import os
import time

# Importe las funciones de preparación de datos del proyecto
from datos_pereira import (cargar_datos, preprocesador, sigma_robusto,
                           validacion_sintetica, FEATURES)

# Importe MLFlow para registrar los experimentos, los regresores base, el meta-modelo y las métricas
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from xgboost import XGBRegressor
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor,
                              StackingRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# defina el servidor para llevar el registro de modelos y artefactos
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://34.227.194.203:8050"))
# registre el experimento
experiment = mlflow.set_experiment("experimento-dsa")

# Cargue los datos: entrenamiento (<=2021), validación (2022) y prueba (2023)
train, val, test = cargar_datos("../pereira.csv")

with mlflow.start_run(run_name="ensamble-stacking", experiment_id=experiment.experiment_id):
    # defina los modelos base y el meta-modelo
    cv = 5
    base = [
        ("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                     max_features=0.6, random_state=42, n_jobs=-1)),
        ("xgb", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             random_state=42, n_jobs=-1)),
        ("hgb", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                              random_state=42)),
    ]
    meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
    # Cree el ensamble con los parámetros definidos y entrénelo
    stack = StackingRegressor(estimators=base, final_estimator=meta, cv=cv)
    modelo = Pipeline([("prep", preprocesador()), ("model", stack)])
    t0 = time.time()
    modelo.fit(train[FEATURES], train["y"])
    tiempo = time.time() - t0

    # Realice predicciones de validación y prueba
    pred_val = modelo.predict(val[FEATURES])
    pred_test = modelo.predict(test[FEATURES])
    sigma = sigma_robusto(val["y"].values - pred_val)
    auc, auc_pr = validacion_sintetica(test["y"].values, pred_test, sigma)

    # Registre los parámetros
    mlflow.log_param("modelos_base", "rf,xgb,hgb")
    mlflow.log_param("meta_modelo", "RidgeCV")
    mlflow.log_param("cv_folds", cv)
    mlflow.log_param("n_train", len(train))

    # Registre el modelo
    mlflow.sklearn.log_model(modelo, name="ensamble-model",
                             serialization_format="cloudpickle",
                             signature=infer_signature(test[FEATURES].head(50), pred_test[:50]),
                             input_example=test[FEATURES].head(5),
                             registered_model_name="valor_esperado_ensamble")

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

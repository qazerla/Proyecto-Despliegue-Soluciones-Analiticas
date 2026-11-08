import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor  # Usamos Regresor ya que 'valor' es continua
from sklearn.tree import DecisionTreeRegressor

# 1. Configurar la URI del servidor de MLflow remoto
mlflow.set_tracking_uri("http://34.227.194.203:8050")

# Definir el nombre del experimento requerido
mlflow.set_experiment("experimento-dsa")

# 2. Cargar el dataset evitando advertencias de tipos mixtos
try:
  df = pd.read_csv("pereira.csv", sep=";", encoding="utf-8-sig", low_memory=False)
except Exception:
  df = pd.read_csv("pereira.csv", sep=";", encoding="latin1", low_memory=False)

# Limpiar espacios en blanco en los nombres de las columnas
df.columns = df.columns.str.strip()

# 3. Definir la variable objetivo y las características (X e y)
target_column = "valor"

if target_column not in df.columns:
  raise ValueError(f"La columna '{target_column}' no se encuentra en el CSV.")

# Columnas a excluir
columns_to_drop = [
    "pk",
    "matricula",
    "fecha_radica_texto",
    "fecha_apertura_texto",
    "documento_justificativo",
    "nombre_natujur",
    "departamento",
    "municipio",
    "tipo_predio_zona",
    "categoria_ruralidad_2024",
    "estado_folio",
    target_column,
]

columns_to_drop_existing = [
    col for col in columns_to_drop if col in df.columns
]

X = df.drop(columns=columns_to_drop_existing).select_dtypes(
    include=["number"]
)
y = df[target_column]

# Limpiar y rellenar nulos
if y.dtype == object:
  y = y.astype(str).str.replace(",", ".").astype(float)

X = X.apply(
    lambda col: col.astype(str).str.replace(",", ".").astype(float)
    if col.dtype == object
    else col
)

X = X.fillna(0)
y = y.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tipos confiables requeridos por skops/MLflow para modelos basados en árboles y distancias
trusted_types = [
    "sklearn.metrics._dist_metrics.EuclideanDistance64",
    "sklearn.neighbors._kd_tree.KDTree",
    "sklearn.tree._tree.Tree",
]

# ==========================================
# MODELO 1: K-Nearest Neighbors (KNN Regressor)
# ==========================================
run_name_knn = "P-dsa-knn-k5"

with mlflow.start_run(run_name=run_name_knn):
  n_neighbors = 5
  mlflow.log_param("model_type", "KNeighborsRegressor")
  mlflow.log_param("n_neighbors", n_neighbors)
  mlflow.log_param("dataset", "pereira.csv")

  knn = KNeighborsRegressor(n_neighbors=n_neighbors)
  knn.fit(X_train, y_train)

  y_pred = knn.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  mse = mean_squared_error(y_test, y_pred)

  mlflow.log_metric("r2_score", r2)
  mlflow.log_metric("mse", mse)

  # Se pasa skops_trusted_types para autorizar los tipos internos del KNN
  mlflow.sklearn.log_model(
      knn, "model_knn", skops_trusted_types=trusted_types
  )
  print(
      f"Run '{run_name_knn}' registrado exitosamente en 'experimento-dsa'."
      f" R2: {r2:.4f}"
  )

# ==========================================
# MODELO 2: Decision Tree Regressor
# ==========================================
run_name_dt = "P-dsa-dt-depth4"

with mlflow.start_run(run_name=run_name_dt):
  max_depth = 4
  mlflow.log_param("model_type", "DecisionTreeRegressor")
  mlflow.log_param("max_depth", max_depth)
  mlflow.log_param("dataset", "pereira.csv")

  dt = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
  dt.fit(X_train, y_train)

  y_pred_dt = dt.predict(X_test)
  r2_dt = r2_score(y_test, y_pred_dt)
  mse_dt = mean_squared_error(y_test, y_pred_dt)

  mlflow.log_metric("r2_score", r2_dt)
  mlflow.log_metric("mse", mse_dt)

  mlflow.sklearn.log_model(
      dt, "model_dt", skops_trusted_types=trusted_types
  )
  print(
      f"Run '{run_name_dt}' registrado exitosamente en 'experimento-dsa'."
      f" R2: {r2_dt:.4f}"
  )
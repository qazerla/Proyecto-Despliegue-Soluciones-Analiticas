# Manual de instalación – Radar de transacciones atípicas

**Proyecto:** Radar de transacciones atípicas de Pereira (ORIP 290) · Despliegue de Soluciones Analíticas – MIAD · Grupo 8

En el manual se realizara la explicacion de como instalar y poner en marcha la API y el tablero del Radar en una máquina con Docker, ya sea una instancia EC2 de AWS o un computador local.

## 1. Componentes

| Componente | Tecnología | Puerto | Función |
|---|---|---|---|
| Servidor MLflow | MLflow en EC2 | 8050 | Guarda experimentos y el Model Registry con los modelos |
| API | FastAPI + uvicorn (contenedor Docker) | 8000 | Carga el modelo desde MLflow y puntúa compraventas |
| Tablero | Dash + waitress (contenedor Docker) | 8051 | Muestra las compraventas priorizadas y consulta la API |

El servidor de MLflow debe estar encendido antes de iniciar la API, porque la API carga los modelos desde su Model Registry.

e recomienda una máquina t3.small, con sistema operativo
Ubuntu 24.04 y 20GB de disco. Incluya un pantallazo de la consola de AWS EC2

## 2. Requisitos

- **Máquina:** Se recomienda una instancia EC2 con Ubuntu 24.04, una maquina t3.small y 20 GB de disco o un computador con Docker.
- **Software:** Git y Docker Engine.
- **Red:** grupo de seguridad con entrada TCP abierta en los puertos 22 (SSH), 8000 (API) y 8051 (tablero). La máquina debe poder conectarse al puerto 8050 del servidor de MLflow.
- **Modelos:** `valor_esperado_random_forest`, `valor_esperado_xgboost` y `valor_esperado_ensamble` registrados en el Model Registry de MLflow.

## 3. Preparar la instancia EC2

1. En la consola de AWS, lanzar una instancia Ubuntu y asociarle el par de llaves y un grupo de seguridad con los puertos 22, 8000 y 8051 abiertos.
2. Conectarse por SSH:

   ```bash
   ssh -i llave.pem ubuntu@<IP_PUBLICA>
   ```

3. Instalar Git y Docker y dar permisos al usuario:

   ```bash
   sudo apt update
   sudo apt install -y git docker.io
   sudo usermod -aG docker ubuntu
   ```

4. Cerrar la sesión SSH y volver a entrar para que el usuario pueda usar Docker sin `sudo`.

## 4. Descargar el código

```bash
git clone https://github.com/qazerla/Proyecto-Despliegue-Soluciones-Analiticas.git
cd Proyecto-Despliegue-Soluciones-Analiticas
```

Antes de construir, revisar la última línea de `radar_dash/Dockerfile`. Debe arrancar `app_v1:server`, porque el archivo del tablero se llama `app_v1.py`:

```dockerfile
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8051", "app_v1:server"]
```

Si dice `app:server`, cambiarlo; de lo contrario el contenedor del tablero se detiene al iniciar.

## 5. Construir las imágenes

Las dos imágenes se construyen desde la raíz del repositorio, porque los Dockerfiles copian archivos de varias carpetas:

```bash
docker build -f api/Dockerfile -t api:lastest .
docker build -f radar_dash/Dockerfile -t radar_dash:lastest .
```

## 6. Ejecutar los contenedores

La API y el tablero deben estar en la misma red de Docker para que el tablero encuentre la API con el nombre `api`:

```bash
docker network create radar-net

docker run -d --name api --network radar-net -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://34.227.194.203:8050 \ #Esta URI debe corresponder a la dirección definida para el MLFLOW
  --restart unless-stopped radar-api

docker run -d --name tablero --network radar-net -p 8051:8051 \
  -e API_URL=http://api:8000 \
  --restart unless-stopped radar-tablero
```

| Variable | Contenedor | Valor por defecto | Uso |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | api | `http://34.227.194.203:8050` | Dirección del servidor de MLflow |
| `API_URL` | tablero | `http://api:8000` | Dirección de la API que consulta el tablero |

## 7. Verificar la instalación

1. `docker ps` debe mostrar los contenedores `api` y `tablero` en estado `Up`.
2. La API responde en la máquina:

   ```bash
   curl http://localhost:8000/health
   ```

   La respuesta debe incluir `"status": "ok"` y `"mlflow_reachable": true`.
3. En el navegador:
   - `http://<IP_PUBLICA>:8051` abre el tablero. En el encabezado debe aparecer la etiqueta "API activa · MLflow conectado".
   - `http://<IP_PUBLICA>:8000` muestra la página de inicio de la API y `http://<IP_PUBLICA>:8000/docs`, su documentación interactiva.

Ejemplo de consulta directa a la API:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"modelo": "rf", "umbral_z": 3.5, "matricula": "290-12345", "valor": 500, "year_radica": 2023, "tipo_predio_zona": "URBANO"}'
```

## 8. Actualizar los datos del tablero (opcional)

Las vistas del tablero leen los archivos de `radar_dash/data`. Para recalcularlos con una versión nueva de los modelos:

```bash
pip install -r radar_dash/requirements.txt
dvc pull                      # descarga pereira.csv desde el remoto de DVC
export MLFLOW_TRACKING_URI=http://34.227.194.203:8050
cd radar_dash
python preparar_datos.py --csv ../pereira.csv
cd ..
```

Después, reconstruir la imagen del tablero y volver a crear su contenedor:

```bash
docker rm -f tablero
docker build -f radar_dash/Dockerfile -t radar-tablero .
docker run -d --name tablero --network radar-net -p 8051:8051 \
  -e API_URL=http://api:8000 --restart unless-stopped radar-tablero
```

Una versión nueva de un modelo en el Model Registry llega a la API sin reconstruirla: basta con reiniciar el contenedor (`docker restart api`).

## 9. Detener y reanudar

```bash
docker stop api tablero     # detiene los servicios
docker start api tablero    # los reanuda
docker logs api             # muestra los registros de la API (igual para tablero)
```

Al terminar, la instancia EC2 se **detiene** desde la consola de AWS. No se debe **terminar**, porque se pierde.

## 10. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| El contenedor del tablero se detiene al iniciar | `radar_dash/Dockerfile` arranca `app:server`; cambiarlo por `app_v1:server` y reconstruir la imagen |
| El tablero o la API no abren desde el navegador | Los puertos 8051 u 8000 no están abiertos en el grupo de seguridad |
| La etiqueta del tablero dice "modo sin conexión directa" | El tablero no alcanza la API: revisar que ambos contenedores estén en `radar-net` y que `api` esté corriendo (`docker ps`) |
| `/health` responde `"mlflow_reachable": false` | El servidor de MLflow está detenido o su puerto 8050 está cerrado; la API no puede cargar los modelos |
| En el resultado de la API, `origen_modelo` empieza por `fallback_parquet` | La API no pudo cargar el modelo desde MLflow y respondió con un valor de respaldo; no usar ese resultado hasta restablecer la conexión |
| `docker: permission denied` | El usuario no está en el grupo `docker`; ejecutar el paso 3.3 y volver a iniciar sesión |

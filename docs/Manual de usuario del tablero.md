# Manual de usuario – Radar de transacciones atípicas

**Proyecto:** Radar de transacciones atípicas de Pereira (ORIP 290) · Despliegue de Soluciones Analíticas – MIAD · Grupo 8

El Radar ayuda a quien revisa registros inmobiliarios a decidir por qué compraventas de Pereira empezar. Compara el valor registrado de cada compraventa con el valor que el modelo espera para una compraventa con esas características, y prioriza las que más se alejan.

## 1. Acceso

- Abrir en el navegador `http://<IP_PUBLICA>:8051`. No requiere usuario ni contraseña.
- La documentación de la API, para usuarios técnicos, está en `http://<IP_PUBLICA>:8000/docs`.

## 2. Conceptos clave

| Concepto | Qué significa |
|---|---|
| Valor esperado | El valor que el modelo estima para una compraventa con esas características (tipo de predio, notaría, zona, año, historia del folio, etc.) |
| z | Cuánto se aleja el valor registrado del esperado, en unidades de dispersión normal. z negativo: valor por debajo de lo esperado (subvalorada). z positivo: por encima (sobrevalorada) |
| Umbral de \|z\| | Distancia a partir de la cual una compraventa se marca como atípica |
| Atípica | Compraventa cuyo \|z\| supera el umbral. **Es una señal para revisar, no una irregularidad ni una conclusión jurídica** |
| Señal de revisión | Regla que ayuda a explicar por qué la compraventa llamó la atención |

## 3. Recorrido por el tablero

### 3.1 Encabezado

Una etiqueta indica el estado de la conexión:

- **"API activa · MLflow conectado":** todo funciona.
- **"API activa · MLflow sin enlace directo":** la API responde, pero no carga los modelos desde MLflow; las consultas en vivo no son confiables.
- **"API en … (modo sin conexión directa)":** la API no responde. Las vistas siguen funcionando, pero no se pueden evaluar compraventas nuevas.

### 3.2 Filtros

Cualquier cambio en un filtro actualiza al instante indicadores, gráficas y tablas.

| Filtro | Uso |
|---|---|
| Modelo | Modelo que calcula el valor esperado: Random Forest, XGBoost o Ensamble. Se recomienda **Random Forest**, que tiene la mejor detección (AUC-PR 0,719). Al abrir el tablero aparece seleccionado el último de la lista |
| Años | 2022, 2023 o ambos (años que el modelo no vio al entrenar) |
| Naturaleza | Fija en compraventa (125) |
| Predio | Todos, Urbano o Rural |
| Umbral \|z\| | Entre 2 y 6; 3,5 por defecto. Un umbral más alto deja una lista más corta y más precisa; uno más bajo encuentra más casos, con más falsas alarmas |

Como referencia, con el Random Forest sobre errores de digitación simulados: con umbral 3 se detecta el 71 % de los errores y la mitad de la lista son errores; con 3,5, el 65 % y seis de cada diez; con 4, el 61 % y casi nueve de cada diez.

### 3.3 Indicadores

1. **Compraventas puntuadas:** total de compraventas con los filtros activos.
2. **Marcadas como atípicas:** cuántas superan el umbral y qué porcentaje representan.
3. **Ajuste del modelo (prueba 2023):** R², RMSE y MAE en escala logarítmica.
4. **Detección sobre errores inyectados:** AUC-PR del modelo sobre errores de digitación simulados; más cerca de 1 es mejor.

### 3.4 Gráficas

| Gráfica | Cómo leerla |
|---|---|
| Atípicas por mes de radicación | En qué meses se concentran las compraventas marcadas |
| Distribución de z | Las líneas punteadas marcan el umbral a cada lado; las barras fuera de ellas son las atípicas |
| Valor registrado vs. esperado | Cada punto es una compraventa; las atípicas quedan fuera de las bandas. Al pasar el cursor sobre un punto se ve su detalle |
| Señales que acompañan a las atípicas | Cuántas atípicas tienen cada señal de revisión |
| Índice de valoración | Mediana anual del valor de las compraventas, base 2015 = 100, en valores nominales; urbano y rural por separado |
| Atípicas por tipo de predio | Porcentaje de marcadas en urbano, rural y sin información |

## 4. Revisar y exportar la lista priorizada

1. Ajustar los filtros y el umbral.
2. Revisar la tabla **"Transacciones priorizadas para revisión"**, ordenada de mayor a menor \|z\|. Cada fila muestra:
   - Folio (enmascarado, por privacidad), año y tipo de predio.
   - Valor registrado, valor esperado y la razón entre ambos.
   - z y la señal de revisión.
3. Pulsar **Exportar lista (CSV)** para descargar `transacciones_priorizadas.csv` con todas las atípicas de los filtros activos. Se abre en Excel.

Señales de revisión posibles:

| Señal | Qué sugiere |
|---|---|
| Posible error de unidad (< $1.000) | El valor parece digitado en una unidad equivocada |
| Valor 10× mayor o menor que el esperado | Diferencia de al menos un orden de magnitud con el valor esperado |
| Cifra redonda extrema | Valor muy alto y exactamente redondo |
| Rezago ≥ 5 años entre documento y radicación | El documento se radicó mucho después de firmarse |
| Folio con 40+ anotaciones | Folio con historia registral muy activa |
| Sin señal de regla | Atípica solo por su distancia al valor esperado |

## 5. Comparar modelos

La tabla **"Comparación de modelos"** muestra RMSE, MAE, R², σ robusto, AUC-PR sintético y tiempo de entrenamiento de cada modelo, con el seleccionado resaltado. Para comparar sus listas, cambiar el filtro Modelo y observar cómo cambian los indicadores y la tabla priorizada.

## 6. Evaluar una compraventa nueva con la API

1. Ir al panel **"Consulta de transacción en vivo vía API"**, al final de la página.
2. Diligenciar:
   - **Matrícula:** por ejemplo, 290-12345.
   - **Valor (COP):** en pesos, sin puntos ni comas.
   - **Año radicación:** por ejemplo, 2023.
   - **Predio:** URBANO o RURAL.
3. Pulsar **Evaluar en API**. Se usan el modelo y el umbral elegidos en los filtros.
4. Leer el resultado:
   - **Estado:** ATÍPICA (priorizar revisión) o NORMAL (dentro de banda).
   - **Valor esperado** y **razón** registrado/esperado.
   - **z** y su sentido (subvalorada o sobrevalorada).
   - **Señal** de revisión y **fuente** del modelo.

**Ejemplo:** una compraventa urbana de 2023 registrada por $500 sale ATÍPICA, con z muy negativo y la señal "Posible error de unidad (< $1.000)".

Si la fuente empieza por `fallback_parquet`, la API no se conectó a MLflow y el resultado no es confiable. Si aparece "No se pudo consultar la API", la API está detenida. En ambos casos, avisar al administrador.



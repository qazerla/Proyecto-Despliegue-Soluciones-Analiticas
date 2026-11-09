"""Tablero de transacciones atipicas (Pereira). Lee lo que deja preparar_datos.py en data/"""
import json  # para leer resumen.json
import os  # para rutas y variables de entorno
from functools import lru_cache  # guarda en memoria lo que ya se leyo

import numpy as np  # calculos con arreglos
import pandas as pd  # tablas de datos
import plotly.graph_objects as go  # graficas de plotly
from dash import Dash, Input, Output, State, dcc, html  # piezas de dash

DATA_DIR = os.environ.get("RADAR_DATA", "data")  # carpeta con los parquet (por defecto data/)
# colores que se repiten en todo el tablero
TEAL, ORANGE, SOFT, GRID, TXT, MUTED = "#0f5c5a", "#b83c0a", "#8db5b3", "#e3e7e4", "#12231f", "#566660"
FONT = "Inter, 'Segoe UI', Arial, sans-serif"  # fuente de las graficas
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]  # eje x de barras

# reglas de revision
UMBRAL_REZAGO_ANIOS = 5  # anios entre el documento y la radicacion
UMBRAL_ANOTACIONES_FOLIO = 40  # anotaciones del folio en el anio
UMBRAL_UNIDAD = 1_000  # menos de $1.000
REDONDA_MIN, REDONDA_MULTIPLO = 1e10, 1e9  # cifras redondas
RAZON_EXTREMA = 10  # 10 veces mas o menos que el esperado
REGLAS = [  # columna, texto (el orden es la prioridad)
    ("r_unidad", "Posible error de unidad (< $1.000)"),
    ("r_razon", "Valor 10× mayor o menor que el esperado"),
    ("r_redonda", "Cifra redonda extrema"),
    ("r_rezago", f"Rezago ≥ {UMBRAL_REZAGO_ANIOS} años entre documento y radicación"),
    ("r_folio", f"Folio con {UMBRAL_ANOTACIONES_FOLIO}+ anotaciones"),
]


# leer los archivos de data/
@lru_cache(maxsize=1)  # se lee una sola vez
def cargar_resumen():
    with open(os.path.join(DATA_DIR, "resumen.json"), encoding="utf-8") as f:  # abre el json
        return json.load(f)  # devuelve el diccionario


@lru_cache(maxsize=1)
def cargar_indice():
    return pd.read_parquet(os.path.join(DATA_DIR, "indice.parquet"))  # indice de valoracion


@lru_cache(maxsize=4)  # uno por modelo
def cargar_modelo(clave):
    # datos de un modelo + las reglas
    df = pd.read_parquet(os.path.join(DATA_DIR, f"scored_{clave}.parquet"))  # compraventas puntuadas
    rz = df["rezago"] if "rezago" in df else pd.Series(np.nan, index=df.index)  # rezago (si existe)
    af = df["anot_folio"] if "anot_folio" in df else pd.Series(np.nan, index=df.index)  # anotaciones por folio
    razon = df["valor"] / df["valor_esperado"]  # registrado sobre esperado
    df["razon"] = razon  # se guarda para la tabla
    df["r_unidad"] = df["valor"] < UMBRAL_UNIDAD  # regla: valor muy bajo
    df["r_razon"] = (razon > RAZON_EXTREMA) | (razon < 1 / RAZON_EXTREMA)  # regla: lejos del esperado
    df["r_redonda"] = (df["valor"] >= REDONDA_MIN) & (df["valor"] % REDONDA_MULTIPLO == 0)  # regla: cifra redonda
    df["r_rezago"] = rz >= UMBRAL_REZAGO_ANIOS  # regla: rezago largo
    df["r_folio"] = af >= UMBRAL_ANOTACIONES_FOLIO  # regla: folio con muchas anotaciones
    df["senal"] = "Sin señal de regla"  # texto por defecto
    for col, texto in reversed(REGLAS):  # la primera regla queda de ultima
        df.loc[df[col], "senal"] = texto  # escribe el texto donde se cumple
    # Rural si el texto trae RUR, si no Urbano
    df["grupo"] = np.where(df["tipo_predio_zona"].astype(str).str.upper().str.contains("RUR"), "Rural", "Urbano")
    df["anio"] = df["year_radica"].astype(int)  # anio como entero
    return df


def modelos_disponibles():
    res = cargar_resumen()["modelos"]  # modelos del resumen
    # solo los que tienen su parquet en data/
    return [{"label": v["etiqueta"], "value": k} for k, v in res.items()
            if os.path.exists(os.path.join(DATA_DIR, f"scored_{k}.parquet"))]


# formato de numeros
def n_es(x, d=0):
    # miles con punto y decimales con coma
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def dinero(v):
    if v < 1_000:  # valores muy chicos, con 2 decimales
        return f"$ {n_es(v, 2)}"
    if v >= 1e6:  # de un millon en adelante se muestra en M
        return f"$ {n_es(v / 1e6)} M"
    return f"$ {n_es(v)}"  # el resto, completo


def razon_txt(r):
    if r < 0.01:  # casi cero
        return "< 0,01×"
    return f"{n_es(r, 2 if r < 1 else (1 if r < 10 else 0))}×"  # menos decimales si es grande


def z_txt(z):
    return ("+" if z > 0 else "−") + n_es(abs(z), 1)  # z con su signo


def folio_txt(m):
    s = str(m)  # matricula como texto
    pref = s.split("-")[0] + "-" if "-" in s else ""  # prefijo antes del guion
    return f"{pref}•••••{s[-2:]}"  # solo se ven los ultimos 2 digitos


# graficas (plotly)
def base_fig(h):
    fig = go.Figure()  # figura vacia
    # fondo blanco, margenes chicos y sin leyenda
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white",
                      plot_bgcolor="white", font=dict(family=FONT, color=MUTED, size=12), showlegend=False)
    return fig


def fig_mes(df, umbral):
    # barras agrupadas por anio (go.Bar, barmode='group')
    fig = base_fig(320)
    at = df[df["z"].abs() > umbral]  # solo las atipicas
    anios = sorted(df["anio"].unique())  # anios que hay en los datos
    cols = [SOFT, TEAL, ORANGE]  # un color por anio
    for i, a in enumerate(anios):
        # conteo de atipicas por mes (rellena con 0 los meses sin datos)
        c = at[at["anio"] == a].groupby("mes_radica").size().reindex(range(1, 13), fill_value=0)
        fig.add_bar(x=MESES, y=c.values, name=str(a), marker_color=cols[i % 3],
                    hovertemplate=f"{a} · %{{x}}: %{{y}} atípicas<extra></extra>")  # texto al pasar el mouse
    fig.update_layout(barmode="group", showlegend=True, legend=dict(orientation="h", y=-0.15))  # leyenda abajo
    fig.update_yaxes(gridcolor=GRID, zeroline=False)  # lineas suaves en y
    fig.update_xaxes(showgrid=False)  # sin lineas en x
    return fig


def fig_z(df, umbral):
    # histograma con np.histogram y lineas del umbral con add_vline
    fig = base_fig(320)
    z = df["z"].clip(-8, 8)  # z extremos se juntan en -8 y 8
    cnt, edges = np.histogram(z, bins=np.linspace(-8, 8, 33))  # 32 barras
    centros = (edges[:-1] + edges[1:]) / 2  # centro de cada barra
    fig.add_bar(x=centros, y=cnt, width=0.46,
                marker_color=[ORANGE if abs(c) > umbral else TEAL for c in centros],  # naranja pasado el umbral
                hovertemplate="z ≈ %{x:.1f}: %{y}<extra></extra>")
    # una linea y un texto por cada lado del umbral
    for s, txt, anc in ((-umbral, f"−{n_es(umbral, 1)} · subvalorada", "right"),
                        (umbral, f"+{n_es(umbral, 1)} · sobrevalorada", "left")):
        fig.add_vline(x=s, line=dict(color=ORANGE, dash="dash", width=2))  # linea punteada
        fig.add_annotation(x=s, y=1.04, yref="paper", text=txt, showarrow=False,
                           xanchor=anc, font=dict(color=ORANGE, size=12))  # texto arriba de la linea
    fig.update_xaxes(range=[-8.3, 8.3], title="z = (log valor − log valor esperado) / σ robusto", showgrid=False)
    fig.update_yaxes(visible=False)  # sin eje y
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))  # mas espacio arriba para los textos
    return fig


def fig_dispersion(df, umbral, sigma):
    # dispersion en escala log, las bandas son esperado * exp(+-umbral*sigma)
    fig = base_fig(380)
    atip = df["z"].abs() > umbral  # True si es atipica
    normal = df[~atip]  # las normales
    if len(normal) > 4000:  # para que la grafica no se ponga lenta
        normal = normal.sample(4000, random_state=1)
    fig.add_scattergl(x=normal["valor_esperado"], y=normal["valor"], mode="markers",
                      marker=dict(color=TEAL, size=4, opacity=0.35), hoverinfo="skip")  # puntos normales
    a = df[atip]  # las atipicas
    fig.add_scatter(x=a["valor_esperado"], y=a["valor"], mode="markers",
                    marker=dict(color=ORANGE, size=7, opacity=0.9),
                    hovertemplate="esperado $%{x:,.0f}<br>registrado $%{y:,.0f}<extra></extra>")  # puntos naranja
    lo, hi = float(df["valor_esperado"].quantile(0.001)), float(df["valor_esperado"].quantile(0.999))  # rango de las lineas
    xs = np.array([lo, hi])
    fig.add_scatter(x=xs, y=xs, mode="lines", line=dict(color=TXT, width=1.5), hoverinfo="skip")  # linea registrado = esperado
    for s in (+1, -1):  # banda de arriba y de abajo
        fig.add_scatter(x=xs, y=xs * np.exp(s * umbral * sigma), mode="lines",
                        line=dict(color=ORANGE, width=1.5, dash="dash"), hoverinfo="skip")
    fig.update_xaxes(type="log", title="Valor esperado por el modelo ($, escala log)", gridcolor=GRID)
    fig.update_yaxes(type="log", title="Valor registrado ($)", gridcolor=GRID)
    return fig


def fig_indice():
    fig = base_fig(280)
    ind = cargar_indice()  # indice por anio y predio
    for g, color, dash in (("Urbano", TEAL, "solid"), ("Rural", ORANGE, "dash")):  # una linea por grupo
        d = ind[ind["grupo"] == g].sort_values("anio")  # datos del grupo ordenados
        fig.add_scatter(x=d["anio"], y=d["indice"], mode="lines+markers", name=g,
                        line=dict(color=color, width=2.5, dash=dash), marker=dict(size=5),
                        hovertemplate=f"{g} %{{x}}: %{{y:.1f}}<extra></extra>")
        # etiqueta al final de la linea con el ultimo valor
        fig.add_annotation(x=d["anio"].iloc[-1], y=d["indice"].iloc[-1], text=f"<b>{g} {d['indice'].iloc[-1]:.0f}</b>",
                           showarrow=False, xanchor="left", xshift=8, font=dict(color=color, size=12))
    fig.update_xaxes(dtick=2, showgrid=False)  # un anio por medio
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.update_layout(margin=dict(l=10, r=70, t=10, b=10))  # espacio a la derecha para las etiquetas
    return fig


# tablas y tarjetas
def th(t, r=False):
    return html.Th(t, className="r" if r else None)  # encabezado (r = alineado a la derecha)


def td(t, r=False, mono=False):
    cls = " ".join(c for c in ("r" if r else "", "mono" if mono else "") if c)  # clases css
    return html.Td(t, className=cls or None)  # celda


def tabla_top(df, umbral, n=15):
    top = df.loc[df["z"].abs().sort_values(ascending=False).index].head(n)  # las n con z mas alto
    top = top[top["z"].abs() > umbral]  # solo las que pasan el umbral
    # una fila por transaccion
    filas = [html.Tr([td(folio_txt(r.matricula), mono=True), td(r.anio), td(r.grupo),
                      td(dinero(r.valor), True, True), td(dinero(r.valor_esperado), True, True),
                      td(razon_txt(r.razon), True, True),
                      html.Td(html.Span(z_txt(r.z), className="z hi" if abs(r.z) >= umbral + 1.5 else "z"), className="r"),
                      td(r.senal)]) for r in top.itertuples()]
    if not filas:  # nada sobre el umbral
        filas = [html.Tr([html.Td("Sin transacciones sobre el umbral con estos filtros.", colSpan=8)])]
    return html.Table([html.Thead(html.Tr([th("Folio"), th("Año"), th("Predio"), th("Valor registrado", True),
                                           th("Valor esperado", True), th("Registrado / esperado", True),
                                           th("z", True), th("Señal de revisión")])), html.Tbody(filas)], className="tbl")


def barras_senales(at):
    n = max(len(at), 1)  # evita dividir por cero
    items = []
    for col, texto in REGLAS:
        c = int(at[col].sum())  # cuantas cumplen la regla
        p = c / n  # proporcion
        # nombre + barra + conteo
        items.append(html.Div([html.Div(texto, className="sig-n"),
                               html.Div(html.Div(style={"width": f"{min(p / 0.65, 1) * 100:.0f}%"}, className="sig-b-fill"),
                                        className="sig-b"),
                               html.Div(f"{n_es(c)} · {p * 100:.0f} %", className="sig-v")], className="sig"))
    return items


def tarjetas_predio(df):
    out = []
    for g in ("Urbano", "Rural"):
        d = df[df["grupo"] == g]  # datos del grupo
        if len(d) == 0:  # si no hay, se salta
            continue
        pct = 100 * d["atipica"].mean()  # porcentaje de atipicas
        out.append(html.Div([html.Div([html.B(g), html.Span(f"{n_es(len(d))} compraventas", className="muted")], className="seg-h"),
                             html.Div(f"{n_es(pct, 1)} %", className="seg-v"),
                             html.Div("de las compraventas puntuadas son atípicas", className="muted"),
                             html.Div(f"Mediana registrada {dinero(d['valor'].median())}", className="muted", style={"marginTop": "8px"})],
                            className="seg"))
    return out


def tabla_modelos(sel):
    res = cargar_resumen()["modelos"]
    filas = []
    for k, v in res.items():
        m = v["metricas"]  # metricas guardadas desde mlflow
        f = lambda key, d=2: "—" if m.get(key) is None else n_es(m[key], d)  # numero o raya si no hay
        t = m.get("tiempo_entrenamiento_s")
        tiempo = "—" if t is None else (f"{int(t // 60)} min {int(t % 60)} s" if t >= 60 else f"{t:.0f} s")  # segundos a min
        filas.append(html.Tr([html.Td([html.B(v["etiqueta"]), html.Div(f"{v['registro']} · v{v['version']}", className="mini")]),
                              td(f("rmse_log"), True, True), td(f("mae_log"), True, True), td(f("r2_log"), True, True),
                              td(f("sigma_robusto"), True, True), td(f("auc_pr_sintetico"), True, True), td(tiempo, True, True)],
                             className="sel" if k == sel else None))  # resalta el modelo elegido
    return html.Table([html.Thead(html.Tr([th("Modelo"), th("RMSE (log)", True), th("MAE (log)", True), th("R² (log)", True),
                                           th("σ robusto", True), th("AUC-PR sintético", True), th("Entrenamiento", True)])),
                       html.Tbody(filas)], className="tbl", style={"minWidth": "700px"})


def kpi(titulo, valor, sub, color=TXT, id_=None):
    # tarjeta chica con titulo, numero grande y texto abajo
    return html.Div([html.Div(titulo, className="kpi-t"),
                     html.Div(valor, className="kpi-v", style={"color": color}, id=f"{id_}-v"),
                     html.Div(sub, className="kpi-s", id=f"{id_}-s")], className="card kpi")


def tarjeta(titulo, sub, cuerpo, extra=None, flex="1 1 520px", sub_id=None):
    p = html.P(sub, className="sub", id=sub_id) if sub_id else html.P(sub, className="sub")  # subtitulo (con id si se actualiza)
    cab = [html.Div([html.H3(titulo), p])]  # encabezado
    if extra is not None:  # por ejemplo el boton de exportar
        cab.append(extra)
    return html.Div([html.Div(cab, className="card-h"), cuerpo], className="card", style={"flex": flex})


# filtros
def filtrar(clave, anios, predio, umbral):
    # aplica anio y predio, marca las atipicas
    df = cargar_modelo(clave)
    anios = anios or sorted(df["anio"].unique())  # si no eligen anio, todos
    df = df[df["anio"].isin(anios)].copy()  # solo los anios elegidos
    df["atipica"] = df["z"].abs() > umbral  # True si pasa el umbral
    por_anio = df  # copia sin filtrar por predio (para las tarjetas)
    if predio and predio != "Todos":
        df = df[df["grupo"] == predio]  # solo el predio elegido
    return df, por_anio


def construir(clave, anios, predio, umbral):
    # arma todo lo que devuelve el callback, en el mismo orden de SALIDAS
    res = cargar_resumen()
    info = res["modelos"][clave]  # datos del modelo elegido
    m = info["metricas"]
    df, por_anio = filtrar(clave, anios, predio, umbral)
    at = df[df["atipica"]]  # solo las atipicas
    n, na = len(df), len(at)  # total y atipicas
    pct = 100 * na / n if n else 0  # porcentaje de atipicas
    r2 = "—" if m.get("r2_log") is None else f"R² {n_es(m['r2_log'], 2)}"  # texto del R2
    ap = "—" if m.get("auc_pr_sintetico") is None else n_es(m["auc_pr_sintetico"], 2)  # texto del AUC-PR
    g = lambda k: "—" if m.get(k) is None else n_es(m[k], 2)  # metrica o raya
    ejes = (fig_mes(df, umbral), fig_z(df, umbral), fig_dispersion(df, umbral, info["sigma"]))  # las 3 graficas
    return (
        n_es(n), f"de {n_es(res['n_unicas'])} compraventas únicas · el modelo se entrenó con ≤ 2021",
        n_es(na), f"{n_es(pct, 1)} % de las puntuadas con |z| > {n_es(umbral, 1)}",
        r2, f"RMSE log {g('rmse_log')} · MAE log {g('mae_log')} · σ robusto {g('sigma_robusto')}",
        ap, "AUC-PR con errores de digitación simulados (×10, ×1000, ÷10, ÷1000)",
        *ejes, tabla_top(df, umbral),
        f"Cuántas de las {n_es(na)} marcadas cumplen cada regla de revisión", barras_senales(at),
        tarjetas_predio(por_anio), tabla_modelos(clave),
    )


# layout
app = Dash(__name__, title="Radar de transacciones atípicas")  # crea la app
server = app.server  # para waitress


def ctrl(label, comp, on=False):
    # caja con etiqueta y un control adentro
    return html.Div([html.Label(label), comp], className="ctrl on" if on else "ctrl")


def layout():
    opts = modelos_disponibles()  # modelos que hay en data/
    df0 = cargar_modelo(opts[0]["value"])
    anios = sorted(df0["anio"].unique())  # anios para el filtro
    return html.Div(className="wrap", children=[
        # fuentes de google
        html.Link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap"),
        html.Div(["Registro inmobiliario · ORIP 290 Pereira"], className="eyebrow"),
        html.H1("Radar de transacciones atípicas"),
        html.P("¿Qué compraventas de Pereira se alejan del valor que el modelo esperaba? El modelo estima un valor esperado "
               "para cada compraventa y, si el valor registrado está muy lejos (z alto), se marca como atípica.", className="lead"),
        # filtros de arriba
        html.Div([
            ctrl("Modelo", dcc.Dropdown(id="f-modelo", options=opts, value=opts[-1]["value"], clearable=False)),
            ctrl("Años", dcc.Dropdown(id="f-anios", options=[{"label": str(a), "value": int(a)} for a in anios],
                                      value=[int(a) for a in anios], multi=True)),
            ctrl("Naturaleza", html.Div("Compraventa (125)", style={"padding": "6px 0", "fontWeight": 600})),
            ctrl("Predio", dcc.Dropdown(id="f-predio", options=["Todos", "Urbano", "Rural"], value="Todos", clearable=False)),
            ctrl("Umbral |z|", dcc.Slider(id="f-umbral", min=2, max=6, step=0.5, value=3.5,
                                          marks={2: "2", 3.5: "3,5", 5: "5"}, tooltip={"always_visible": False}), on=False),
        ], className="controls"),
        # los 4 numeros grandes
        html.Div([kpi("Compraventas puntuadas", "", "", id_="k1"),
                  kpi("Marcadas como atípicas", "", "", ORANGE, id_="k2"),
                  kpi("Ajuste del modelo (prueba 2023)", "", "", id_="k3"),
                  kpi("Detección sobre errores inyectados", "", "", id_="k4")], className="row"),
        # barras por mes e histograma de z
        html.Div([tarjeta("Atípicas por mes de radicación", "Compraventas con |z| sobre el umbral, años puntuados fuera de la muestra de entrenamiento",
                          dcc.Graph(id="g-mes", config={"displayModeBar": False}), flex="1.15 1 560px"),
                  tarjeta("Distribución de z", "Cuanto más lejos de 0, más se aleja el valor registrado del esperado; el signo indica el sentido",
                          dcc.Graph(id="g-z", config={"displayModeBar": False}), flex="1 1 480px")], className="row"),
        # tabla de revision y dispersion
        html.Div([tarjeta("Transacciones priorizadas para revisión",
                          "Atípico no significa irregular: es una señal para revisar, no una conclusión jurídica.",
                          html.Div(id="t-top", className="scroll"),
                          extra=html.Div([html.Button("Exportar lista (CSV)", id="btn-csv", className="btn"), dcc.Download(id="dl")]),
                          flex="1.4 1 640px"),
                  tarjeta("Valor registrado vs. esperado", "Cada punto es una compraventa; las atípicas quedan fuera de las bandas",
                          dcc.Graph(id="g-disp", config={"displayModeBar": False}), flex="1 1 440px")], className="row"),
        # señales e indice de valoracion
        html.Div([tarjeta("Señales que acompañan a las atípicas", "", html.Div([
                      html.Div(id="t-sig", style={"marginTop": "8px"}),
                      html.Div("Las señales son reglas sobre variables del modelo (rezago, anotaciones por folio, valor). Orientan la revisión, "
                               "no explican por qué el modelo calculó ese valor esperado. Una transacción puede tener varias.", className="note")]),
                          flex="1.2 1 560px", sub_id="sig-sub"),
                  tarjeta("Índice de valoración", "Mediana de compraventas, base 2015 = 100, valores nominales",
                          [dcc.Graph(figure=fig_indice(), config={"displayModeBar": False}),
                           html.P("Continua: urbano · Discontinua: rural", className="muted")], flex="1 1 440px")], className="row"),
        # predios y comparacion de modelos
        html.Div([tarjeta("Atípicas por tipo de predio", "¿Dónde se concentran las marcadas? (años seleccionados)",
                          [html.Div(id="t-predio", className="seggrid"),
                           html.P("Corte por tipo de predio (tipo_predio_zona). El modelo ya incorpora zona, ruralidad y comuna catastral como variables.",
                                  className="muted", style={"marginTop": "16px"})], flex="1 1 420px"),
                  tarjeta("Comparación de modelos", "Los modelos registrados en MLflow",
                          [html.Div(id="t-modelos", className="scroll"),
                           html.P("Métricas registradas en MLflow sobre la prueba 2023. El modelo resaltado alimenta los gráficos.",
                                  className="muted", style={"marginTop": "14px"})], flex="1.5 1 600px")], className="row"),
        # notas del final
        html.P("Alcance: Pereira, compraventas (125) con valor positivo, una fila por documento y valor. Entrenamiento 2015–2021, "
               "σ robusto calculado en 2022, prueba 2023. Fuente: datos abiertos de Colombia, conjunto 7y2j-43cv.", className="foot"),
        html.P("Fuera de alcance: otros municipios, características físicas del inmueble, conclusiones de fraude y recomendaciones de inversión individuales.",
               className="foot"),
    ])


app.layout = layout  # dash llama a layout() cada vez que se abre la pagina

# lo que se actualiza cuando cambia un filtro
SALIDAS = [Output("k1-v", "children"), Output("k1-s", "children"), Output("k2-v", "children"), Output("k2-s", "children"),
           Output("k3-v", "children"), Output("k3-s", "children"), Output("k4-v", "children"), Output("k4-s", "children"),
           Output("g-mes", "figure"), Output("g-z", "figure"), Output("g-disp", "figure"), Output("t-top", "children"),
           Output("sig-sub", "children"), Output("t-sig", "children"), Output("t-predio", "children"), Output("t-modelos", "children")]
# los filtros que disparan el callback
ENTRADAS = [Input("f-modelo", "value"), Input("f-anios", "value"), Input("f-predio", "value"), Input("f-umbral", "value")]


@app.callback(SALIDAS, ENTRADAS)
def actualizar(clave, anios, predio, umbral):
    return construir(clave, anios, predio, umbral)  # todo el calculo esta en construir()


# State para que el boton no se dispare al cambiar un filtro
ESTADOS = [State("f-modelo", "value"), State("f-anios", "value"), State("f-predio", "value"), State("f-umbral", "value")]


@app.callback(Output("dl", "data"), Input("btn-csv", "n_clicks"), *ESTADOS, prevent_initial_call=True)
def exportar(_, clave, anios, predio, umbral):
    df, _ = filtrar(clave, anios, predio, umbral)  # mismos filtros de la pantalla
    at = df[df["atipica"]].sort_values("z", key=abs, ascending=False)  # atipicas, de mayor a menor z
    cols = ["matricula", "anio", "grupo", "valor", "valor_esperado", "razon", "z", "senal"]  # columnas del csv
    return dcc.send_data_frame(at[cols].to_csv, "transacciones_priorizadas.csv", index=False, encoding="utf-8-sig")  # descarga


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 8051)))  # puerto 8051 (el 8050 es de mlflow)

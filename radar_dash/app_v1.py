"""Tablero de transacciones atipicas (Pereira) integrado con la API de inferencia."""
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, dcc, html

DATA_DIR = os.environ.get("RADAR_DATA", "data")
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

TEAL, ORANGE, SOFT, GRID, TXT, MUTED = "#0f5c5a", "#b83c0a", "#8db5b3", "#e3e7e4", "#12231f", "#566660"
FONT = "Inter, 'Segoe UI', Arial, sans-serif"
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

UMBRAL_REZAGO_ANIOS = 5
UMBRAL_ANOTACIONES_FOLIO = 40
UMBRAL_UNIDAD = 1_000
REDONDA_MIN, REDONDA_MULTIPLO = 1e10, 1e9
RAZON_EXTREMA = 10
REGLAS = [
    ("r_unidad", "Posible error de unidad (< $1.000)"),
    ("r_razon", "Valor 10× mayor o menor que el esperado"),
    ("r_redonda", "Cifra redonda extrema"),
    ("r_rezago", f"Rezago ≥ {UMBRAL_REZAGO_ANIOS} años entre documento y radicación"),
    ("r_folio", f"Folio con {UMBRAL_ANOTACIONES_FOLIO}+ anotaciones"),
]


@lru_cache(maxsize=1)
def cargar_resumen():
    with open(os.path.join(DATA_DIR, "resumen.json"), encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def cargar_indice():
    return pd.read_parquet(os.path.join(DATA_DIR, "indice.parquet"))


@lru_cache(maxsize=4)
def cargar_modelo(clave):
    df = pd.read_parquet(os.path.join(DATA_DIR, f"scored_{clave}.parquet"))
    rz = df["rezago"] if "rezago" in df else pd.Series(np.nan, index=df.index)
    af = df["anot_folio"] if "anot_folio" in df else pd.Series(np.nan, index=df.index)
    razon = df["valor"] / df["valor_esperado"]
    df["razon"] = razon
    df["r_unidad"] = df["valor"] < UMBRAL_UNIDAD
    df["r_razon"] = (razon > RAZON_EXTREMA) | (razon < 1 / RAZON_EXTREMA)
    df["r_redonda"] = (df["valor"] >= REDONDA_MIN) & (df["valor"] % REDONDA_MULTIPLO == 0)
    df["r_rezago"] = rz >= UMBRAL_REZAGO_ANIOS
    df["r_folio"] = af >= UMBRAL_ANOTACIONES_FOLIO
    df["senal"] = "Sin señal de regla"
    for col, texto in reversed(REGLAS):
        df.loc[df[col], "senal"] = texto
    df["grupo"] = np.where(df["tipo_predio_zona"].astype(str).str.upper().str.contains("RUR"), "Rural", "Urbano")
    df["anio"] = df["year_radica"].astype(int)
    return df


def modelos_disponibles():
    res = cargar_resumen()["modelos"]
    return [
        {"label": v["etiqueta"], "value": k}
        for k, v in res.items()
        if os.path.exists(os.path.join(DATA_DIR, f"scored_{k}.parquet"))
    ]


def n_es(x, d=0):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def dinero(v):
    if v is None or np.isnan(v):
        return "—"
    if v < 1_000:
        return f"$ {n_es(v, 2)}"
    if v >= 1e6:
        return f"$ {n_es(v / 1e6)} M"
    return f"$ {n_es(v)}"


def razon_txt(r):
    if r is None or np.isnan(r):
        return "—"
    if r < 0.01:
        return "< 0,01×"
    return f"{n_es(r, 2 if r < 1 else (1 if r < 10 else 0))}×"


def z_txt(z):
    return ("+" if z > 0 else "−") + n_es(abs(z), 1)


def folio_txt(m):
    s = str(m)
    pref = s.split("-")[0] + "-" if "-" in s else ""
    return f"{pref}•••••{s[-2:]}"


def estado_api_badge():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            data = r.json()
            ml_txt = "MLflow conectado" if data.get("mlflow_reachable") else "MLflow sin enlace directo (respaldo local)"
            return html.Span(f"API activa · {ml_txt}", className="tag")
    except Exception:
        pass
    return html.Span(f"API en {API_URL} (modo sin conexión directa)", className="tag")


def base_fig(h):
    fig = go.Figure()
    fig.update_layout(
        height=h,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, color=MUTED, size=12),
        showlegend=False,
    )
    return fig


def fig_mes(df, umbral):
    fig = base_fig(320)
    at = df[df["z"].abs() > umbral]
    anios = sorted(df["anio"].unique())
    cols = [SOFT, TEAL, ORANGE]
    for i, a in enumerate(anios):
        c = at[at["anio"] == a].groupby("mes_radica").size().reindex(range(1, 13), fill_value=0)
        fig.add_bar(
            x=MESES,
            y=c.values,
            name=str(a),
            marker_color=cols[i % 3],
            hovertemplate=f"{a} · %{{x}}: %{{y}} atípicas<extra></extra>",
        )
    fig.update_layout(barmode="group", showlegend=True, legend=dict(orientation="h", y=-0.15))
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def fig_z(df, umbral):
    fig = base_fig(320)
    z = df["z"].clip(-8, 8)
    cnt, edges = np.histogram(z, bins=np.linspace(-8, 8, 33))
    centros = (edges[:-1] + edges[1:]) / 2
    fig.add_bar(
        x=centros,
        y=cnt,
        width=0.46,
        marker_color=[ORANGE if abs(c) > umbral else TEAL for c in centros],
        hovertemplate="z ≈ %{x:.1f}: %{y}<extra></extra>",
    )
    for s, txt, anc in (
        (-umbral, f"−{n_es(umbral, 1)} · subvalorada", "right"),
        (umbral, f"+{n_es(umbral, 1)} · sobrevalorada", "left"),
    ):
        fig.add_vline(x=s, line=dict(color=ORANGE, dash="dash", width=2))
        fig.add_annotation(
            x=s, y=1.04, yref="paper", text=txt, showarrow=False, xanchor=anc, font=dict(color=ORANGE, size=12)
        )
    fig.update_xaxes(range=[-8.3, 8.3], title="z = (log valor − log valor esperado) / σ robusto", showgrid=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def fig_dispersion(df, umbral, sigma):
    fig = base_fig(380)
    atip = df["z"].abs() > umbral
    normal = df[~atip]
    if len(normal) > 4000:
        normal = normal.sample(4000, random_state=1)
    fig.add_scattergl(
        x=normal["valor_esperado"],
        y=normal["valor"],
        mode="markers",
        marker=dict(color=TEAL, size=4, opacity=0.35),
        hoverinfo="skip",
    )
    a = df[atip]
    fig.add_scatter(
        x=a["valor_esperado"],
        y=a["valor"],
        mode="markers",
        marker=dict(color=ORANGE, size=7, opacity=0.9),
        hovertemplate="esperado $%{x:,.0f}<br>registrado $%{y:,.0f}<extra></extra>",
    )
    lo, hi = float(df["valor_esperado"].quantile(0.001)), float(df["valor_esperado"].quantile(0.999))
    xs = np.array([lo, hi])
    fig.add_scatter(x=xs, y=xs, mode="lines", line=dict(color=TXT, width=1.5), hoverinfo="skip")
    for s in (+1, -1):
        fig.add_scatter(
            x=xs,
            y=xs * np.exp(s * umbral * sigma),
            mode="lines",
            line=dict(color=ORANGE, width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    fig.update_xaxes(type="log", title="Valor esperado por el modelo ($, escala log)", gridcolor=GRID)
    fig.update_yaxes(type="log", title="Valor registrado ($)", gridcolor=GRID)
    return fig


def fig_indice():
    fig = base_fig(280)
    ind = cargar_indice()
    for g, color, dash in (("Urbano", TEAL, "solid"), ("Rural", ORANGE, "dash")):
        d = ind[ind["grupo"] == g].sort_values("anio")
        fig.add_scatter(
            x=d["anio"],
            y=d["indice"],
            mode="lines+markers",
            name=g,
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=5),
            hovertemplate=f"{g} %{{x}}: %{{y:.1f}}<extra></extra>",
        )
        fig.add_annotation(
            x=d["anio"].iloc[-1],
            y=d["indice"].iloc[-1],
            text=f"<b>{g} {d['indice'].iloc[-1]:.0f}</b>",
            showarrow=False,
            xanchor="left",
            xshift=8,
            font=dict(color=color, size=12),
        )
    fig.update_xaxes(dtick=2, showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.update_layout(margin=dict(l=10, r=70, t=10, b=10))
    return fig


def th(t, r=False):
    return html.Th(t, className="r" if r else None)


def td(t, r=False, mono=False):
    cls = " ".join(c for c in ("r" if r else "", "mono" if mono else "") if c)
    return html.Td(t, className=cls or None)


def tabla_top(df, umbral, n=15):
    top = df.loc[df["z"].abs().sort_values(ascending=False).index].head(n)
    top = top[top["z"].abs() > umbral]
    filas = [
        html.Tr([
            td(folio_txt(r.matricula), mono=True),
            td(r.anio),
            td(r.grupo),
            td(dinero(r.valor), True, True),
            td(dinero(r.valor_esperado), True, True),
            td(razon_txt(r.razon), True, True),
            html.Td(html.Span(z_txt(r.z), className="z hi" if abs(r.z) >= umbral + 1.5 else "z"), className="r"),
            td(r.senal),
        ])
        for r in top.itertuples()
    ]
    if not filas:
        filas = [html.Tr([html.Td("Sin transacciones sobre el umbral con estos filtros.", colSpan=8)])]
    return html.Table(
        [
            html.Thead(
                html.Tr([
                    th("Folio"),
                    th("Año"),
                    th("Predio"),
                    th("Valor registrado", True),
                    th("Valor esperado", True),
                    th("Registrado / esperado", True),
                    th("z", True),
                    th("Señal de revisión"),
                ])
            ),
            html.Tbody(filas),
        ],
        className="tbl",
    )


def barras_senales(at):
    n = max(len(at), 1)
    items = []
    for col, texto in REGLAS:
        c = int(at[col].sum())
        p = c / n
        items.append(
            html.Div(
                [
                    html.Div(texto, className="sig-n"),
                    html.Div(
                        html.Div(style={"width": f"{min(p / 0.65, 1) * 100:.0f}%"}, className="sig-b-fill"),
                        className="sig-b",
                    ),
                    html.Div(f"{n_es(c)} · {p * 100:.0f} %", className="sig-v"),
                ],
                className="sig",
            )
        )
    return items


def tarjetas_predio(df):
    out = []
    for g in ("Urbano", "Rural"):
        d = df[df["grupo"] == g]
        if len(d) == 0:
            continue
        pct = 100 * d["atipica"].mean()
        out.append(
            html.Div(
                [
                    html.Div([html.B(g), html.Span(f"{n_es(len(d))} compraventas", className="muted")], className="seg-h"),
                    html.Div(f"{n_es(pct, 1)} %", className="seg-v"),
                    html.Div("de las compraventas puntuadas son atípicas", className="muted"),
                    html.Div(
                        f"Mediana registrada {dinero(d['valor'].median())}",
                        className="muted",
                        style={"marginTop": "8px"},
                    ),
                ],
                className="seg",
            )
        )
    return out


def tabla_modelos(sel):
    res = cargar_resumen()["modelos"]
    filas = []
    for k, v in res.items():
        m = v["metricas"]
        f = lambda key, d=2: "—" if m.get(key) is None else n_es(m[key], d)
        t = m.get("tiempo_entrenamiento_s")
        tiempo = "—" if t is None else (f"{int(t // 60)} min {int(t % 60)} s" if t >= 60 else f"{t:.0f} s")
        filas.append(
            html.Tr(
                [
                    html.Td([html.B(v["etiqueta"]), html.Div(f"{v['registro']} · v{v['version']}", className="mini")]),
                    td(f("rmse_log"), True, True),
                    td(f("mae_log"), True, True),
                    td(f("r2_log"), True, True),
                    td(f("sigma_robusto"), True, True),
                    td(f("auc_pr_sintetico"), True, True),
                    td(tiempo, True, True),
                ],
                className="sel" if k == sel else None,
            )
        )
    return html.Table(
        [
            html.Thead(
                html.Tr([
                    th("Modelo"),
                    th("RMSE (log)", True),
                    th("MAE (log)", True),
                    th("R² (log)", True),
                    th("σ robusto", True),
                    th("AUC-PR sintético", True),
                    th("Entrenamiento", True),
                ])
            ),
            html.Tbody(filas),
        ],
        className="tbl",
        style={"minWidth": "700px"},
    )


def kpi(titulo, valor, sub, color=TXT, id_=None):
    return html.Div(
        [
            html.Div(titulo, className="kpi-t"),
            html.Div(valor, className="kpi-v", style={"color": color}, id=f"{id_}-v"),
            html.Div(sub, className="kpi-s", id=f"{id_}-s"),
        ],
        className="card kpi",
    )


def tarjeta(titulo, sub, cuerpo, extra=None, flex="1 1 520px", sub_id=None):
    p = html.P(sub, className="sub", id=sub_id) if sub_id else html.P(sub, className="sub")
    cab = [html.Div([html.H3(titulo), p])]
    if extra is not None:
        cab.append(extra)
    return html.Div([html.Div(cab, className="card-h"), cuerpo], className="card", style={"flex": flex})


def filtrar(clave, anios, predio, umbral):
    df = cargar_modelo(clave)
    anios = anios or sorted(df["anio"].unique())
    df = df[df["anio"].isin(anios)].copy()
    df["atipica"] = df["z"].abs() > umbral
    por_anio = df
    if predio and predio != "Todos":
        df = df[df["grupo"] == predio]
    return df, por_anio


def construir(clave, anios, predio, umbral):
    res = cargar_resumen()
    info = res["modelos"][clave]
    m = info["metricas"]
    df, por_anio = filtrar(clave, anios, predio, umbral)
    at = df[df["atipica"]]
    n, na = len(df), len(at)
    pct = 100 * na / n if n else 0
    r2 = "—" if m.get("r2_log") is None else f"R² {n_es(m['r2_log'], 2)}"
    ap = "—" if m.get("auc_pr_sintetico") is None else n_es(m["auc_pr_sintetico"], 2)
    g = lambda k: "—" if m.get(k) is None else n_es(m[k], 2)
    ejes = (fig_mes(df, umbral), fig_z(df, umbral), fig_dispersion(df, umbral, info["sigma"]))
    return (
        n_es(n),
        f"de {n_es(res['n_unicas'])} compraventas únicas · el modelo se entrenó con ≤ 2021",
        n_es(na),
        f"{n_es(pct, 1)} % de las puntuadas con |z| > {n_es(umbral, 1)}",
        r2,
        f"RMSE log {g('rmse_log')} · MAE log {g('mae_log')} · σ robusto {g('sigma_robusto')}",
        ap,
        "AUC-PR con errores de digitación simulados (×10, ×1000, ÷10, ÷1000)",
        *ejes,
        tabla_top(df, umbral),
        f"Cuántas de las {n_es(na)} marcadas cumplen cada regla de revisión",
        barras_senales(at),
        tarjetas_predio(por_anio),
        tabla_modelos(clave),
    )


app = Dash(__name__, title="Radar de transacciones atípicas")
server = app.server


def ctrl(label, comp, on=False):
    return html.Div([html.Label(label), comp], className="ctrl on" if on else "ctrl")


def layout():
    opts = modelos_disponibles()
    df0 = cargar_modelo(opts[0]["value"])
    anios = sorted(df0["anio"].unique())
    cuerpo_api = html.Div([
        html.Div(
            [
                ctrl("Matrícula", dcc.Input(id="sim-mat", type="text", value="290-99999", style={"width": "100%"})),
                ctrl("Valor (COP)", dcc.Input(id="sim-val", type="number", value=500, style={"width": "100%"})),
                ctrl("Año radicación", dcc.Input(id="sim-anio", type="number", value=2023, style={"width": "100%"})),
                ctrl(
                    "Tipo predio",
                    dcc.Dropdown(id="sim-predio", options=["URBANO", "RURAL"], value="URBANO", clearable=False),
                ),
                html.Button("Evaluar en API", id="btn-api", className="btn"),
            ],
            className="controls",
            style={"marginTop": "14px", "marginBottom": "10px"},
        ),
        html.Div(id="res-api", className="note"),
    ])

    return html.Div(
        className="wrap",
        children=[
            html.Link(
                rel="stylesheet",
                href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap",
            ),
            html.Div(["Registro inmobiliario · ORIP 290 Pereira", estado_api_badge()], className="eyebrow"),
            html.H1("Radar de transacciones atípicas"),
            html.P(
                "¿Qué compraventas de Pereira se alejan del valor que el modelo esperaba? El modelo estima un valor esperado "
                "para cada compraventa y, si el valor registrado está muy lejos (z alto), se marca como atípica.",
                className="lead",
            ),
            html.Div(
                [
                    ctrl("Modelo", dcc.Dropdown(id="f-modelo", options=opts, value=opts[-1]["value"], clearable=False)),
                    ctrl(
                        "Años",
                        dcc.Dropdown(
                            id="f-anios",
                            options=[{"label": str(a), "value": int(a)} for a in anios],
                            value=[int(a) for a in anios],
                            multi=True,
                        ),
                    ),
                    ctrl("Naturaleza", html.Div("Compraventa (125)", style={"padding": "6px 0", "fontWeight": 600})),
                    ctrl(
                        "Predio",
                        dcc.Dropdown(id="f-predio", options=["Todos", "Urbano", "Rural"], value="Todos", clearable=False),
                    ),
                    ctrl(
                        "Umbral |z|",
                        dcc.Slider(
                            id="f-umbral",
                            min=2,
                            max=6,
                            step=0.5,
                            value=3.5,
                            marks={2: "2", 3.5: "3,5", 5: "5"},
                            tooltip={"always_visible": False},
                        ),
                        on=False,
                    ),
                ],
                className="controls",
            ),
            html.Div(
                [
                    kpi("Compraventas puntuadas", "", "", id_="k1"),
                    kpi("Marcadas como atípicas", "", "", ORANGE, id_="k2"),
                    kpi("Ajuste del modelo (prueba 2023)", "", "", id_="k3"),
                    kpi("Detección sobre errores inyectados", "", "", id_="k4"),
                ],
                className="row",
            ),
            html.Div(
                [
                    tarjeta(
                        "Atípicas por mes de radicación",
                        "Compraventas con |z| sobre el umbral, años puntuados fuera de la muestra de entrenamiento",
                        dcc.Graph(id="g-mes", config={"displayModeBar": False}),
                        flex="1.15 1 560px",
                    ),
                    tarjeta(
                        "Distribución de z",
                        "Cuanto más lejos de 0, más se aleja el valor registrado del esperado; el signo indica el sentido",
                        dcc.Graph(id="g-z", config={"displayModeBar": False}),
                        flex="1 1 480px",
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    tarjeta(
                        "Transacciones priorizadas para revisión",
                        "Atípico no significa irregular: es una señal para revisar, no una conclusión jurídica.",
                        html.Div(id="t-top", className="scroll"),
                        extra=html.Div([
                            html.Button("Exportar lista (CSV)", id="btn-csv", className="btn"),
                            dcc.Download(id="dl"),
                        ]),
                        flex="1.4 1 640px",
                    ),
                    tarjeta(
                        "Valor registrado vs. esperado",
                        "Cada punto es una compraventa; las atípicas quedan fuera de las bandas",
                        dcc.Graph(id="g-disp", config={"displayModeBar": False}),
                        flex="1 1 440px",
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    tarjeta(
                        "Señales que acompañan a las atípicas",
                        "",
                        html.Div([
                            html.Div(id="t-sig", style={"marginTop": "8px"}),
                            html.Div(
                                "Las señales son reglas sobre variables del modelo (rezago, anotaciones por folio, valor). Orientan la revisión, "
                                "no explican por qué el modelo calculó ese valor esperado. Una transacción puede tener varias.",
                                className="note",
                            ),
                        ]),
                        flex="1.2 1 560px",
                        sub_id="sig-sub",
                    ),
                    tarjeta(
                        "Índice de valoración",
                        "Mediana de compraventas, base 2015 = 100, valores nominales",
                        [
                            dcc.Graph(figure=fig_indice(), config={"displayModeBar": False}),
                            html.P("Continua: urbano · Discontinua: rural", className="muted"),
                        ],
                        flex="1 1 440px",
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    tarjeta(
                        "Atípicas por tipo de predio",
                        "¿Dónde se concentran las marcadas? (años seleccionados)",
                        [
                            html.Div(id="t-predio", className="seggrid"),
                            html.P(
                                "Corte por tipo de predio (tipo_predio_zona). El modelo ya incorpora zona, ruralidad y comuna catastral como variables.",
                                className="muted",
                                style={"marginTop": "16px"},
                            ),
                        ],
                        flex="1 1 420px",
                    ),
                    tarjeta(
                        "Comparación de modelos",
                        "Los modelos registrados en MLflow",
                        [
                            html.Div(id="t-modelos", className="scroll"),
                            html.P(
                                "Métricas registradas en MLflow sobre la prueba 2023. El modelo resaltado alimenta los gráficos.",
                                className="muted",
                                style={"marginTop": "14px"},
                            ),
                        ],
                        flex="1.5 1 600px",
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    tarjeta(
                        "Consulta de transacción en vivo vía API",
                        f"Envía una compraventa al servicio REST ({API_URL}/predict) para obtener valor esperado, puntaje z y señal.",
                        cuerpo_api,
                        flex="1 1 100%",
                    )
                ],
                className="row",
            ),
            html.P(
                "Alcance: Pereira, compraventas (125) con valor positivo, una fila por documento y valor. Entrenamiento 2015–2021, "
                "σ robusto calculado en 2022, prueba 2023. Fuente: datos abiertos de Colombia, conjunto 7y2j-43cv.",
                className="foot",
            ),
            html.P(
                "Fuera de alcance: otros municipios, características físicas del inmueble, conclusiones de fraude y recomendaciones de inversión individuales.",
                className="foot",
            ),
        ],
    )


app.layout = layout

SALIDAS = [
    Output("k1-v", "children"),
    Output("k1-s", "children"),
    Output("k2-v", "children"),
    Output("k2-s", "children"),
    Output("k3-v", "children"),
    Output("k3-s", "children"),
    Output("k4-v", "children"),
    Output("k4-s", "children"),
    Output("g-mes", "figure"),
    Output("g-z", "figure"),
    Output("g-disp", "figure"),
    Output("t-top", "children"),
    Output("sig-sub", "children"),
    Output("t-sig", "children"),
    Output("t-predio", "children"),
    Output("t-modelos", "children"),
]
ENTRADAS = [Input("f-modelo", "value"), Input("f-anios", "value"), Input("f-predio", "value"), Input("f-umbral", "value")]


@app.callback(SALIDAS, ENTRADAS)
def actualizar(clave, anios, predio, umbral):
    return construir(clave, anios, predio, umbral)


ESTADOS = [State("f-modelo", "value"), State("f-anios", "value"), State("f-predio", "value"), State("f-umbral", "value")]


@app.callback(Output("dl", "data"), Input("btn-csv", "n_clicks"), *ESTADOS, prevent_initial_call=True)
def exportar(_, clave, anios, predio, umbral):
    df, _ = filtrar(clave, anios, predio, umbral)
    at = df[df["atipica"]].sort_values("z", key=abs, ascending=False)
    cols = ["matricula", "anio", "grupo", "valor", "valor_esperado", "razon", "z", "senal"]
    return dcc.send_data_frame(at[cols].to_csv, "transacciones_priorizadas.csv", index=False, encoding="utf-8-sig")


@app.callback(
    Output("res-api", "children"),
    Input("btn-api", "n_clicks"),
    State("sim-mat", "value"),
    State("sim-val", "value"),
    State("sim-anio", "value"),
    State("sim-predio", "value"),
    State("f-modelo", "value"),
    State("f-umbral", "value"),
    prevent_initial_call=True,
)
def consultar_api(_, mat, val, anio, predio, clave, umbral):
    payload = {
        "modelo": clave,
        "umbral_z": float(umbral or 3.5),
        "matricula": str(mat or "290-99999"),
        "valor": float(val or 100_000_000),
        "year_radica": int(anio or 2023),
        "tipo_predio_zona": str(predio or "URBANO"),
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        resp.raise_for_status()
        d = resp.json()
        estado = "ATÍPICA (priorizar revisión)" if d["atipica"] else "NORMAL (dentro de banda)"
        return (
            f"Estado: {estado} · Valor esperado: {dinero(d['valor_esperado'])} · "
            f"z: {z_txt(d['z'])} ({d['sentido']}) · Razón: {razon_txt(d.get('razon_registrado_esperado'))} · "
            f"Señal: {d['senal_revision']} · Fuente: {d['origen_modelo']}"
        )
    except Exception as exc:
        return f"No se pudo consultar la API en {API_URL}/predict: {exc}"


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8051)))
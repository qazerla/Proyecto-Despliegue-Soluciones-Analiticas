# -*- coding: utf-8 -*-
"""Guía 4 · Calidad, limpieza y entendimiento de los datos
Registro de Transacciones Inmobiliarias IGAC 2015-2023, municipio de Pereira.
Entrada : pereira.csv (separador ';', codificación utf-8-sig)
Salida  : resultados.json, embudo_pares.png, distribucion_valor.png, base_limpia_compraventas.csv
"""
import re, json
import numpy as np, pandas as pd
from scipy.stats import skew
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUTA = "pereira.csv"
UMBRAL_VALOR = 5_000_000      # regla A4
FOLIOS_CONJUNTA = 3           # se excluye si el mismo documento y valor está en >= 3 folios
R = {}

df = pd.read_csv(RUTA, sep=";", encoding="utf-8-sig", dtype={"pk": str, "matricula": str}, low_memory=False)

#1. CALIDAD
R["filas"] = len(df); R["folios"] = int(df.matricula.nunique()); R["columnas"] = df.shape[1]
R["pk_duplicados"] = int(df.pk.duplicated().sum())
R["filas_duplicadas"] = int(df.duplicated().sum())
R["constantes"] = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
R["nulos_pct"] = (df.isna().mean() * 100).round(2).loc[lambda s: s > 0].to_dict()
R["valor_cero"] = int((df.valor == 0).sum()); R["valor_cero_pct"] = round(100 * (df.valor == 0).mean(), 2)
R["valor_pos"] = int((df.valor > 0).sum())
R["tiene_valor_uno_con_valor_cero"] = int(((df.tiene_valor == 1) & (df.valor == 0)).sum())
R["tiene_mas_de_un_valor"] = int((df.tiene_mas_de_un_valor == 1).sum())
p = df.pk.str.split("-", expand=True)
mat = df.matricula.str.split("-", expand=True)
R["pk_vs_campos_discrepancias"] = int(((pd.to_numeric(p[4]) != df.cod_natujur) | (pd.to_numeric(p[5]) != df.year_radica) |
                                       (pd.to_numeric(p[3]) != df.num_anotacion) | (pd.to_numeric(p[1]) != pd.to_numeric(mat[0])) |
                                       (pd.to_numeric(p[2]) != pd.to_numeric(mat[1]))).sum())
R["nombres_con_2_codigos"] = df.groupby("nombre_natujur").cod_natujur.nunique().loc[lambda s: s > 1].index.tolist()
fr = pd.to_datetime(df.fecha_radica_texto, errors="coerce"); fa = pd.to_datetime(df.fecha_apertura_texto, errors="coerce")
R["fecha_radica_vs_year_discrepancias"] = int((fr.dt.year != df.year_radica)[fr.notna()].sum())
R["fecha_apertura_centinela_1899"] = int((fa < "1900-01-01").sum())
R["tipo_predio"] = df.tipo_predio_zona.value_counts().to_dict()
R["natujur_distintas"] = int(df.nombre_natujur.nunique())
R["top10_natujur_pct"] = round(100 * df.nombre_natujur.value_counts(normalize=True).head(10).sum(), 1)

# 2. PARSEO DEL DOCUMENTO JUSTIFICATIVO (compraventas)
pat = re.compile(r"^\s*\d+-([A-ZÁÉÍÓÚÑ ]+?)\s+([\d\.\-A-Z/]+)\s+(?:DE|DEL)\s+(\d{1,2}/\d{1,2}/\d{2,4})\s*;\s*(.*)$")
def parsear(t):
    m = pat.match(str(t))
    if not m: return (None, None, pd.NaT, None, False)
    tipo, num, f, ent = m.groups(); d, mm, y = f.split("/"); dos = len(y) == 2
    y = "20" + y if dos else y
    return (tipo.strip(), num, pd.to_datetime(f"{d}/{mm}/{y}", format="%d/%m/%Y", errors="coerce"), ent.strip(), dos)

c = df[df.nombre_natujur == "COMPRAVENTA"].copy()
P = c.documento_justificativo.map(parsear)
c["doc_tipo"] = [x[0] for x in P]; c["doc_num"] = [x[1] for x in P]
c["doc_fecha"] = pd.to_datetime([x[2] for x in P]); c["doc_ent"] = [x[3] for x in P]; c["doc_anio2"] = [x[4] for x in P]
c["doc_key"] = np.where(c.doc_tipo.notna(), c.doc_tipo + "|" + c.doc_num.fillna("") + "|" + c.doc_fecha.dt.strftime("%Y-%m-%d").fillna(""), c.documento_justificativo)
R["compraventas"] = len(c)
R["cv_cod_distintos"] = c.cod_natujur.value_counts().to_dict()
R["cv_nulos_fecha_radica_pct"] = round(100 * pd.to_datetime(c.fecha_radica_texto, errors="coerce").isna().mean(), 2)
R["cv_doc_parseado_pct"] = round(100 * c.doc_tipo.notna().mean(), 3)
R["cv_doc_escritura_pct"] = round(100 * (c.doc_tipo == "ESCRITURA").mean(), 2)
R["cv_doc_anio_2_digitos_pct"] = round(100 * c.doc_anio2.mean(), 1)
R["cv_entidad_sin_notaria_pct"] = round(100 * (~c.doc_ent.fillna("").str.contains("NOTARIA")).mean(), 1)
R["cv_doc_fecha_posterior_anio_radica"] = int((c.doc_fecha.dt.year > c.year_radica).sum())
R["cv_rezago_mayor_5"] = int(((c.year_radica - c.doc_fecha.dt.year) > 5).sum())
R["cv_por_anio"] = c.groupby("year_radica").size().to_dict()
vpos = c[c.valor > 0].valor
R["cv_skew_valor"] = round(float(skew(vpos)), 1); R["cv_skew_log10"] = round(float(skew(np.log10(vpos))), 2)
R["cv_valor_menor_1000"] = int(((c.valor > 0) & (c.valor < 1000)).sum())

#3. LIMPIEZA
# embudo de reglas
def pares(d):
    m = d.groupby("matricula").size()
    return dict(compraventas=len(d), matriculas=int(len(m)), con_2_o_mas=int((m >= 2).sum()), pares=int((m - 1).clip(lower=0).sum()))
E = []
E.append(("Compraventas en el registro", pares(c)))
c1 = c[c.valor > 0]; E.append(("R1. Se excluye valor = 0", pares(c1)))
c2 = c1[c1.valor >= UMBRAL_VALOR]; E.append(("R2. Se excluye valor < $5 millones", pares(c2)))
dup = c2.duplicated(subset=["matricula", "doc_key"], keep="first")
c3 = c2[~dup].copy(); E.append(("R3. Se eliminan duplicados (misma matrícula y mismo documento)", pares(c3)))
c3["n_folios_doc"] = c3.groupby(["doc_key", "valor"]).matricula.transform("nunique")
c4 = c3[c3.n_folios_doc < FOLIOS_CONJUNTA].copy(); E.append(("R4. Se excluyen escrituras con el mismo valor en 3 o más folios", pares(c4)))
c4b = c3[c3.n_folios_doc < 2]; E.append(("Sensibilidad: se excluyen también las de 2 folios", pares(c4b)))
R["embudo"] = [(n, v) for n, v in E]
R["valor_cero_compraventas"] = int((c.valor == 0).sum()); R["valor_bajo_compraventas"] = int(((c.valor > 0) & (c.valor < UMBRAL_VALOR)).sum())
R["duplicados_doc"] = int(dup.sum())
g = c3.groupby(["doc_key", "valor"]).matricula.nunique()
R["conjuntas_compraventas_2mas"] = int((c3.n_folios_doc >= 2).sum()); R["conjuntas_documentos_2mas"] = int((g >= 2).sum())
R["conjuntas_compraventas_3mas"] = int((c3.n_folios_doc >= 3).sum()); R["conjuntas_documentos_3mas"] = int((g >= 3).sum())
R["conjuntas_compraventas_solo2"] = int((c3.n_folios_doc == 2).sum())
R["folios_por_doc"] = g.clip(upper=10).value_counts().sort_index().to_dict()
R["meta_80pct"] = int(np.ceil(0.8 * E[2][1]["pares"]))
R["sensibilidad_folios"] = {k: pares(c3[c3.n_folios_doc <= k]) for k in (1, 2, 3, 4, 5)}

# pares de la base limpia
s = c4.sort_values(["matricula", "year_radica", "num_anotacion"]).copy()
for col in ("valor", "year_radica", "doc_fecha", "doc_key"): s["p_" + col] = s.groupby("matricula")[col].shift()
pr = s.dropna(subset=["p_year_radica"]).copy()
pr["dy"] = pr.year_radica - pr.p_year_radica
pr["log_ratio"] = np.log(pr.valor / pr.p_valor)
R["pares_por_anio_segunda"] = pr.groupby("year_radica").size().to_dict()
R["pares_mismo_anio"] = int((pr.dy == 0).sum()); R["pares_mismo_valor"] = int((pr.valor == pr.p_valor).sum())
R["pares_orden_doc_contradictorio"] = int((pr.doc_fecha < pr.p_doc_fecha).sum())
R["pares_dy_mediana"] = float(pr.dy.median())
pa = pr[pr.dy >= 1].copy(); pa["var_anual_pct"] = (np.exp(pa.log_ratio / pa.dy) - 1) * 100
R["pares_dy1_n"] = len(pa); R["var_anual_mediana_pct"] = round(float(pa.var_anual_pct.median()), 1)
R["var_anual_p25_p75"] = [round(float(pa.var_anual_pct.quantile(.25)), 1), round(float(pa.var_anual_pct.quantile(.75)), 1)]
R["var_anual_mediana_por_anio"] = pa.groupby("year_radica").var_anual_pct.median().round(1).to_dict()

# atípicos (se marcan)
lv = np.log10(c4.valor); q1, q3 = lv.quantile([.25, .75]); iqr = q3 - q1
f_mod, f_ext = 10 ** (q3 + 1.5 * iqr), 10 ** (q3 + 3 * iqr)
c4["atipico_moderado"] = c4.valor > f_mod; c4["atipico_extremo"] = c4.valor > f_ext
R["limite_moderado"] = round(float(f_mod)); R["limite_extremo"] = round(float(f_ext))
R["atipicos_moderados"] = int(c4.atipico_moderado.sum()); R["atipicos_extremos"] = int(c4.atipico_extremo.sum())
R["max_valor_base_limpia"] = float(c4.valor.max())
R["skew_base_limpia"] = round(float(skew(c4.valor)), 1); R["skew_log_base_limpia"] = round(float(skew(lv)), 2)
y = c4.groupby("year_radica").valor
R["valor_por_anio"] = {int(k): dict(n=int(len(v)), p10=float(v.quantile(.1)), mediana=float(v.median()), p90=float(v.quantile(.9))) for k, v in y}

# trayectoria por matrícula
fol = df.groupby("matricula").size(); R["anot_por_folio"] = {k: float(v) for k, v in fol.describe(percentiles=[.5, .9, .99]).round(1).items()}
R["folios_mas_50_anotaciones"] = int((fol > 50).sum())

# 4. SALIDAS
c4.drop(columns=["doc_anio2"]).to_csv("base_limpia_compraventas.csv", index=False, sep=";", encoding="utf-8-sig")
json.dump(R, open("resultados.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

fig, ax = plt.subplots(figsize=(9, 3.8))
nombres = ["Compraventas\nen el registro", "R1\nvalor > 0", "R2\n≥ $5 millones", "R3\nsin duplicados", "R4\nsin escrituras\nen 3+ folios"]
vals = [v["pares"] for _, v in E[:4]] + [E[4][1]["pares"]]
b = ax.bar(nombres, vals, color=["#9FB0C8"] * 3 + ["#9FB0C8", "#5B2A86"])
ax.axhline(R["meta_80pct"], color="#C0392B", ls="--", lw=1.2); ax.text(-0.45, R["meta_80pct"] + 250, "Meta: 80 % de 19.359 = 15.488 pares", color="#C0392B", ha="left", fontsize=8)
for r_, v in zip(b, vals): ax.text(r_.get_x() + r_.get_width() / 2, v + 250, f"{v:,}".replace(",", "."), ha="center", fontsize=8)
ax.set_ylabel("Pares de ventas repetidas"); ax.set_title("Pares de ventas repetidas tras cada regla de limpieza", fontsize=10); ax.tick_params(axis="x", labelsize=8)
plt.tight_layout(); plt.savefig("embudo_pares.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.hist(np.log10(c1.valor), bins=80, color="#C9D3E3", label="Compraventas con valor > 0")
ax.hist(lv, bins=80, color="#5B2A86", alpha=.85, label="Base limpia (R1–R4)")
for x, t in ((np.log10(UMBRAL_VALOR), "$5 millones"), (np.log10(f_ext), "límite de atípicos extremos")):
    ax.axvline(x, color="#C0392B", ls="--", lw=1); ax.text(x, ax.get_ylim()[1] * .92, " " + t, color="#C0392B", fontsize=8)
ax.set_xticks(range(3, 13)); ax.set_xticklabels([f"$10^{i}$" for i in range(3, 13)], fontsize=8)
ax.set_xlabel("Valor de la compraventa (escala logarítmica, pesos)"); ax.set_ylabel("Compraventas"); ax.legend(fontsize=8)
ax.set_title("Distribución del valor antes y después de la limpieza", fontsize=10)
plt.tight_layout(); plt.savefig("distribucion_valor.png", dpi=170); plt.close()
print("OK")

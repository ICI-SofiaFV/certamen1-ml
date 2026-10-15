# =============================================================================
# Dashboard · Certamen 1 · Grupo 15
# ¿A qué clientes llamar? Costo de los errores y umbral óptimo
# Datos: probabilidades en escala real del Random Forest sobre el test
# =============================================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Umbral óptimo · Seguro de vehículo", page_icon="🚗", layout="wide")

# --- Paleta (la misma que se usa en la presentación de Canva) ---
NAVY, TEAL, ROJO, VERDE, GRIS, ARENA = "#14304F", "#1A8C8C", "#C8414B", "#2E8B57", "#8A94A6", "#F4F1EA"


# --- Formato chileno: punto para miles, coma para decimales ---
def usd(x):
    return "US$ " + f"{x:,.0f}".replace(",", ".")


def num(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, dec=1):
    return num(100 * x, dec) + " %"


# --- Estilos (banner, tarjetas y mensaje principal) ---
st.markdown(f"""
<style>
.block-container {{padding-top: 3rem;}}
.banner {{background: linear-gradient(120deg, {NAVY} 0%, {TEAL} 100%); color: white;
          padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1rem;}}
.banner h1 {{color: white; font-size: 2rem; margin: 0 0 .3rem 0; padding: 0;}}
.banner p {{margin: 0; opacity: .92; font-size: 1rem;}}
.card {{background: white; border: 1px solid #E3E8EF; border-left: 6px solid var(--c);
        border-radius: 12px; padding: .9rem 1rem; min-height: 132px;}}
.card .t {{font-size: .82rem; color: #5B6575; margin-bottom: .2rem;}}
.card .v {{font-size: 1.65rem; font-weight: 700; color: {NAVY}; line-height: 1.2;}}
.card .s {{font-size: .78rem; color: #5B6575; margin-top: .2rem;}}
.msg {{border-radius: 12px; padding: .9rem 1.2rem; margin: 1rem 0 .4rem 0; font-size: 1.05rem;}}
</style>
""", unsafe_allow_html=True)


def tarjeta(col, icono, titulo, valor, sub, color):
    col.markdown(f'<div class="card" style="--c:{color}"><div class="t">{icono} {titulo}</div>'
                 f'<div class="v">{valor}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 1) Datos: y_test (real) + proba (probabilidad de compra en escala real)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("predicciones_test.csv")
    p_pesos = df["proba_pesos_etapa_B"].to_numpy() if "proba_pesos_etapa_B" in df else None
    return df["y_test"].to_numpy(), df["proba"].to_numpy(), p_pesos


y, p, p_pesos = cargar_datos()
P, N = int((y == 1).sum()), int((y == 0).sum())            # compradores y no compradores reales
p_pos, p_neg = np.sort(p[y == 1]), np.sort(p[y == 0])       # ordenadas: permite contar rápido


def matriz(t):
    """Matriz de confusión al umbral t (se llama si proba >= t). Devuelve TN, FP, FN, TP."""
    tp = P - np.searchsorted(p_pos, t, side="left")
    fp = N - np.searchsorted(p_neg, t, side="left")
    return N - fp, fp, P - tp, tp


def costo(t, c_fp, c_fn):
    _, fp, fn, _ = matriz(t)
    return fp * c_fp + fn * c_fn


# ---------------------------------------------------------------------------
# 2) Panel de control
# ---------------------------------------------------------------------------
ESCENARIOS = {"Caso base": (5.0, 50.0),
              "Errores iguales": (10.0, 10.0),
              "Venta muy valiosa": (5.0, 100.0)}
for k, v in {"c_fp": 5.0, "c_fn": 50.0}.items():
    st.session_state.setdefault(k, v)


def aplicar(nombre):
    st.session_state.c_fp, st.session_state.c_fn = ESCENARIOS[nombre]


with st.sidebar:
    st.markdown("## ⚙️ Panel de control")

    st.markdown("**Escenarios rápidos**")
    for nombre, (a, b) in ESCENARIOS.items():
        st.button(f"{nombre}  (FP {num(a)} · FN {num(b)})", on_click=aplicar, args=(nombre,),
                  use_container_width=True)

    st.markdown("**1. Costo de cada error (US$)**")
    c_fp = st.slider("📞 Llamada inútil (FP): llamar a quien NO compra", 1.0, 50.0, step=0.5, key="c_fp",
                     help="Costo de una llamada de call center (fuente: US$ 2,70–5,60).")
    c_fn = st.slider("💸 Venta perdida (FN): NO llamar a quien sí compraría", 5.0, 500.0, step=5.0, key="c_fn",
                     help="Margen perdido de la venta (comisión ~10–12 % de la prima).")

    p_star = c_fp / (c_fp + c_fn)                                # umbral teórico (Elkan, 2001)
    st.markdown(f"""<div class="card" style="--c:{TEAL}"><div class="t">🎯 Umbral óptimo p*</div>
                <div class="v">{num(p_star, 3)}</div>
                <div class="s">p* = C_FP / (C_FP + C_FN) = {num(c_fp, 1)} / ({num(c_fp, 1)} + {num(c_fn, 1)})
                · razón FN/FP = {num(c_fn / c_fp, 1)}</div></div>""", unsafe_allow_html=True)

    st.markdown("**2. Umbral de decisión**")
    modo = st.radio("¿Qué umbral evaluar?", ["Óptimo p*", "Manual"], horizontal=True)
    t_manual = st.slider("Umbral manual", 0.01, 0.99, 0.50, 0.01, disabled=(modo != "Manual"))
    umbral = p_star if modo == "Óptimo p*" else t_manual
    st.caption("Regla: se llama al cliente si su probabilidad de compra ≥ umbral.")

# ---------------------------------------------------------------------------
# 3) Cálculos
# ---------------------------------------------------------------------------
tn, fp, fn, tp = matriz(umbral)
costo_sel, costo_05 = fp * c_fp + fn * c_fn, costo(0.5, c_fp, c_fn)
recall, precision = tp / P, (tp / (tp + fp) if (tp + fp) else 0.0)
ahorro = costo_05 - costo_sel
grilla = np.linspace(0.01, 0.99, 197)
costos = np.array([costo(t, c_fp, c_fn) for t in grilla])

# ---------------------------------------------------------------------------
# 4) Encabezado, indicadores y mensaje principal
# ---------------------------------------------------------------------------
st.markdown(f"""<div class="banner"><h1>🚗 ¿A qué clientes llamar para vender el seguro de vehículo?</h1>
<p>Modelo Random Forest · {num(P + N)} clientes de prueba · {pct(P / (P + N))} se interesa ·
Mueve los costos y el umbral en el panel de la izquierda.</p></div>""", unsafe_allow_html=True)

k = st.columns(5)
tarjeta(k[0], "🎯", "Umbral evaluado", num(umbral, 3), f"Óptimo p* = {num(p_star, 3)}", TEAL)
tarjeta(k[1], "🔎", "Recall: compradores detectados", pct(recall), f"{num(tp)} de {num(P)}", NAVY)
tarjeta(k[2], "📞", "Precisión: llamadas con venta", pct(precision), f"{num(tp + fp)} llamadas", NAVY)
tarjeta(k[3], "💰", "Costo total", usd(costo_sel), f"Umbral 0,5: {usd(costo_05)}", ROJO)
tarjeta(k[4], "✅", "Ahorro vs umbral 0,5", usd(ahorro),
        pct(ahorro / costo_05) + " menos" if costo_05 else "", VERDE if ahorro >= 0 else ROJO)

if ahorro > 0:
    st.markdown(f'<div class="msg" style="background:#E8F5EE;border-left:6px solid {VERDE}">'
                f'💡 Con este umbral la empresa <b>ahorra {usd(ahorro)}</b> ({pct(ahorro / costo_05)}) '
                f'frente a la regla tradicional de 0,5, y detecta al <b>{pct(recall)}</b> de los compradores.</div>',
                unsafe_allow_html=True)
elif ahorro == 0:
    st.markdown(f'<div class="msg" style="background:#EEF2F7;border-left:6px solid {NAVY}">'
                f'⚖️ Con estos costos el umbral óptimo coincide con 0,5: no hay ahorro. '
                f'El 0,5 solo es óptimo cuando ambos errores cuestan lo mismo.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="msg" style="background:#FBECEC;border-left:6px solid {ROJO}">'
                f'⚠️ Este umbral <b>cuesta {usd(-ahorro)} más</b> que el umbral 0,5. '
                f'Prueba con el óptimo p* = {num(p_star, 3)}.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📉 Costo vs umbral", "🧮 Matriz de confusión",
                                        "📊 ¿En qué se va la plata?", "📈 Curva ROC", "📘 Cómo leerlo"])
LAYOUT = dict(plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY), separators=",.",
              margin=dict(t=40, b=10, l=10, r=10))

# ---------------------------------------------------------------------------
# 5) Curva de costo total vs umbral
# ---------------------------------------------------------------------------
with tab1:
    fig = go.Figure()
    lo, hi = sorted([umbral, 0.5])
    fig.add_vrect(x0=lo, x1=hi, fillcolor=VERDE if ahorro > 0 else ROJO, opacity=0.08, line_width=0,
                  annotation_text=("Zona de ahorro" if ahorro > 0 else ""), annotation_position="top left")
    fig.add_trace(go.Scatter(x=grilla, y=costos, mode="lines", name="Costo total",
                             line=dict(color=NAVY, width=3),
                             hovertemplate="Umbral %{x:.2f}<br>Costo US$ %{y:,.0f}<extra></extra>"))
    for t, color, nombre, dash in [(0.5, GRIS, "Umbral 0,5", "dash"),
                                   (p_star, TEAL, f"Óptimo p* = {num(p_star, 3)}", "solid")]:
        fig.add_vline(x=t, line=dict(color=color, dash=dash, width=2))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=nombre,
                                 line=dict(color=color, dash=dash, width=2)))
    fig.add_trace(go.Scatter(x=[0.5], y=[costo_05], mode="markers", name="Costo con 0,5",
                             marker=dict(color=GRIS, size=11), hovertemplate="US$ %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[umbral], y=[costo_sel], mode="markers", name=f"Umbral evaluado {num(umbral, 3)}",
                             marker=dict(color=ROJO, size=14, symbol="diamond"),
                             hovertemplate="US$ %{y:,.0f}<extra></extra>"))
    fig.update_layout(height=430, xaxis_title="Umbral de decisión", yaxis_title="Costo total (US$)",
                      yaxis_tickformat=",", legend=dict(orientation="h", y=-0.22), **LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor="#EEF1F5"); fig.update_yaxes(gridcolor="#EEF1F5", rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.info(f"**Umbral 0,5:** {usd(costo_05)}")
    c2.success(f"**Óptimo p\\* = {num(p_star, 3)}:** {usd(costo(p_star, c_fp, c_fn))}")
    c3.warning(f"**Mínimo observado en test** (umbral {num(grilla[int(np.argmin(costos))], 2)}): "
               f"{usd(costos.min())} · *solo referencia, se calcula mirando el test*")

# ---------------------------------------------------------------------------
# 6) Matriz de confusión con el costo en dinero de cada error
# ---------------------------------------------------------------------------
with tab2:
    texto = [[f"<b>{num(tn)}</b><br>✅ No se llamó y no compraba<br>US$ 0",
              f"<b>{num(fp)}</b><br>📞 Llamada inútil (FP)<br>{usd(fp * c_fp)}"],
             [f"<b>{num(fn)}</b><br>💸 Venta perdida (FN)<br>{usd(fn * c_fn)}",
              f"<b>{num(tp)}</b><br>✅ Venta lograda<br>US$ 0"]]
    fig = go.Figure(go.Heatmap(z=[[0, fp * c_fp], [fn * c_fn, 0]], text=texto, texttemplate="%{text}",
                               textfont=dict(size=16), x=["Predicho: NO llamar", "Predicho: LLAMAR"],
                               y=["Real: no compra", "Real: compra"],
                               colorscale=[[0, "#F7F9FB"], [1, ROJO]], showscale=False, hoverinfo="skip",
                               xgap=4, ygap=4))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=430, title=f"Umbral {num(umbral, 3)} · Costo total {usd(costo_sel)}", **LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Más rojo = más dinero perdido. Los aciertos (✅) no cuestan nada.")

# ---------------------------------------------------------------------------
# 7) Desglose del costo por estrategia
# ---------------------------------------------------------------------------
with tab3:
    filas = [("No llamar a nadie", 0, 0, P, 0), ("Llamar a todos", 0, N, 0, P),
             ("Umbral 0,5", *matriz(0.5)), (f"Óptimo p* ({num(p_star, 3)})", *matriz(p_star))]
    if p_pesos is not None:                                       # modelo con pesos (Etapa B), umbral 0,5
        pr = p_pesos >= 0.5
        filas.append(("Modelo con pesos (Etapa B, 0,5)", int((~pr & (y == 0)).sum()), int((pr & (y == 0)).sum()),
                      int((~pr & (y == 1)).sum()), int((pr & (y == 1)).sum())))
    comp = pd.DataFrame(filas, columns=["Estrategia", "TN", "FP", "FN", "TP"])
    comp["Costo FP"], comp["Costo FN"] = comp["FP"] * c_fp, comp["FN"] * c_fn
    comp["Costo total"] = comp["Costo FP"] + comp["Costo FN"]
    comp["Ahorro vs 0,5"] = costo_05 - comp["Costo total"]
    comp = comp.sort_values("Costo total", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(y=comp["Estrategia"], x=comp["Costo FP"], orientation="h", name="📞 Llamadas inútiles (FP)",
                         marker_color=TEAL, hovertemplate="US$ %{x:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(y=comp["Estrategia"], x=comp["Costo FN"], orientation="h", name="💸 Ventas perdidas (FN)",
                         marker_color=ROJO, text=[usd(v) for v in comp["Costo total"]], textposition="outside",
                         hovertemplate="US$ %{x:,.0f}<extra></extra>"))
    fig.update_layout(barmode="stack", height=400, xaxis_title="Costo total (US$)", xaxis_tickformat=",",
                      legend=dict(orientation="h", y=-0.25), **LAYOUT)
    fig.update_xaxes(range=[0, comp["Costo total"].max() * 1.22], gridcolor="#EEF1F5")
    st.plotly_chart(fig, use_container_width=True)

    tabla = comp[["Estrategia", "TN", "FP", "FN", "TP", "Costo total", "Ahorro vs 0,5"]].copy()
    for c in ["TN", "FP", "FN", "TP"]:
        tabla[c] = tabla[c].map(num)
    for c in ["Costo total", "Ahorro vs 0,5"]:
        tabla[c] = tabla[c].map(usd)
    st.dataframe(tabla, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# 8) (Opcional) Curva ROC con recta de iso-costo
# ---------------------------------------------------------------------------
with tab4:
    ts = np.r_[0.0, np.linspace(0.001, 0.999, 999), 1.01]
    tpr = np.array([matriz(t)[3] / P for t in ts])
    fpr = np.array([matriz(t)[1] / N for t in ts])
    izq, der = np.searchsorted(p_neg, p_pos, "left"), np.searchsorted(p_neg, p_pos, "right")
    auc = (izq + 0.5 * (der - izq)).sum() / (P * N)             # AUC exacta (pares bien ordenados)
    m = (c_fp * N) / (c_fn * P)                                  # pendiente de la recta de iso-costo
    fpr_s, tpr_s = fp / N, tp / P
    xs = np.linspace(0, 1, 50)
    t05 = matriz(0.5)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Random Forest (AUC = {num(auc, 3)})",
                             line=dict(color=NAVY, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(color=GRIS, dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=tpr_s + m * (xs - fpr_s), mode="lines",
                             name=f"Iso-costo (pendiente {num(m, 2)})", line=dict(color="black", dash="dot")))
    fig.add_trace(go.Scatter(x=[t05[1] / N], y=[t05[3] / P], mode="markers", name="Umbral 0,5",
                             marker=dict(color=GRIS, size=12)))
    fig.add_trace(go.Scatter(x=[fpr_s], y=[tpr_s], mode="markers", name=f"Umbral evaluado {num(umbral, 3)}",
                             marker=dict(color=ROJO, size=16, symbol="star")))
    fig.update_layout(height=480, xaxis_title="FPR (llamadas inútiles / no compradores)",
                      yaxis_title="TPR = Recall (compradores detectados)", xaxis_range=[0, 1], yaxis_range=[0, 1.02],
                      legend=dict(x=0.52, y=0.05), **LAYOUT)
    fig.update_xaxes(gridcolor="#EEF1F5"); fig.update_yaxes(gridcolor="#EEF1F5")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cada punto de la curva es un umbral distinto del mismo modelo. El mejor umbral es donde la "
               "recta de iso-costo toca la curva más arriba y a la izquierda.")

# ---------------------------------------------------------------------------
# 9) Guía para la gerencia
# ---------------------------------------------------------------------------
with tab5:
    st.markdown(f"""
**¿Qué hace el modelo?** Estima, para cada cliente, la probabilidad de que compre el seguro de vehículo.

**¿Qué decide el umbral?** Se llama a todo cliente cuya probabilidad sea mayor o igual al umbral.

**Dos errores con distinto precio**
- 📞 **Llamada inútil (FP):** llamamos y no compra → perdemos el costo de la llamada ({usd(c_fp)}).
- 💸 **Venta perdida (FN):** no llamamos y sí compraba → perdemos la venta ({usd(c_fn)}).

**El umbral óptimo** compara el costo esperado de llamar, (1 − p)·C_FP, con el de no llamar, p·C_FN:
conviene llamar si **p ≥ C_FP / (C_FP + C_FN) = {num(p_star, 3)}**.

**¿Por qué no 0,5?** El 0,5 solo es óptimo si ambos errores cuestan lo mismo (prueba el escenario
*"Errores iguales"*). Aquí perder una venta cuesta {num(c_fn / c_fp, 1)} veces más que una llamada inútil,
así que conviene llamar a más gente.

**Solo importa la razón FN/FP:** si duplicas ambos costos, el umbral no cambia; solo cambian los montos.
""")
    st.caption("Grupo 15 · Certamen 1 Machine Learning · Base: Health Insurance Cross Sell Prediction (Kaggle). "
               "Costos supuestos apoyados en fuentes; probabilidades del modelo reentrenado en proporción real.")

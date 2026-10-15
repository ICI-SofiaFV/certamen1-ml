# =============================================================================
# Dashboard · Certamen 1 · Grupo 15
# ¿A qué clientes llamar? Costo de los errores y umbral óptimo
# Datos: probabilidades en escala real del Random Forest sobre el test (76.168 clientes)
# =============================================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Umbral óptimo · Venta cruzada de seguros",
                   page_icon="🚗", layout="wide")

AZUL, ROJO, VERDE, GRIS, NARANJA = "#1F3A5F", "#D1495B", "#2A9D8F", "#8A94A6", "#E08E45"


# ---------------------------------------------------------------------------
# 1) Datos: y_test (real) + proba (probabilidad de compra en escala real)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("predicciones_test.csv")
    y = df["y_test"].to_numpy()
    p = df["proba"].to_numpy()
    p_pesos = df["proba_pesos_etapa_B"].to_numpy() if "proba_pesos_etapa_B" in df else None
    return y, p, p_pesos


y, p, p_pesos = cargar_datos()
P, N = int((y == 1).sum()), int((y == 0).sum())           # compradores y no compradores reales
p_pos, p_neg = np.sort(p[y == 1]), np.sort(p[y == 0])      # ordenadas: permite contar rápido


def matriz(t):
    """Matriz de confusión al umbral t: se llama al cliente si proba >= t."""
    tp = P - np.searchsorted(p_pos, t, side="left")         # compradores llamados
    fp = N - np.searchsorted(p_neg, t, side="left")         # no compradores llamados
    return N - fp, fp, P - tp, tp                           # TN, FP, FN, TP


def costo(t, c_fp, c_fn):
    tn, fp, fn, tp = matriz(t)
    return fp * c_fp + fn * c_fn


# ---------------------------------------------------------------------------
# 2) Panel de control (barra lateral)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Panel de control")

    st.subheader("1. Costo de cada error (US$)")
    c_fp = st.slider("Falso positivo: llamar a quien NO compra", 1.0, 50.0, 5.0, 0.5,
                     help="Costo de una llamada de call center (fuente: US$ 2,70–5,60).")
    c_fn = st.slider("Falso negativo: NO llamar a quien sí compraría", 5.0, 500.0, 50.0, 5.0,
                     help="Margen perdido de la venta (comisión ~10–12% de la prima).")

    p_star = c_fp / (c_fp + c_fn)                            # umbral teórico (Elkan, 2001)
    st.metric("Umbral óptimo p*", f"{p_star:.3f}",
              help="p* = C_FP / (C_FP + C_FN) (Elkan, 2001)")
    st.caption(f"p* = {c_fp:g} / ({c_fp:g} + {c_fn:g}) · razón FN/FP = {c_fn / c_fp:.1f}")

    st.subheader("2. Umbral de decisión")
    modo = st.radio("¿Qué umbral evaluar?", ["Óptimo p*", "Manual"], horizontal=True)
    t_manual = st.slider("Umbral manual", 0.01, 0.99, 0.50, 0.01, disabled=(modo != "Manual"))
    umbral = p_star if modo == "Óptimo p*" else t_manual

    st.caption("Regla: se llama al cliente si su probabilidad de compra ≥ umbral.")

# ---------------------------------------------------------------------------
# 3) Cálculos para el umbral elegido
# ---------------------------------------------------------------------------
tn, fp, fn, tp = matriz(umbral)
costo_sel = fp * c_fp + fn * c_fn
costo_05 = costo(0.5, c_fp, c_fn)
recall = tp / P
precision = tp / (tp + fp) if (tp + fp) else 0.0
ahorro = costo_05 - costo_sel

grilla = np.linspace(0.01, 0.99, 197)
costos = np.array([costo(t, c_fp, c_fn) for t in grilla])
t_min = grilla[int(np.argmin(costos))]                       # mínimo de la curva (test)

# ---------------------------------------------------------------------------
# 4) Encabezado e indicadores
# ---------------------------------------------------------------------------
st.title("🚗 ¿A qué clientes llamar para vender el seguro de vehículo?")
st.markdown(f"Modelo **Random Forest** · {P + N:,} clientes de test · "
            f"{P / (P + N):.1%} se interesa. Mueve los costos y el umbral en el panel de la izquierda.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Umbral evaluado", f"{umbral:.3f}", help=f"Umbral óptimo p* = {p_star:.3f}")
k2.metric("Recall (compradores detectados)", f"{recall:.1%}")
k3.metric("Precisión (llamadas con venta)", f"{precision:.1%}")
k4.metric("Costo total", f"US$ {costo_sel:,.0f}")
k5.metric("Ahorro vs umbral 0,5", f"US$ {ahorro:,.0f}",
          f"{ahorro / costo_05:.1%}" if costo_05 else None)

tab1, tab2, tab3, tab4 = st.tabs(["📉 Costo vs umbral", "🧮 Matriz de confusión",
                                  "📈 Curva ROC", "📘 Cómo leerlo"])

# ---------------------------------------------------------------------------
# 5) Curva de costo total vs umbral
# ---------------------------------------------------------------------------
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grilla, y=costos, mode="lines", name="Costo total",
                             line=dict(color=AZUL, width=3),
                             hovertemplate="Umbral %{x:.2f}<br>Costo US$ %{y:,.0f}<extra></extra>"))
    for t, color, nombre, dash in [(0.5, GRIS, "Umbral 0,5", "dash"),
                                   (p_star, VERDE, f"Óptimo p* = {p_star:.3f}", "solid"),
                                   (umbral, ROJO, f"Evaluado = {umbral:.3f}", "dot")]:
        fig.add_trace(go.Scatter(x=[t, t], y=[0, costos.max() * 1.05], mode="lines", name=nombre,
                                 line=dict(color=color, dash=dash, width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[umbral], y=[costo_sel], mode="markers", name="Costo del umbral evaluado",
                             marker=dict(color=ROJO, size=12),
                             hovertemplate="US$ %{y:,.0f}<extra></extra>"))
    fig.update_layout(height=430, xaxis_title="Umbral de decisión", yaxis_title="Costo total (US$)",
                      yaxis_tickformat=",", legend=dict(orientation="h", y=-0.2),
                      margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.info(f"**Umbral 0,5:** US$ {costo_05:,.0f}")
    c2.success(f"**Óptimo p\\* = {p_star:.3f}:** US$ {costo(p_star, c_fp, c_fn):,.0f}")
    c3.warning(f"**Mínimo de la curva** (umbral {t_min:.2f}): US$ {costos.min():,.0f}")

    # Comparación de estrategias con los costos actuales
    filas = [("No llamar a nadie", 0, 0, P, 0),
             ("Llamar a todos", 0, N, 0, P)]
    for nombre, t in [("Umbral 0,5", 0.5), (f"Óptimo p* ({p_star:.3f})", p_star)]:
        a, b, c, d = matriz(t)
        filas.append((nombre, a, b, c, d))
    if p_pesos is not None:                                  # modelo con pesos (Etapa B), umbral 0,5
        pred = p_pesos >= 0.5
        filas.append(("Modelo con pesos (Etapa B, 0,5)", int(((~pred) & (y == 0)).sum()),
                      int((pred & (y == 0)).sum()), int(((~pred) & (y == 1)).sum()),
                      int((pred & (y == 1)).sum())))
    comp = pd.DataFrame(filas, columns=["Estrategia", "TN", "FP", "FN", "TP"])
    comp["Costo total (US$)"] = comp["FP"] * c_fp + comp["FN"] * c_fn
    comp["Ahorro vs 0,5 (US$)"] = costo_05 - comp["Costo total (US$)"]
    st.subheader("Comparación de estrategias")
    st.dataframe(comp.style.format({c: "{:,.0f}" for c in comp.columns[1:]})
                 .highlight_min(subset=["Costo total (US$)"], color="#C8E6C9"),
                 hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# 6) Matriz de confusión con el costo en dinero de cada error
# ---------------------------------------------------------------------------
with tab2:
    texto = [[f"<b>{tn:,}</b><br>Bien: no se llamó<br>US$ 0",
              f"<b>{fp:,}</b><br>Llamada inútil (FP)<br>US$ {fp * c_fp:,.0f}"],
             [f"<b>{fn:,}</b><br>Venta perdida (FN)<br>US$ {fn * c_fn:,.0f}",
              f"<b>{tp:,}</b><br>Bien: venta lograda<br>US$ 0"]]
    fig = go.Figure(go.Heatmap(z=[[0, fp * c_fp], [fn * c_fn, 0]], text=texto,
                               texttemplate="%{text}", textfont=dict(size=16),
                               x=["Predicho: NO llamar", "Predicho: LLAMAR"],
                               y=["Real: no compra", "Real: compra"],
                               colorscale=[[0, "#F5F7FA"], [1, ROJO]], showscale=False,
                               hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=430, margin=dict(t=30, b=10),
                      title=f"Umbral {umbral:.3f} · Costo total US$ {costo_sel:,.0f}")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Gasto en llamadas inútiles (FP × C_FP)", f"US$ {fp * c_fp:,.0f}")
    c1.caption(f"{fp:,} llamadas × US$ {c_fp:g}")
    c2.metric("Ventas perdidas (FN × C_FN)", f"US$ {fn * c_fn:,.0f}")
    c2.caption(f"{fn:,} clientes × US$ {c_fn:g}")

# ---------------------------------------------------------------------------
# 7) (Opcional) Curva ROC con recta de iso-costo
# ---------------------------------------------------------------------------
with tab3:
    ts = np.r_[0.0, np.linspace(0.001, 0.999, 999), 1.01]
    tpr = np.array([matriz(t)[3] / P for t in ts])
    fpr = np.array([matriz(t)[1] / N for t in ts])
    # AUC exacta: proporción de pares (comprador, no comprador) bien ordenados
    izq, der = np.searchsorted(p_neg, p_pos, "left"), np.searchsorted(p_neg, p_pos, "right")
    auc = (izq + 0.5 * (der - izq)).sum() / (P * N)
    m = (c_fp * N) / (c_fn * P)                              # pendiente iso-costo
    fpr_s, tpr_s = matriz(umbral)[1] / N, tp / P
    xs = np.linspace(0, 1, 50)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Random Forest (AUC = {auc:.3f})",
                             line=dict(color=AZUL, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                             line=dict(color=GRIS, dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=tpr_s + m * (xs - fpr_s), mode="lines",
                             name=f"Iso-costo (pendiente {m:.2f})", line=dict(color="black", dash="dot")))
    t05 = matriz(0.5)
    fig.add_trace(go.Scatter(x=[t05[1] / N], y=[t05[3] / P], mode="markers", name="Umbral 0,5",
                             marker=dict(color=GRIS, size=12)))
    fig.add_trace(go.Scatter(x=[fpr_s], y=[tpr_s], mode="markers", name=f"Umbral evaluado {umbral:.3f}",
                             marker=dict(color=ROJO, size=16, symbol="star")))
    fig.update_layout(height=480, xaxis_title="FPR (llamadas inútiles / no compradores)",
                      yaxis_title="TPR = Recall (compradores detectados)",
                      xaxis_range=[0, 1], yaxis_range=[0, 1.02], margin=dict(t=30, b=10),
                      legend=dict(x=0.55, y=0.05))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cada punto de la curva es un umbral distinto. El mejor umbral es donde la recta "
               "de iso-costo toca la curva más arriba y a la izquierda.")

# ---------------------------------------------------------------------------
# 8) Guía para la gerencia
# ---------------------------------------------------------------------------
with tab4:
    st.markdown(f"""
**¿Qué hace el modelo?** Estima, para cada cliente, la probabilidad de que compre el seguro de vehículo.

**¿Qué decide el umbral?** Se llama a todo cliente cuya probabilidad sea mayor o igual al umbral.

**Dos errores con distinto precio**
- **Falso positivo (FP):** llamamos y no compra → perdemos el costo de la llamada (US$ {c_fp:,.0f}).
- **Falso negativo (FN):** no llamamos y sí compraba → perdemos la venta (US$ {c_fn:,.0f}).

**El umbral óptimo** compara el costo esperado de llamar, (1 − p)·C_FP, con el de no llamar, p·C_FN:
conviene llamar si **p ≥ C_FP / (C_FP + C_FN) = {p_star:.3f}**.

**¿Por qué no 0,5?** El 0,5 solo es óptimo si ambos errores cuestan lo mismo. Aquí perder una venta
cuesta {c_fn / c_fp:.0f} veces más que una llamada inútil, así que conviene llamar a más gente.

**Cómo usarlo:** mueve los costos en el panel y observa cómo se desplaza p* y cambia el ahorro.
Prueba el umbral manual para comparar cualquier regla con la óptima.
""")
    st.caption("Grupo 15 · Certamen 1 Machine Learning · Base: Health Insurance Cross Sell Prediction (Kaggle). "
               "Costos supuestos apoyados en fuentes; probabilidades del modelo reentrenado en proporción real.")

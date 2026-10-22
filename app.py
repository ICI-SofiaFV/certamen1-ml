# =============================================================================
# Dashboard · Certamen 1 · Grupo 15
# ¿A qué clientes llamar? Costo de los errores y umbral óptimo
# Datos: probabilidades en escala real del Random Forest sobre el test
# =============================================================================
import base64

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Umbral óptimo · Seguro de vehículo", page_icon="🚗", layout="wide")

# --- Paleta (la misma de la presentación en Canva) ---
NAVY = "#1E2E4E"      # títulos y texto principal
NAVY_OSC = "#14213D"  # fondo del banner y del panel lateral
AZUL = "#2E3F9E"      # llamar / falso positivo (lo barato)
NARANJO = "#E8893A"   # umbral óptimo p* y acentos
ROJO = "#C8414B"      # venta perdida / falso negativo (lo caro)
VERDE = "#2E8B57"     # ahorro
GRIS = "#8A94A6"      # referencias (umbral 0,5, azar)
CREMA = "#F6F1E4"     # fondos suaves
GRILLA = "#ECE7DA"    # líneas de los gráficos
FUENTE = "Open Sans, sans-serif"


# --- Formato chileno: punto para miles, coma para decimales ---
def usd(x):
    return "US$ " + f"{x:,.0f}".replace(",", ".")


def num(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, dec=1):
    return num(100 * x, dec) + " %"


# =============================================================================
# Ilustraciones propias en SVG (auto genérico, sin marcas) e íconos de línea
# =============================================================================
def img(svg, alto, extra=""):
    """Convierte un SVG en <img> (así Streamlit lo muestra sin problemas)."""
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" style="height:{alto}px;{extra}">'


def auto(color=NARANJO, x=0, y=0, s=1.0):
    """Auto sedán de perfil, mirando a la derecha (suelo en y=184)."""
    return f"""
<g transform="translate({x},{y}) scale({s})">
  <ellipse cx="305" cy="212" rx="215" ry="7" fill="#000" opacity=".28"/>
  <path d="M110 184 L110 160 Q112 144 134 140 L196 132 Q222 100 262 92 L350 88 Q392 88 420 114 L446 132
           Q490 136 502 150 Q508 158 506 184 Z" fill="{color}"/>
  <path d="M134 140 L196 132 Q222 100 262 92 L350 88 Q392 88 420 114 L446 132 Q470 134 486 140 Z"
        fill="#fff" opacity=".10"/>
  <path d="M212 132 Q234 106 266 100 L300 98 L300 132 Z" fill="#CFE2F3"/>
  <path d="M312 98 L350 97 Q384 98 404 122 L410 132 L312 132 Z" fill="#CFE2F3"/>
  <path d="M222 128 Q238 110 262 104" stroke="#fff" stroke-width="4" opacity=".6" fill="none" stroke-linecap="round"/>
  <line x1="306" y1="134" x2="306" y2="182" stroke="#000" opacity=".15" stroke-width="2"/>
  <rect x="268" y="146" width="18" height="4" rx="2" fill="#000" opacity=".25"/>
  <rect x="378" y="146" width="18" height="4" rx="2" fill="#000" opacity=".25"/>
  <path d="M488 142 Q500 146 504 156 L490 156 Z" fill="#FFF3C4"/>
  <rect x="110" y="148" width="8" height="14" rx="2" fill="{ROJO}"/>
  <circle cx="190" cy="184" r="31" fill="{NAVY_OSC}"/><circle cx="420" cy="184" r="31" fill="{NAVY_OSC}"/>
  <circle cx="190" cy="184" r="26" fill="#1B1B1F"/><circle cx="420" cy="184" r="26" fill="#1B1B1F"/>
  <circle cx="190" cy="184" r="14" fill="#C9CED8"/><circle cx="420" cy="184" r="14" fill="#C9CED8"/>
  <circle cx="190" cy="184" r="5" fill="{GRIS}"/><circle cx="420" cy="184" r="5" fill="{GRIS}"/>
</g>"""


ESCENA_HERO = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 240">
  <circle cx="470" cy="70" r="46" fill="{NARANJO}" opacity=".85"/>
  <g fill="#fff" opacity=".07">
    <rect x="20" y="90" width="44" height="100"/><rect x="70" y="60" width="36" height="130"/>
    <rect x="112" y="110" width="50" height="80"/><rect x="170" y="40" width="40" height="150"/>
    <rect x="216" y="96" width="56" height="94"/><rect x="280" y="70" width="34" height="120"/>
    <rect x="320" y="120" width="60" height="70"/><rect x="388" y="84" width="40" height="106"/>
    <rect x="436" y="128" width="54" height="62"/><rect x="496" y="100" width="44" height="90"/>
  </g>
  <rect x="0" y="190" width="560" height="50" fill="#0D182E"/>
  <g stroke="#fff" stroke-width="4" opacity=".5" stroke-dasharray="26 22"><line x1="0" y1="226" x2="560" y2="226"/></g>
  <polygon points="506,150 560,128 560,178" fill="#FFF3C4" opacity=".16"/>
  <g stroke="#fff" stroke-width="5" stroke-linecap="round" opacity=".35">
    <line x1="22" y1="140" x2="88" y2="140"/><line x1="42" y1="160" x2="96" y2="160"/>
    <line x1="14" y1="178" x2="80" y2="178"/>
  </g>
  {auto(NARANJO)}
</svg>"""

LOGO = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="100 80 420 140">{auto(NARANJO)}</svg>"""

PIE = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 70" preserveAspectRatio="xMidYMax slice">
  <rect x="0" y="52" width="1200" height="18" fill="{NAVY_OSC}"/>
  <g stroke="#fff" stroke-width="2" opacity=".5" stroke-dasharray="18 14"><line x1="0" y1="61" x2="1200" y2="61"/></g>
  {auto(AZUL, 120, -2, .25)}{auto(NARANJO, 560, -2, .25)}{auto(GRIS, 930, -2, .25)}
</svg>"""

# Íconos de línea (trazos simples, estilo "outline")
ICONOS = {
    "objetivo": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "buscar": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "telefono": '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 '
                '19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 '
                '2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 '
                '2.81.7A2 2 0 0 1 22 16.92z"/>',
    "dinero": '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "ahorro": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "alerta": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
              '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "grafico": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>'
               '<line x1="6" y1="20" x2="6" y2="14"/>',
    "balanza": '<line x1="12" y1="3" x2="12" y2="21"/><path d="M5 7h14"/><path d="M5 7l-3 7a4 4 0 0 0 6 0z"/>'
               '<path d="M19 7l-3 7a4 4 0 0 0 6 0z"/><line x1="8" y1="21" x2="16" y2="21"/>',
}


def icono(nombre, color, tam=26):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONOS[nombre]}</svg>')
    return img(svg, tam)


# =============================================================================
# Estilos
# =============================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&family=Open+Sans:wght@400;600;700&display=swap');
.block-container {{padding-top: 2.2rem; padding-bottom: 1rem; max-width: 1450px;}}
h1, h2, h3, h4 {{font-family: 'Montserrat', sans-serif !important; letter-spacing: -.01em;}}
#MainMenu, footer {{visibility: hidden;}}

/* Banner principal */
.hero {{background: linear-gradient(120deg, {NAVY_OSC} 0%, {NAVY} 55%, {AZUL} 135%); border-radius: 22px;
        padding: 1.8rem 2.2rem; display: flex; align-items: center; gap: 1.5rem; overflow: hidden;
        box-shadow: 0 10px 30px rgba(20,33,61,.25); margin-bottom: 1.3rem;}}
.hero .txt {{flex: 1.25; min-width: 0;}}
.hero .ilus {{flex: 1; min-width: 0; text-align: right;}}
.hero .ilus img {{width: 100%; height: auto !important; max-height: 230px;}}
.hero .tag {{display: inline-block; background: rgba(232,137,58,.18); color: #FFC48F; border: 1px solid rgba(232,137,58,.5);
             border-radius: 999px; padding: .25rem .9rem; font-size: .85rem; font-weight: 700; letter-spacing: .06em;
             text-transform: uppercase;}}
.hero .h1 {{font-family: 'Montserrat', sans-serif; color: white; font-size: 2.5rem; font-weight: 800; line-height: 1.15; margin: .7rem 0 .6rem 0;}}
.hero .h1 em {{color: {NARANJO}; font-style: normal;}}
.hero p {{color: #D9DEEA; font-size: 1.12rem; margin: 0 0 1.1rem 0; line-height: 1.5;}}
.chips {{display: flex; flex-wrap: wrap; gap: .6rem;}}
.chip {{background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); border-radius: 12px;
        padding: .5rem .9rem; color: white; font-size: .95rem;}}
.chip b {{font-family: 'Montserrat', sans-serif; color: #FFC48F; font-size: 1.1rem; margin-right: .25rem;}}

/* Títulos de sección */
.seccion {{display: flex; align-items: center; gap: .6rem; margin: 1.2rem 0 .7rem 0;}}
.seccion h3 {{margin: 0; padding: 0; font-size: 1.45rem; color: {NAVY};}}
.seccion .linea {{flex: 1; height: 2px; background: linear-gradient(90deg, {NARANJO}, transparent);}}

/* Tarjetas de indicadores */
.card {{background: white; border-radius: 18px; padding: 1.1rem 1.2rem 1rem 1.2rem; min-height: 0; overflow: hidden;
        box-shadow: 0 4px 16px rgba(20,33,61,.08); border-top: 5px solid var(--c); position: relative;}}
.card .ic {{width: 46px; height: 46px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
            background: var(--bg); margin-bottom: .55rem;}}
.card .t {{font-size: 1.02rem; color: #4A5160; font-weight: 600; line-height: 1.3; white-space: nowrap;}}
.card .v {{font-family: 'Montserrat', sans-serif; font-size: clamp(1.3rem, 1.75vw, 1.9rem); font-weight: 800;
           line-height: 1.15; white-space: nowrap; color: var(--c); margin-top: .15rem;}}
.card .s {{font-size: .92rem; color: #5A6273; margin-top: .25rem; white-space: nowrap;}}

/* Mensaje principal */
.msg {{display: flex; align-items: center; gap: 1rem; border-radius: 18px; padding: 1.1rem 1.4rem; margin: 1.2rem 0 .4rem 0;
       font-size: 1.18rem; color: {NAVY}; line-height: 1.5;}}
.msg.ok {{background: #FBEBCF; border: 2px solid #F1C77A;}}
.msg.neutro {{background: #EEF1F5; border: 2px solid #D5DBE5;}}
.msg.mal {{background: #FBE4E4; border: 2px solid #E7A3A8;}}

/* Pestañas tipo "píldora" */
[role="tablist"] {{gap: .4rem; background: white; padding: .45rem; border-radius: 14px; border: none !important;
                  box-shadow: 0 2px 10px rgba(20,33,61,.06); flex-wrap: wrap;}}
[data-testid="stTab"] {{height: auto !important; padding: .6rem 1.15rem !important; border-radius: 10px; transition: background .15s;}}
[data-testid="stTab"] p {{font-size: 1.08rem !important; font-weight: 600; color: {NAVY};}}
[data-testid="stTab"]:hover {{background: {CREMA};}}
[data-testid="stTab"][aria-selected="true"] {{background: {NAVY};}}
[data-testid="stTab"][aria-selected="true"] p {{color: white !important;}}
[role="tablist"] .react-aria-SelectionIndicator {{display: none;}}

/* Contenedores con borde = tarjetas blancas */
[class*="st-key-caja"] {{background: white; border-radius: 18px !important; border: none !important;
                         box-shadow: 0 4px 16px rgba(20,33,61,.08); padding: 1rem 1.2rem !important;}}

.mini {{background: white; border-radius: 14px; padding: .9rem 1.1rem; font-size: 1.05rem; color: {NAVY};
        border-left: 5px solid var(--c); box-shadow: 0 2px 10px rgba(20,33,61,.06);}}
.mini b {{font-family: 'Montserrat', sans-serif; color: var(--c);}}
.mini .nota {{font-size: .88rem; color: #5A6273; margin-top: .15rem;}}

/* Guía */
.paso {{background: white; border-radius: 18px; padding: 1.2rem; min-height: 265px; box-shadow: 0 4px 16px rgba(20,33,61,.07);}}
.paso .ic2 {{width: 46px; height: 46px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: .6rem;}}
.paso .n {{font-family: 'Montserrat', sans-serif; font-weight: 800; color: {NARANJO}; font-size: .95rem; letter-spacing: .05em;}}
.paso h4 {{margin: .35rem 0 .3rem 0; padding: 0; font-size: 1.15rem; color: {NAVY};}}
.paso p {{margin: 0; font-size: 1rem; color: #3E4656; line-height: 1.5;}}

/* Panel lateral oscuro */
[data-testid="stSidebar"] .logo {{display: flex; flex-direction: column; align-items: flex-start; gap: .5rem; margin: -1.5rem 0 .6rem 0;}}
[data-testid="stSidebar"] .logo b {{font-family: 'Montserrat', sans-serif; font-size: 1.3rem; color: white; line-height: 1.1;}}
[data-testid="stSidebar"] .logo small {{display: block; color: #AEB7CC; font-size: .85rem; font-weight: 400;}}
.lat-tit {{font-family: 'Montserrat', sans-serif; font-weight: 700; color: #FFC48F; font-size: .85rem;
           letter-spacing: .08em; text-transform: uppercase; margin: 1.1rem 0 .4rem 0;}}
.side-card {{background: rgba(255,255,255,.06); border: 1px solid rgba(232,137,58,.55); border-radius: 16px;
             padding: 1rem 1.1rem; margin: .8rem 0 .5rem 0;}}
.side-card .t {{font-size: .92rem; color: #D9DEEA; font-weight: 600;}}
.side-card .v {{font-family: 'Montserrat', sans-serif; font-size: 2.6rem; font-weight: 800; color: {NARANJO}; line-height: 1.1;}}
.side-card .s {{font-size: .85rem; color: #AEB7CC; line-height: 1.45;}}
[data-testid="stSidebar"] .stButton button {{justify-content: flex-start; border-radius: 12px; padding: .55rem .9rem;}}
[data-testid="stSidebar"] .stButton button:hover {{border-color: {NARANJO}; color: {NARANJO};}}

.pie {{margin-top: 1.6rem;}}
.pie img {{width: 100%; height: 70px !important; display: block;}}
.pie p {{text-align: center; color: #6B7385; font-size: .9rem; margin: .6rem 0 0 0;}}

@media (max-width: 900px) {{ .hero {{flex-direction: column;}} .hero .h1 {{font-size: 1.9rem;}} }}
</style>
""", unsafe_allow_html=True)


def tarjeta(col, ic, titulo, valor, sub, color, fondo):
    col.markdown(f'<div class="card" style="--c:{color}; --bg:{fondo}"><div class="ic">{icono(ic, color)}</div>'
                 f'<div class="t">{titulo}</div><div class="v">{valor}</div><div class="s">{sub}</div></div>',
                 unsafe_allow_html=True)


def mini(col, texto, nota="", color=NAVY):
    col.markdown(f'<div class="mini" style="--c:{color}">{texto}<div class="nota">{nota}</div></div>',
                 unsafe_allow_html=True)


def seccion(titulo):
    st.markdown(f'<div class="seccion"><h3>{titulo}</h3><div class="linea"></div></div>', unsafe_allow_html=True)


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
izq, der = np.searchsorted(p_neg, p_pos, "left"), np.searchsorted(p_neg, p_pos, "right")
auc = (izq + 0.5 * (der - izq)).sum() / (P * N)             # AUC exacta (pares bien ordenados)


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
ESCENARIOS = {"Caso base": (5.0, 50.0, ":material/directions_car:"),
              "Errores iguales": (10.0, 10.0, ":material/balance:"),
              "Venta muy valiosa": (5.0, 100.0, ":material/trending_up:")}
for k, v in {"c_fp": 5.0, "c_fn": 50.0}.items():
    st.session_state.setdefault(k, v)


def aplicar(nombre):
    st.session_state.c_fp, st.session_state.c_fn = ESCENARIOS[nombre][:2]


with st.sidebar:
    st.markdown(f'<div class="logo">{img(LOGO, 30)}<b>Panel de control<small>Seguro de vehículo · Grupo 15</small></b></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="lat-tit">Escenarios rápidos</div>', unsafe_allow_html=True)
    for nombre, (a, b, ic) in ESCENARIOS.items():
        st.button(f"{nombre}  ({num(a)} / {num(b)})", on_click=aplicar, args=(nombre,), icon=ic,
                  use_container_width=True)
    st.caption("Entre paréntesis: costo de llamada inútil / venta perdida (US$).")

    st.markdown('<div class="lat-tit">1 · Costo de cada error (US$)</div>', unsafe_allow_html=True)
    c_fp = st.slider(":material/call: Llamada inútil (FP): llamar a quien NO compra", 1.0, 50.0, step=0.5,
                     key="c_fp", help="Costo de una llamada de call center (fuente: US$ 2,70–5,60).")
    c_fn = st.slider(":material/money_off: Venta perdida (FN): NO llamar a quien sí compraría", 5.0, 500.0,
                     step=5.0, key="c_fn", help="Margen perdido de la venta (comisión ~10–12 % de la prima).")

    p_star = c_fp / (c_fp + c_fn)                                # umbral teórico (Elkan, 2001)
    st.markdown(f"""<div class="side-card"><div class="t">Umbral óptimo p*</div>
                <div class="v">{num(p_star, 3)}</div>
                <div class="s">p* = C_FP / (C_FP + C_FN)<br>= {num(c_fp, 1)} / ({num(c_fp, 1)} + {num(c_fn, 1)})<br>
                Perder una venta cuesta <b style="color:#FFC48F">{num(c_fn / c_fp, 1)} veces</b> una llamada inútil</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="lat-tit">2 · Umbral de decisión</div>', unsafe_allow_html=True)
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
st.markdown(f"""
<div class="hero">
  <div class="txt">
    <div class="tag">Venta cruzada · Seguro de vehículo</div>
    <div class="h1">¿A qué clientes <em>llamar</em>?</div>
    <p>El modelo estima la probabilidad de que cada cliente compre el seguro. Elige los costos de cada error
       en el panel izquierdo y mira cuánto ahorra la empresa al usar el umbral óptimo en vez de 0,5.</p>
    <div class="chips">
      <div class="chip"><b>Random Forest</b> modelo</div>
      <div class="chip"><b>{num(P + N)}</b> clientes de prueba</div>
      <div class="chip"><b>{pct(P / (P + N))}</b> se interesa</div>
      <div class="chip"><b>{num(auc, 3)}</b> AUC</div>
    </div>
  </div>
  <div class="ilus">{img(ESCENA_HERO, 230)}</div>
</div>""", unsafe_allow_html=True)

seccion(f"Resultados con umbral {num(umbral, 3)}")
k = st.columns(5)
tarjeta(k[0], "objetivo", "Umbral evaluado", num(umbral, 3), f"Óptimo p* = {num(p_star, 3)}", NARANJO, "#FDF0E3")
tarjeta(k[1], "buscar", "Recall", pct(recall), f"{num(tp)} de {num(P)}", AZUL, "#E9ECF8")
tarjeta(k[2], "telefono", "Precisión", pct(precision), f"de {num(tp + fp)} llamadas", AZUL, "#E9ECF8")
tarjeta(k[3], "dinero", "Costo total", usd(costo_sel), f"Con 0,5: {usd(costo_05)}", ROJO, "#FAE8E9")
tarjeta(k[4], "ahorro", "Ahorro vs umbral 0,5", usd(ahorro),
        pct(ahorro / costo_05) + " menos" if costo_05 else "", VERDE if ahorro >= 0 else ROJO,
        "#E5F3EB" if ahorro >= 0 else "#FAE8E9")

if ahorro > 0:
    st.markdown(f'<div class="msg ok">{icono("ahorro", "#B9741F", 34)}<div>'
                f'Con este umbral la empresa <b>ahorra {usd(ahorro)}</b> ({pct(ahorro / costo_05)}) '
                f'frente a la regla tradicional de 0,5, y detecta al <b>{pct(recall)}</b> de los compradores.</div></div>',
                unsafe_allow_html=True)
elif ahorro == 0:
    st.markdown(f'<div class="msg neutro">{icono("balanza", NAVY, 34)}<div>'
                f'Con estos costos el umbral óptimo coincide con 0,5: no hay ahorro. '
                f'El 0,5 solo es óptimo cuando ambos errores cuestan lo mismo.</div></div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="msg mal">{icono("alerta", ROJO, 34)}<div>'
                f'Este umbral <b>cuesta {usd(-ahorro)} más</b> que el umbral 0,5. '
                f'Prueba con el óptimo p* = {num(p_star, 3)}.</div></div>', unsafe_allow_html=True)

seccion("Análisis")
tab1, tab2, tab3, tab4, tab5 = st.tabs([":material/show_chart: Costo vs umbral",
                                        ":material/grid_view: Matriz de confusión",
                                        ":material/payments: ¿En qué se va la plata?",
                                        ":material/timeline: Curva ROC",
                                        ":material/menu_book: Cómo leerlo"])
LAYOUT = dict(plot_bgcolor="white", paper_bgcolor="white", font=dict(family=FUENTE, color=NAVY, size=16),
              separators=",.", margin=dict(t=50, b=10, l=10, r=20), hoverlabel=dict(font_family=FUENTE, font_size=15),
              title_font=dict(family="Montserrat, sans-serif", size=19, color=NAVY))

# ---------------------------------------------------------------------------
# 5) Curva de costo total vs umbral
# ---------------------------------------------------------------------------
with tab1:
    with st.container(border=True, key="caja_1"):
        fig = go.Figure()
        lo, hi = sorted([umbral, 0.5])
        fig.add_vrect(x0=lo, x1=hi, fillcolor=VERDE if ahorro > 0 else ROJO, opacity=0.08, line_width=0,
                      annotation_text=("Zona de ahorro" if ahorro > 0 else ""), annotation_position="top left",
                      annotation_font=dict(color=VERDE, size=16))
        fig.add_trace(go.Scatter(x=grilla, y=costos, mode="lines", name="Costo total",
                                 line=dict(color=NAVY, width=4), fill="tozeroy", fillcolor="rgba(30,46,78,.04)",
                                 hovertemplate="Umbral %{x:.2f}<br>Costo US$ %{y:,.0f}<extra></extra>"))
        for t, color, nombre, dash in [(0.5, GRIS, "Umbral 0,5", "dash"),
                                       (p_star, NARANJO, f"Óptimo p* = {num(p_star, 3)}", "solid")]:
            fig.add_vline(x=t, line=dict(color=color, dash=dash, width=2.5))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=nombre,
                                     line=dict(color=color, dash=dash, width=2.5)))
        fig.add_trace(go.Scatter(x=[0.5], y=[costo_05], mode="markers", name="Costo con 0,5",
                                 marker=dict(color=GRIS, size=13, line=dict(color="white", width=2)),
                                 hovertemplate="US$ %{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[umbral], y=[costo_sel], mode="markers", name=f"Umbral evaluado {num(umbral, 3)}",
                                 marker=dict(color=NARANJO, size=18, symbol="diamond", line=dict(color="white", width=2)),
                                 hovertemplate="US$ %{y:,.0f}<extra></extra>"))
        fig.update_layout(height=460, title="Costo total según el umbral de decisión",
                          xaxis_title="Umbral de decisión", yaxis_title="Costo total (US$)",
                          yaxis_tickformat=",", legend=dict(orientation="h", y=-0.2, font=dict(size=15)), **LAYOUT)
        fig.update_xaxes(showgrid=True, gridcolor=GRILLA); fig.update_yaxes(gridcolor=GRILLA, rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    mini(c1, f"Umbral 0,5: <b>{usd(costo_05)}</b>", "Regla tradicional", GRIS)
    mini(c2, f"Óptimo p* = {num(p_star, 3)}: <b>{usd(costo(p_star, c_fp, c_fn))}</b>", "Regla por costos", NARANJO)
    mini(c3, f"Mínimo en test ({num(grilla[int(np.argmin(costos))], 2)}): "
             f"<b>{usd(costos.min())}</b>", "Solo referencia: se calcula mirando el test", NAVY)

# ---------------------------------------------------------------------------
# 6) Matriz de confusión con el costo en dinero de cada error
# ---------------------------------------------------------------------------
with tab2:
    with st.container(border=True, key="caja_2"):
        texto = [[f"<b>{num(tn)}</b><br>✓ No se llamó y no compraba<br>US$ 0",
                  f"<b>{num(fp)}</b><br>Llamada inútil (FP)<br>{usd(fp * c_fp)}"],
                 [f"<b>{num(fn)}</b><br>Venta perdida (FN)<br>{usd(fn * c_fn)}",
                  f"<b>{num(tp)}</b><br>✓ Venta lograda<br>US$ 0"]]
        fig = go.Figure(go.Heatmap(z=[[0, fp * c_fp], [fn * c_fn, 0]], text=texto, texttemplate="%{text}",
                                   textfont=dict(size=19, family=FUENTE), x=["Predicho: NO llamar", "Predicho: LLAMAR"],
                                   y=["Real: no compra", "Real: compra"],
                                   colorscale=[[0, CREMA], [1, ROJO]], showscale=False, hoverinfo="skip",
                                   xgap=6, ygap=6))
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=16))
        fig.update_xaxes(tickfont=dict(size=16))
        fig.update_layout(height=470, title=f"Umbral {num(umbral, 3)} · Costo total {usd(costo_sel)}", **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Más rojo = más dinero perdido. Los aciertos (✓) no cuestan nada.")

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

    with st.container(border=True, key="caja_3"):
        fig = go.Figure()
        fig.add_trace(go.Bar(y=comp["Estrategia"], x=comp["Costo FP"], orientation="h", name="Llamadas inútiles (FP)",
                             marker_color=AZUL, hovertemplate="US$ %{x:,.0f}<extra></extra>"))
        fig.add_trace(go.Bar(y=comp["Estrategia"], x=comp["Costo FN"], orientation="h", name="Ventas perdidas (FN)",
                             marker_color=ROJO, text=[usd(v) for v in comp["Costo total"]], textposition="outside",
                             textfont=dict(size=16, color=NAVY), hovertemplate="US$ %{x:,.0f}<extra></extra>"))
        fig.update_layout(barmode="stack", height=440, title="¿En qué se va la plata? Costo por estrategia",
                          xaxis_title="Costo total (US$)", xaxis_tickformat=",", bargap=0.35,
                          legend=dict(orientation="h", y=-0.22, font=dict(size=15)), **LAYOUT)
        fig.update_xaxes(range=[0, comp["Costo total"].max() * 1.22], gridcolor=GRILLA)
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
    m = (c_fp * N) / (c_fn * P)                                  # pendiente de la recta de iso-costo
    fpr_s, tpr_s = fp / N, tp / P
    xs = np.linspace(0, 1, 50)
    t05 = matriz(0.5)

    with st.container(border=True, key="caja_4"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Random Forest (AUC = {num(auc, 3)})",
                                 line=dict(color=NAVY, width=4)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(color=GRIS, dash="dash")))
        fig.add_trace(go.Scatter(x=xs, y=tpr_s + m * (xs - fpr_s), mode="lines",
                                 name=f"Iso-costo (pendiente {num(m, 2)})", line=dict(color=NARANJO, dash="dot", width=2.5)))
        fig.add_trace(go.Scatter(x=[t05[1] / N], y=[t05[3] / P], mode="markers", name="Umbral 0,5",
                                 marker=dict(color=GRIS, size=13)))
        fig.add_trace(go.Scatter(x=[fpr_s], y=[tpr_s], mode="markers", name=f"Umbral evaluado {num(umbral, 3)}",
                                 marker=dict(color=NARANJO, size=22, symbol="star", line=dict(color="white", width=1))))
        fig.update_layout(height=510, title="Curva ROC y recta de iso-costo",
                          xaxis_title="FPR (llamadas inútiles / no compradores)",
                          yaxis_title="TPR = Recall (compradores detectados)", xaxis_range=[0, 1], yaxis_range=[0, 1.02],
                          legend=dict(x=0.5, y=0.05, font=dict(size=15), bgcolor="rgba(255,255,255,.85)"), **LAYOUT)
        fig.update_xaxes(gridcolor=GRILLA); fig.update_yaxes(gridcolor=GRILLA)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Cada punto de la curva es un umbral distinto del mismo modelo. El mejor umbral es donde la "
                   "recta de iso-costo toca la curva más arriba y a la izquierda.")

# ---------------------------------------------------------------------------
# 9) Guía para la gerencia
# ---------------------------------------------------------------------------
with tab5:
    pasos = [("grafico", AZUL, "#E9ECF8", "Paso 1", "El modelo estima",
              "Para cada cliente calcula la probabilidad de que compre el seguro de vehículo."),
             ("objetivo", NARANJO, "#FDF0E3", "Paso 2", "El umbral decide",
              "Se llama a todo cliente cuya probabilidad sea mayor o igual al umbral."),
             ("telefono", AZUL, "#E9ECF8", "Error barato", "Llamada inútil (FP)",
              f"Llamamos y no compra: perdemos el costo de la llamada ({usd(c_fp)})."),
             ("dinero", ROJO, "#FAE8E9", "Error caro", "Venta perdida (FN)",
              f"No llamamos y sí compraba: perdemos la venta ({usd(c_fn)}).")]
    cols = st.columns(4)
    for col, (ic, color, fondo, n, tit, txt) in zip(cols, pasos):
        col.markdown(f'<div class="paso"><div class="ic2" style="background:{fondo}">{icono(ic, color)}</div>'
                     f'<div class="n">{n.upper()}</div><h4>{tit}</h4><p>{txt}</p></div>',
                     unsafe_allow_html=True)

    st.write("")
    with st.container(border=True, key="caja_5"):
        st.markdown(f"""
#### El umbral óptimo
Compara el costo esperado de llamar, (1 − p)·C_FP, con el de no llamar, p·C_FN:
conviene llamar si **p ≥ C_FP / (C_FP + C_FN) = {num(p_star, 3)}**.

#### ¿Por qué no 0,5?
El 0,5 solo es óptimo si ambos errores cuestan lo mismo (prueba el escenario *"Errores iguales"*).
Aquí perder una venta cuesta **{num(c_fn / c_fp, 1)} veces** más que una llamada inútil, así que conviene llamar a más gente.

#### Solo importa la razón FN/FP
Si duplicas ambos costos, el umbral no cambia; solo cambian los montos.
""")

# ---------------------------------------------------------------------------
# Pie de página
# ---------------------------------------------------------------------------
st.markdown(f'<div class="pie">{img(PIE, 70)}<p>Grupo 15 · Certamen 1 Machine Learning · Base: Health Insurance '
            f'Cross Sell Prediction (Kaggle) · Costos supuestos apoyados en fuentes; probabilidades del modelo '
            f'reentrenado en proporción real.</p></div>', unsafe_allow_html=True)

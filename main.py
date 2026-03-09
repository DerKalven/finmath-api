# =============================================================================
# FIXED INCOME RISK ANALYZER — Módulo 1: Time Value Engine
# =============================================================================
# Autor: Fernando  + Claude
# o: Herramienta interactiva de matemática financiera alineada
#            al syllabus del examen FM (SOA/CAS).
# Stack: Python + Streamlit
# Estilo: Institucional McKinsey — blanco, azul marino, dorado.
#
# Para correr localmente:
#   pip install streamlit
#   streamlit run app.py
#
# Para desplegar en Streamlit Cloud:
#   1. Sube este archivo a GitHub
#   2. Ve a share.streamlit.io
#   3. Conecta el repositorio
# =============================================================================

import streamlit as st
import math
import pandas as pd

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN DE PÁGINA Y ESTILOS
# =============================================================================
# st.set_page_config debe ser la primera llamada de Streamlit.
# Aquí definimos el título, ícono y layout de la app.
# =============================================================================

st.set_page_config(
    page_title="Fixed Income Risk Analyzer · FM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado: inyectado con st.markdown para sobreescribir el tema
# de Streamlit y lograr el estilo McKinsey (azul marino + dorado + blanco).
MCKINSEY_CSS = """
<style>
  /* ── Fuentes ── */
  @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

  /* ── Fondo y texto base ── */
  .stApp { background-color: #f7f8fa !important; }
  html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace !important; }

  /* ── Sidebar: azul marino ── */
  [data-testid="stSidebar"] {
    background-color: #002244 !important;
    border-right: 3px solid #b8960c !important;
  }
  [data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stNumberInput label { color: rgba(255,255,255,0.6) !important; font-size: 11px !important; }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: rgba(255,255,255,0.5) !important; font-size: 10px !important; }

  /* ── Tabs: estilo institucional ── */
  .stTabs [data-baseweb="tab-list"] {
    background-color: #ffffff;
    border-bottom: 2px solid #e0e3e8;
    gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
    font-size: 11px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #999999 !important;
    background: transparent !important;
    border-bottom: 3px solid transparent !important;
    padding: 10px 24px !important;
  }
  .stTabs [aria-selected="true"] {
    color: #002244 !important;
    border-bottom-color: #002244 !important;
    font-weight: 700 !important;
  }

  /* ── Métricas: tarjetas limpias ── */
  [data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e0e3e8;
    border-top: 3px solid #002244;
    padding: 16px 20px;
    border-radius: 0 !important;
  }
  [data-testid="metric-container"] label {
    font-size: 9px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #999999 !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Libre Baskerville', serif !important;
    font-size: 28px !important;
    color: #002244 !important;
    font-weight: 700 !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 11px !important;
  }

  /* ── Botón primario ── */
  .stButton > button {
    background-color: #002244 !important;
    color: white !important;
    border: none !important;
    border-radius: 0 !important;
    font-size: 10px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 12px 24px !important;
    width: 100% !important;
    font-family: 'IBM Plex Mono', monospace !important;
    transition: background 0.15s !important;
  }
  .stButton > button:hover { background-color: #0a4d8c !important; }

  /* ── DataFrames / Tablas ── */
  [data-testid="stDataFrame"] { border: 1px solid #e0e3e8 !important; }
  .stDataFrame thead th {
    background-color: #f7f8fa !important;
    font-size: 9px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #999999 !important;
    border-bottom: 2px solid #002244 !important;
  }

  /* ── Inputs y selectbox ── */
  .stNumberInput input, .stSelectbox select {
    border-radius: 0 !important;
    border-bottom: 2px solid #002244 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
  }

  /* ── Info / success boxes ── */
  .stAlert { border-radius: 0 !important; }

  /* ── Divisores ── */
  hr { border-color: #e0e3e8 !important; }

  /* ── Ocultar el menú de Streamlit (opcional) ── */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
</style>
"""

# Inyectar el CSS en la app
st.markdown(MCKINSEY_CSS, unsafe_allow_html=True)


# =============================================================================
# SECCIÓN 2: FUNCIONES MATEMÁTICAS
# =============================================================================
# Funciones puras de matemática financiera. Sin dependencias de Streamlit.
# Separadas del código de interfaz para facilitar pruebas y reutilización.
# =============================================================================

def effective_rate(rate: float, m: int, conv: str = "nominal") -> float:
    """
    Convierte cualquier convención de tasa a tasa efectiva anual.

    El FM define tres formas de expresar una tasa:
      - "nominal":    i^(m) capitalizada m veces/año → (1 + i/m)^m − 1
      - "effective":  ya es efectiva anual → sin transformación
      - "continuous": fuerza de interés δ → e^δ − 1

    Args:
        rate: tasa en decimal (ej: 0.06 para 6%)
        m:    períodos de capitalización (solo aplica si conv="nominal")
        conv: convención de la tasa

    Returns:
        Tasa efectiva anual equivalente (decimal)
    """
    if conv == "continuous":
        return math.exp(rate) - 1
    elif conv == "effective":
        return rate
    else:  # nominal
        return (1 + rate / m) ** m - 1


def calc_fv(pv: float, i: float, n: float) -> float:
    """
    Future Value de un capital único.
    Fórmula FM: FV = PV · (1 + i)^n
    """
    return pv * (1 + i) ** n


def calc_pv(fv: float, i: float, n: float) -> float:
    """
    Present Value de un capital único.
    Fórmula FM: PV = FV · v^n   donde v = 1/(1+i)
    """
    return fv / (1 + i) ** n


def annuity_immediate(pmt: float, i: float, n: float) -> float:
    """
    PV de anualidad-vencida (pagos al FINAL del período).
    Notación FM: a⌐n|i
    Fórmula:     PV = PMT · (1 − v^n) / i

    Caso especial i=0: PV = PMT × n
    """
    if i == 0:
        return pmt * n
    return pmt * (1 - (1 + i) ** (-n)) / i


def annuity_due(pmt: float, i: float, n: float) -> float:
    """
    PV de anualidad-anticipada (pagos al INICIO del período).
    Notación FM: ä⌐n|i
    Relación:    ä = a · (1 + i)
    """
    return annuity_immediate(pmt, i, n) * (1 + i)


def annuity_deferred(pmt: float, i: float, n: float, d: float) -> float:
    """
    PV de anualidad diferida d períodos.
    Fórmula FM: d|a⌐n|i = v^d · a⌐n|i
    """
    return annuity_immediate(pmt, i, n) * (1 + i) ** (-d)


def amortization_schedule(
    pv: float,
    i: float,
    n: int,
    scheme: str = "french",
    extra_map: dict = None
) -> pd.DataFrame:
    """
    Genera la tabla de amortización completa, con soporte para abonos
    extraordinarios.

    Esquemas disponibles:
      - "french":   Cuota constante PMT = PV·i / (1−(1+i)^−n)
                    El interés decrece y el capital crece cada período.
                    Más común en hipotecas y créditos de consumo.

      - "german":   Capital constante K = PV/n
                    La cuota total decrece cada período.

      - "american": Solo interés cada período. Capital completo al final (bullet).
                    Común en bonos corporativos.

    Abonos extraordinarios (extra_map):
      Diccionario { período: monto } que se aplica DIRECTAMENTE al capital.
      Efecto: reduce saldo, recalcula cuota sobre saldo residual (sistema francés).

    Args:
        pv:        monto del préstamo
        i:         tasa efectiva por período (decimal)
        n:         número de períodos original
        scheme:    esquema de amortización
        extra_map: { período (int): monto_abono (float) }

    Returns:
        DataFrame con columnas: Período, Cuota, Interés, Capital, Abono Extra, Saldo
    """
    if extra_map is None:
        extra_map = {}

    rows = []

    if scheme == "french":
        # Cuota constante inicial
        pmt       = (pv * i) / (1 - (1 + i) ** (-n))
        balance   = pv
        remaining = n
        t         = 1

        # El loop continúa hasta que el saldo sea ~0 o se supere el plazo
        while balance > 0.005 and t <= n + len(extra_map) + 1:
            interest  = balance * i
            principal = min(pmt - interest, balance)
            extra     = min(extra_map.get(t, 0), max(0, balance - principal))

            balance = max(0, balance - principal - extra)

            rows.append({
                "Período":     t,
                "Cuota":       round(principal + interest, 2),
                "Interés":     round(interest, 2),
                "Capital":     round(principal, 2),
                "Abono Extra": round(extra, 2) if extra > 0 else 0,
                "Saldo":       round(balance, 2),
                "Cancelado":   balance < 0.005,
            })

            remaining -= 1

            # Después de un abono: recalcular cuota sobre saldo y períodos restantes
            if extra > 0 and balance > 0.005 and remaining > 0:
                pmt = (balance * i) / (1 - (1 + i) ** (-remaining))

            if balance < 0.005:
                break
            t += 1

    elif scheme == "german":
        base_principal = pv / n   # capital constante por período
        balance = pv

        for t in range(1, n + 1):
            if balance < 0.005:
                break
            interest  = balance * i
            principal = min(base_principal, balance)
            extra     = min(extra_map.get(t, 0), max(0, balance - principal))

            balance = max(0, balance - principal - extra)

            rows.append({
                "Período":     t,
                "Cuota":       round(principal + interest, 2),
                "Interés":     round(interest, 2),
                "Capital":     round(principal, 2),
                "Abono Extra": round(extra, 2) if extra > 0 else 0,
                "Saldo":       round(balance, 2),
                "Cancelado":   balance < 0.005,
            })

    else:  # american / bullet
        balance = pv

        for t in range(1, n + 1):
            if balance < 0.005:
                break
            interest  = balance * i
            principal = balance if t == n else 0
            extra     = min(extra_map.get(t, 0), balance) if t < n else 0

            balance = max(0, balance - principal - extra)

            rows.append({
                "Período":     t,
                "Cuota":       round(principal + interest, 2),
                "Interés":     round(interest, 2),
                "Capital":     round(principal, 2),
                "Abono Extra": round(extra, 2) if extra > 0 else 0,
                "Saldo":       round(balance, 2),
                "Cancelado":   balance < 0.005,
            })

    return pd.DataFrame(rows)


def equivalent_rates(i_eff: float) -> pd.DataFrame:
    """
    Dado i_eff (tasa efectiva anual), calcula todas las tasas equivalentes.
    Fórmula inversa: i^(m) = m · [(1+i_eff)^(1/m) − 1]
    La fuerza de interés δ = ln(1 + i_eff)

    Returns:
        DataFrame con columnas: Convención, Tasa
    """
    delta = math.log(1 + i_eff)
    data = [
        ("Efectiva anual (m=1)",     (1 + i_eff) ** (1/1)   - 1),
        ("Nominal semestral (m=2)",  2   * ((1 + i_eff) ** (1/2)   - 1)),
        ("Nominal trimestral (m=4)", 4   * ((1 + i_eff) ** (1/4)   - 1)),
        ("Nominal mensual (m=12)",   12  * ((1 + i_eff) ** (1/12)  - 1)),
        ("Nominal diaria (m=365)",   365 * ((1 + i_eff) ** (1/365) - 1)),
        ("Fuerza de interés δ",      delta),
    ]
    df = pd.DataFrame(data, columns=["Convención", "Tasa"])
    df["Tasa (%)"] = df["Tasa"].apply(lambda x: f"{x*100:.4f}%")
    return df[["Convención", "Tasa (%)"]]


# =============================================================================
# SECCIÓN 3: SIDEBAR — Navegación y configuración global
# =============================================================================
# En Streamlit, el sidebar es ideal para controles globales y navegación.
# st.sidebar.* funciona igual que st.* pero en el panel lateral.
# =============================================================================

with st.sidebar:
    # Logo / título de la app
    st.markdown("""
    <div style='padding-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;'>
        <div style='font-family: "Libre Baskerville", serif; font-size: 16px; font-weight: 700; color: white;'>
            Fixed Income Risk Analyzer
        </div>
        <div style='font-size: 9px; letter-spacing: 0.2em; text-transform: uppercase; color: rgba(255,255,255,0.45); margin-top: 4px;'>
            Módulo 1 · FM Exam Prep · SOA
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Selector de herramienta — controla qué tab se muestra como activo
    st.markdown("**Herramienta activa**")
    tool = st.selectbox(
        label="Seleccionar herramienta",
        options=[
            "💰 Valor del Dinero en el Tiempo",
            "📅 Anualidades",
            "🏦 Amortización",
            "🔄 Conversión de Tasas",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Referencia FM en el sidebar
    st.markdown("""
    <div style='font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase;
                color: rgba(255,255,255,0.35); margin-bottom: 8px;'>
        Próximos módulos
    </div>
    """, unsafe_allow_html=True)

    # Lista de módulos futuros (deshabilitados visualmente)
    for item in ["📈 Bond Pricer", "📉 Yield Curve", "📐 Duration & Convexity", "🛡️ Immunización"]:
        st.markdown(
            f"<div style='font-size:12px; color:rgba(255,255,255,0.2); padding: 6px 0;'>{item}</div>",
            unsafe_allow_html=True
        )

    st.divider()

    # Créditos
    st.markdown(
        "<div style='font-size:9px; color:rgba(255,255,255,0.25);'>Built with Python + Streamlit</div>",
        unsafe_allow_html=True
    )


# =============================================================================
# SECCIÓN 4: HEADER PRINCIPAL
# =============================================================================
# Título de la app con estilo editorial McKinsey.
# st.markdown con unsafe_allow_html=True permite HTML/CSS personalizado.
# =============================================================================

# Mapeo de herramienta → metadatos de la página
PAGE_META = {
    "💰 Valor del Dinero en el Tiempo": {
        "eyebrow": "Sección 1 & 2 — FM",
        "title":   "Valor del Dinero en el Tiempo",
        "desc":    "Calcula valores presentes y futuros bajo cualquier convención de tasa: nominal, efectiva anual o fuerza de interés.",
        "fm":      "FM Sections 1–2: Interest Measurement",
    },
    "📅 Anualidades": {
        "eyebrow": "Sección 2 — FM",
        "title":   "Valuación de Anualidades",
        "desc":    "Anualidades vencidas, anticipadas y diferidas con notación estándar FM. PV, valor acumulado y desglose de interés.",
        "fm":      "FM Section 2: Annuities",
    },
    "🏦 Amortización": {
        "eyebrow": "Sección 2 — FM",
        "title":   "Tablas de Amortización",
        "desc":    "Tabla período a período para esquemas francés, alemán y americano. Soporte para abonos extraordinarios con comparativo de ahorro.",
        "fm":      "FM Section 2: Loan Repayment",
    },
    "🔄 Conversión de Tasas": {
        "eyebrow": "Sección 1 — FM",
        "title":   "Conversión de Tasas de Interés",
        "desc":    "Dado i^(m), calcula instantáneamente todas las tasas equivalentes y la fuerza de interés δ.",
        "fm":      "FM Section 1: Interest Rate Measurement",
    },
}

meta = PAGE_META[tool]

# Header con estilo McKinsey
st.markdown(f"""
<div style='margin-bottom: 8px;'>
    <div style='font-family: "Libre Baskerville", serif; font-style: italic;
                font-size: 13px; color: #999999; margin-bottom: 4px;'>
        {meta["eyebrow"]}
    </div>
    <div style='font-family: "Libre Baskerville", serif; font-size: 32px;
                font-weight: 700; color: #002244; letter-spacing: -0.5px; line-height: 1.15;'>
        {meta["title"]}
    </div>
</div>
<div style='display: inline-block; background: #002244; color: white; font-size: 8px;
            letter-spacing: 0.18em; text-transform: uppercase; padding: 3px 10px; margin-bottom: 8px;'>
    ▸ {meta["fm"]}
</div>
<div style='font-size: 12px; color: #666666; line-height: 1.7; max-width: 600px; margin-bottom: 32px;'>
    {meta["desc"]}
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SECCIÓN 5: HERRAMIENTAS
# =============================================================================
# Cada herramienta tiene su propio bloque condicional (if/elif).
# Streamlit re-ejecuta el script completo en cada interacción del usuario,
# por eso los inputs de st.number_input, st.selectbox, etc. mantienen
# su valor gracias al session_state interno de Streamlit.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 5.1 VALOR DEL DINERO EN EL TIEMPO
# ─────────────────────────────────────────────────────────────────────────────
if tool == "💰 Valor del Dinero en el Tiempo":

    # Fórmula activa
    st.markdown("""
    <div style='background:#002244; color:rgba(255,255,255,0.85); padding:12px 20px;
                font-size:12px; font-style:italic; margin-bottom:24px;
                border-left:3px solid #b8960c;'>
        FV = PV · (1 + i)ⁿ &nbsp;|&nbsp; PV = FV · v^n &nbsp;|&nbsp;
        v = 1/(1+i) &nbsp;|&nbsp; δ = ln(1+i) &nbsp;|&nbsp; i_eff = (1 + i^(m)/m)^m − 1
    </div>
    """, unsafe_allow_html=True)

    # Layout: columna de inputs | columna de resultado
    col_inputs, col_result = st.columns([1, 1], gap="large")

    with col_inputs:
        st.markdown("**Parámetros**")

        # Modo: calcular FV o PV
        mode = st.selectbox(
            "Calcular",
            options=["Future Value (dado PV)", "Present Value (dado FV)"]
        )

        # Input dinámico según el modo
        if "FV" in mode:
            pv_input = st.number_input("Present Value ($)", value=1000.0, step=100.0, format="%.2f")
        else:
            fv_input = st.number_input("Future Value ($)", value=2000.0, step=100.0, format="%.2f")

        # Tasa
        rate_pct = st.number_input("Tasa (%)", value=5.0, step=0.1, format="%.4f")

        # Convención de la tasa
        conv = st.selectbox(
            "Convención de tasa",
            options=["Nominal i^(m)", "Efectiva anual i", "Fuerza de interés δ"]
        )

        # Selector de m — solo visible si la tasa es nominal
        if conv == "Nominal i^(m)":
            m_map = {"Anual (m=1)": 1, "Semestral (m=2)": 2,
                     "Trimestral (m=4)": 4, "Mensual (m=12)": 12, "Diaria (m=365)": 365}
            m_label = st.selectbox("Capitalización (m)", list(m_map.keys()), index=3)
            m = m_map[m_label]
        else:
            m = 1

        n = st.number_input("Períodos (n)", value=10, step=1, min_value=1)

        calcular = st.button("▸ Calcular")

    with col_result:
        if calcular:
            # Convertir la convención al string que usa effective_rate()
            conv_key = {"Nominal i^(m)": "nominal",
                        "Efectiva anual i": "effective",
                        "Fuerza de interés δ": "continuous"}[conv]

            i_eff = effective_rate(rate_pct / 100, m, conv_key)

            if "FV" in mode:
                result_val = calc_fv(pv_input, i_eff, n)
                result_label = "Future Value"
                detail = f"PV ${pv_input:,.2f} crece durante {n} períodos"
            else:
                result_val = calc_pv(fv_input, i_eff, n)
                result_label = "Present Value"
                detail = f"FV ${fv_input:,.2f} descontado {n} períodos"

            delta_val = math.log(1 + i_eff)

            # Métricas principales
            m1, m2, m3 = st.columns(3)
            m1.metric(result_label, f"${result_val:,.2f}")
            m2.metric("i efectiva anual", f"{i_eff*100:.4f}%")
            m3.metric("Fuerza de interés δ", f"{delta_val*100:.4f}%")

            st.caption(detail)

            # Tabla de equivalencias de tasas
            st.divider()
            st.markdown("**Tasas equivalentes**")
            df_rates = equivalent_rates(i_eff)
            st.dataframe(df_rates, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5.2 ANUALIDADES
# ─────────────────────────────────────────────────────────────────────────────
elif tool == "📅 Anualidades":

    # ------------------------------------------------------------------
    # FREQUENCY HELPERS
    # FM key concept: when payments are more frequent than annual, you
    # must convert the annual effective rate to a per-period rate first.
    # Formula: i_period = (1 + i_annual)^(1/freq) − 1
    # The annuity formulas themselves don't change — only the rate and
    # the interpretation of n (= total number of payments) change.
    # ------------------------------------------------------------------
    FREQ_OPTIONS = {
        "Mensual — 12 pagos/año":      12,
        "Bimestral — 6 pagos/año":      6,
        "Trimestral — 4 pagos/año":     4,
        "Semestral — 2 pagos/año":      2,
        "Anual — 1 pago/año":           1,
    }
    FREQ_N_LABEL = {
        12: "Número de meses (n)",
        6:  "Número de bimestres (n)",
        4:  "Número de trimestres (n)",
        2:  "Número de semestres (n)",
        1:  "Número de años (n)",
    }
    FREQ_LABEL = {
        12: "mensual",
        6:  "bimestral",
        4:  "trimestral",
        2:  "semestral",
        1:  "anual",
    }

    # Selector del tipo de anualidad como tabs de Streamlit
    tab_imm, tab_due, tab_def = st.tabs([
        "Vencida (Immediate)",
        "Anticipada (Due)",
        "Diferida (Deferred)",
    ])

    # ── Anualidad Vencida ──
    with tab_imm:
        st.markdown("""
        <div style='background:#002244; color:rgba(255,255,255,0.85); padding:12px 20px;
                    font-size:12px; font-style:italic; margin-bottom:24px;
                    border-left:3px solid #b8960c;'>
            PV = PMT · a⌐n|i = PMT · (1 − vⁿ) / i &nbsp;|&nbsp;
            i_período = (1 + i_anual)^(1/freq) − 1
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("**Parámetros**")
            pmt_i   = st.number_input("Pago periódico PMT ($)", value=100.0, step=10.0, key="imm_pmt")
            i_i     = st.number_input("Tasa efectiva anual (%)", value=5.0, step=0.1, key="imm_i")
            freq_i  = st.selectbox("Frecuencia de pago", list(FREQ_OPTIONS.keys()), key="imm_freq")
            freq_i_val = FREQ_OPTIONS[freq_i]
            n_i     = st.number_input(FREQ_N_LABEL[freq_i_val], value=20, step=1, min_value=1, key="imm_n")
            calc_i  = st.button("▸ Calcular", key="calc_imm")

        with col2:
            if calc_i:
                i_annual  = i_i / 100
                # Rate conversion: annual → per-period
                i_period  = (1 + i_annual) ** (1 / freq_i_val) - 1
                pv        = annuity_immediate(pmt_i, i_period, n_i)
                fv        = calc_fv(pv, i_period, n_i)
                total_pmts     = pmt_i * n_i
                total_interest = fv - total_pmts

                m1, m2 = st.columns(2)
                m1.metric("Present Value a⌐n|i", f"${pv:,.2f}")
                m2.metric("Valor Acumulado (FV)", f"${fv:,.2f}")

                m3, m4 = st.columns(2)
                m3.metric("Suma de pagos", f"${total_pmts:,.2f}")
                m4.metric("Interés total generado", f"${total_interest:,.2f}")

                st.info(
                    f"**Tasa {FREQ_LABEL[freq_i_val]} efectiva**: {i_period*100:.4f}%  \n"
                    f"Converted from annual {i_annual*100:.4f}% using "
                    f"i_period = (1 + i)^(1/{freq_i_val}) − 1"
                )

    # ── Anualidad Anticipada ──
    with tab_due:
        st.markdown("""
        <div style='background:#002244; color:rgba(255,255,255,0.85); padding:12px 20px;
                    font-size:12px; font-style:italic; margin-bottom:24px;
                    border-left:3px solid #b8960c;'>
            PV = PMT · ä⌐n|i = PMT · (1+i) · a⌐n|i &nbsp;|&nbsp;
            i_período = (1 + i_anual)^(1/freq) − 1
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("**Parámetros**")
            pmt_d   = st.number_input("Pago periódico PMT ($)", value=100.0, step=10.0, key="due_pmt")
            i_d     = st.number_input("Tasa efectiva anual (%)", value=5.0, step=0.1, key="due_i")
            freq_d  = st.selectbox("Frecuencia de pago", list(FREQ_OPTIONS.keys()), key="due_freq")
            freq_d_val = FREQ_OPTIONS[freq_d]
            n_d     = st.number_input(FREQ_N_LABEL[freq_d_val], value=20, step=1, min_value=1, key="due_n")
            calc_d  = st.button("▸ Calcular", key="calc_due")

        with col2:
            if calc_d:
                i_annual  = i_d / 100
                i_period  = (1 + i_annual) ** (1 / freq_d_val) - 1
                pv        = annuity_due(pmt_d, i_period, n_d)
                fv        = calc_fv(pv, i_period, n_d)
                total_pmts     = pmt_d * n_d
                total_interest = fv - total_pmts

                # Comparison: due vs immediate (same freq)
                pv_imm = annuity_immediate(pmt_d, i_period, n_d)

                m1, m2 = st.columns(2)
                m1.metric("Present Value ä⌐n|i", f"${pv:,.2f}",
                          delta=f"+${pv - pv_imm:,.2f} vs vencida")
                m2.metric("Valor Acumulado (FV)", f"${fv:,.2f}")

                m3, m4 = st.columns(2)
                m3.metric("Suma de pagos", f"${total_pmts:,.2f}")
                m4.metric("Interés total", f"${total_interest:,.2f}")

                st.info(
                    f"**Tasa {FREQ_LABEL[freq_d_val]} efectiva**: {i_period*100:.4f}%  \n"
                    f"The due annuity is worth **${pv - pv_imm:,.2f} more** than the immediate "
                    f"annuity because each payment arrives one period earlier (×(1+i) factor)."
                )

    # ── Anualidad Diferida ──
    with tab_def:
        st.markdown("""
        <div style='background:#002244; color:rgba(255,255,255,0.85); padding:12px 20px;
                    font-size:12px; font-style:italic; margin-bottom:24px;
                    border-left:3px solid #b8960c;'>
            PV = v^d · PMT · a⌐n|i &nbsp;|&nbsp; Primer pago en período d+1 &nbsp;|&nbsp;
            i_período = (1 + i_anual)^(1/freq) − 1
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("**Parámetros**")
            pmt_df  = st.number_input("Pago periódico PMT ($)", value=100.0, step=10.0, key="def_pmt")
            i_df    = st.number_input("Tasa efectiva anual (%)", value=5.0, step=0.1, key="def_i")
            freq_df = st.selectbox("Frecuencia de pago", list(FREQ_OPTIONS.keys()), key="def_freq")
            freq_df_val = FREQ_OPTIONS[freq_df]
            n_df    = st.number_input(FREQ_N_LABEL[freq_df_val], value=20, step=1, min_value=1, key="def_n")
            d_df    = st.number_input(
                f"Diferimiento d (períodos {FREQ_LABEL[freq_df_val]}s)",
                value=5, step=1, min_value=1, key="def_d"
            )
            calc_df = st.button("▸ Calcular", key="calc_def")

        with col2:
            if calc_df:
                i_annual    = i_df / 100
                i_period    = (1 + i_annual) ** (1 / freq_df_val) - 1
                pv          = annuity_deferred(pmt_df, i_period, n_df, d_df)
                pv_no_defer = annuity_immediate(pmt_df, i_period, n_df)
                # FV is calculated at the end of all payments (d + n periods from now)
                fv          = calc_fv(pv, i_period, n_df + d_df)

                m1, m2 = st.columns(2)
                m1.metric("Present Value diferida", f"${pv:,.2f}",
                          delta=f"${pv - pv_no_defer:,.2f} vs sin diferir")
                m2.metric("Valor Acumulado (FV)", f"${fv:,.2f}")

                m3, m4 = st.columns(2)
                m3.metric(f"Primer pago en período", f"{int(d_df) + 1}")
                m4.metric(f"Último pago en período", f"{int(d_df) + int(n_df)}")

                st.info(
                    f"**Tasa {FREQ_LABEL[freq_df_val]} efectiva**: {i_period*100:.4f}%  \n"
                    f"The annuity is worth **${pv_no_defer - pv:,.2f} less** than without deferral "
                    f"because {int(d_df)} additional periods are discounted (v^{int(d_df)} factor)."
                )


# ─────────────────────────────────────────────────────────────────────────────
# 5.3 AMORTIZACIÓN CON ABONOS EXTRAORDINARIOS
# ─────────────────────────────────────────────────────────────────────────────
elif tool == "🏦 Amortización":

    st.markdown("""
    <div style='background:#002244; color:rgba(255,255,255,0.85); padding:12px 20px;
                font-size:12px; font-style:italic; margin-bottom:24px;
                border-left:3px solid #b8960c;'>
        Francés: PMT = PV·i / (1−(1+i)⁻ⁿ) &nbsp;|&nbsp;
        Alemán: K = PV/n &nbsp;|&nbsp;
        Abono extra → reduce capital directo → recalcula cuota
    </div>
    """, unsafe_allow_html=True)

    # ── Parámetros base ──
    col_params, col_summary = st.columns([1, 1], gap="large")

    with col_params:
        st.markdown("**Parámetros del préstamo**")
        pv_a  = st.number_input("Monto del préstamo ($)", value=10000.0, step=500.0, key="amort_pv_w")
        i_a   = st.number_input("Tasa efectiva anual (%)", value=6.0, step=0.1, key="amort_i_w")
        n_a   = st.number_input("Número de períodos", value=12, step=1, min_value=1, key="amort_n_w")
        scheme_map = {
            "Francés — cuota constante": "french",
            "Alemán — capital constante": "german",
            "Americano — bullet (bala)": "american",
        }
        scheme_label = st.selectbox("Esquema", list(scheme_map.keys()))
        scheme = scheme_map[scheme_label]
        gen_table = st.button("▸ Generar Tabla")

    # Guardamos la tabla base en session_state para que persista entre interacciones
    # session_state es el mecanismo de Streamlit para mantener estado entre re-runs
    if gen_table:
        i_dec = i_a / 100
        df_base = amortization_schedule(pv_a, i_dec, int(n_a), scheme, {})
        st.session_state["df_base"]  = df_base
        st.session_state["pv_stored"] = pv_a
        st.session_state["i_stored"]  = i_dec
        st.session_state["n_stored"]  = int(n_a)
        st.session_state["scheme_stored"]   = scheme
        st.session_state["extra_map"] = {}   # resetear abonos al generar nueva tabla

    # ── Panel de abonos extraordinarios ──
    # Solo visible cuando ya existe una tabla generada
    if "df_base" in st.session_state:
        df_base = st.session_state["df_base"]
        extra_map: dict = st.session_state.get("extra_map", {})

        # Resumen base (sin abonos)
        total_int_base = df_base["Interés"].sum()
        total_pmt_base = df_base["Cuota"].sum()

        with col_summary:
            st.markdown("**Resumen**")
            m1, m2 = st.columns(2)
            m1.metric("Períodos base", len(df_base))
            m2.metric("Interés total base", f"${total_int_base:,.2f}")

        st.divider()

        # ── Agregar abono extraordinario ──
        st.markdown("**⚡ Abonos Extraordinarios**")
        st.caption(
            "Un abono extraordinario se aplica directamente al capital, "
            "reduciendo el saldo y recalculando la cuota sobre el saldo residual."
        )

        col_e1, col_e2, col_e3 = st.columns([1, 2, 1])
        with col_e1:
            # El período del abono no puede exceder el número de períodos original
            periodo_extra = st.number_input(
                "Período", min_value=1, max_value=int(n_a), value=3, key="extra_t"
            )
        with col_e2:
            monto_extra = st.number_input(
                "Monto del abono ($)", min_value=0.0, value=500.0, step=100.0, key="extra_amt"
            )
        with col_e3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            agregar = st.button("+ Agregar")

        # Al agregar: actualizar el mapa y recalcular la tabla
        if agregar and monto_extra > 0:
            extra_map[int(periodo_extra)] = extra_map.get(int(periodo_extra), 0) + monto_extra
            st.session_state["extra_map"] = extra_map

        # Mostrar abonos registrados con botón de eliminar
        if extra_map:
            st.markdown("**Abonos registrados:**")
            cols_extras = st.columns(min(len(extra_map), 4))
            for idx, (t, amt) in enumerate(sorted(extra_map.items())):
                with cols_extras[idx % 4]:
                    st.markdown(
                        f"""<div style='background:#fffbeb; border:1px solid #b8960c;
                                       border-left:3px solid #b8960c; padding:8px 12px;
                                       font-size:11px; margin-bottom:8px;'>
                            <div style='color:#666;'>Período {t}</div>
                            <div style='color:#b8960c; font-weight:700;'>${amt:,.2f}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                    # Botón para eliminar este abono
                    if st.button(f"× Eliminar P{t}", key=f"del_{t}"):
                        del extra_map[t]
                        st.session_state["extra_map"] = extra_map
                        st.rerun()   # forzar re-run para actualizar la tabla

        st.divider()

        # ── Calcular tabla con abonos ──
        i_dec   = st.session_state["i_stored"]
        n_orig  = st.session_state["n_stored"]
        scheme  = st.session_state["scheme_stored"]
        pv_orig = st.session_state["pv_stored"]

        df_extra = amortization_schedule(pv_orig, i_dec, n_orig, scheme, extra_map)
        has_extras = len(extra_map) > 0

        # ── Métricas comparativas ──
        if has_extras:
            total_int_extra = df_extra["Interés"].sum()
            periods_saved   = len(df_base) - len(df_extra)
            int_saved       = total_int_base - total_int_extra
            total_extras    = sum(extra_map.values())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Períodos", len(df_extra),
                      delta=f"-{periods_saved}" if periods_saved > 0 else "0",
                      delta_color="inverse")
            m2.metric("Interés total", f"${total_int_extra:,.2f}",
                      delta=f"-${int_saved:,.2f}",
                      delta_color="inverse")
            m3.metric("💰 Interés ahorrado", f"${int_saved:,.2f}")
            m4.metric("Total abonos extra", f"${total_extras:,.2f}")

            if periods_saved > 0:
                st.success(
                    f"✅ Préstamo cancelado **{periods_saved} período(s) antes** del plazo original. "
                    f"Interés ahorrado: **${int_saved:,.2f}**"
                )
        else:
            m1, m2 = st.columns(2)
            m1.metric("Períodos", len(df_base))
            m2.metric("Interés total", f"${total_int_base:,.2f}")

        # ── Tabla de amortización ──
        st.markdown(
            f"**Tabla de Amortización** — {len(df_extra if has_extras else df_base)} períodos"
            + (" ★ Con abonos extraordinarios" if has_extras else ""),
        )

        # Preparar DataFrame para mostrar (sin columna 'Cancelado' que es interna)
        df_display = (df_extra if has_extras else df_base).copy()

        # Formatear columnas numéricas como strings para mejor visualización
        for col in ["Cuota", "Interés", "Capital", "Saldo"]:
            df_display[col] = df_display[col].apply(lambda x: f"${x:,.2f}")

        # Abono Extra: mostrar "—" si es 0
        df_display["Abono Extra"] = df_display["Abono Extra"].apply(
            lambda x: f"${float(x.replace('$','').replace(',','')):,.2f}"
            if isinstance(x, str) and float(x.replace('$','').replace(',','')) > 0
            else ("—" if x == 0 or x == "0" else f"${float(str(x)):,.2f}")
        )

        # Mostrar solo columnas relevantes
        cols_show = ["Período", "Cuota", "Interés", "Capital"]
        if has_extras:
            cols_show.append("Abono Extra")
        cols_show.append("Saldo")

        st.dataframe(
            df_display[cols_show],
            hide_index=True,
            use_container_width=True,
        )

        # Botón de descarga como CSV
        csv = (df_extra if has_extras else df_base)[cols_show[:-1] + ["Saldo"]].to_csv(index=False)
        st.download_button(
            label="📥 Descargar tabla como CSV",
            data=csv,
            file_name="amortizacion.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5.4 CONVERSIÓN DE TASAS
# ─────────────────────────────────────────────────────────────────────────────
elif tool == "🔄 Conversión de Tasas":

    st.markdown("""
    <div style='background:#002244; color:rgba(255,255,255,0.85); padding:12px 20px;
                font-size:12px; font-style:italic; margin-bottom:24px;
                border-left:3px solid #b8960c;'>
        i^(m) = m · [(1+i)^(1/m) − 1] &nbsp;|&nbsp; δ = ln(1 + i) &nbsp;|&nbsp; i = e^δ − 1
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown("**Tasa de entrada**")

        conv_r = st.selectbox(
            "Convención",
            ["Nominal i^(m)", "Efectiva anual i", "Fuerza de interés δ"],
            key="rates_conv"
        )

        rate_r = st.number_input("Tasa (%)", value=6.0, step=0.01, format="%.4f", key="rates_val")

        if conv_r == "Nominal i^(m)":
            m_r_map = {"Anual (m=1)": 1, "Semestral (m=2)": 2,
                       "Trimestral (m=4)": 4, "Mensual (m=12)": 12, "Diaria (m=365)": 365}
            m_r_label = st.selectbox("Capitalización", list(m_r_map.keys()), index=3, key="rates_m")
            m_r = m_r_map[m_r_label]
        else:
            m_r = 1

        conv_key_r = {"Nominal i^(m)": "nominal",
                      "Efectiva anual i": "effective",
                      "Fuerza de interés δ": "continuous"}[conv_r]

        # La conversión es en tiempo real — sin botón Calcular
        i_eff_r = effective_rate(rate_r / 100, m_r, conv_key_r)
        delta_r  = math.log(1 + i_eff_r)

        # Métrica principal
        st.metric("Tasa Efectiva Anual Equivalente", f"{i_eff_r*100:.4f}%")
        st.metric("Fuerza de interés δ", f"{delta_r*100:.4f}%")

    with col2:
        st.markdown("**Todas las tasas equivalentes**")
        st.caption("Producen la misma acumulación en 1 año que la tasa ingresada.")

        df_eq = equivalent_rates(i_eff_r)
        st.dataframe(df_eq, hide_index=True, use_container_width=True)

        st.info(
            "💡 **Regla clave del FM**: antes de aplicar cualquier fórmula, "
            "todas las tasas involucradas deben estar en la **misma convención**. "
            "Esta tabla te da las equivalencias instantáneamente."
        )

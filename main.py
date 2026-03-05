# =============================================================================
# FIXED INCOME RISK ANALYZER — Backend FastAPI
# =============================================================================
# Este archivo es el servidor Python que hace TODOS los cálculos matemáticos.
# El frontend React lo llama por HTTP y solo se encarga del diseño.
#
# Arquitectura:
#   React (Vercel) → HTTP POST → FastAPI (Railway) → resultado JSON → React
#
# Para correr localmente:
#   pip install fastapi uvicorn
#   uvicorn main:app --reload --port 8000
#
# Endpoints disponibles:
#   POST /tvm          → Valor del Dinero en el Tiempo
#   POST /annuity      → Valuación de Anualidades
#   POST /amortization → Tabla de Amortización
#   POST /rates        → Conversión de Tasas
#   GET  /health       → Health check (para Railway)
# =============================================================================

import math
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN DE LA APP
# =============================================================================
# FastAPI es el framework web. CORS es necesario para que el frontend React
# (en otro dominio) pueda hacer llamadas HTTP al backend.
# =============================================================================

app = FastAPI(
    title="Fixed Income Risk Analyzer API",
    description="Backend de cálculos actuariales para el examen FM (SOA/CAS)",
    version="1.0.0",
)

# CORS (Cross-Origin Resource Sharing):
# Sin esto, el navegador bloquea las llamadas de React → FastAPI porque
# están en dominios diferentes (vercel.app → railway.app).
# En producción deberías restringir allow_origins a tu dominio de Vercel.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # En producción: ["https://tu-app.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# SECCIÓN 2: MODELOS DE REQUEST Y RESPONSE (Pydantic)
# =============================================================================
# Pydantic valida automáticamente los datos que llegan del frontend.
# Si React manda un string donde se espera un float, FastAPI retorna
# un error 422 claro en lugar de un crash silencioso.
# =============================================================================

# ── TVM ──
class TVMRequest(BaseModel):
    """Parámetros para calcular FV o PV de un capital único."""
    mode: str           # "fv" | "pv"
    pv: Optional[float] = None
    fv: Optional[float] = None
    rate: float         # tasa en decimal (ej: 0.05 para 5%)
    n: float            # períodos
    conv: str           # "nominal" | "effective" | "continuous"
    m: int = 12         # capitalización (solo si conv="nominal")

class TVMResponse(BaseModel):
    value: float
    label: str
    detail: str
    i_eff: float
    i_nom: float
    delta: float

# ── Anualidades ──
class AnnuityRequest(BaseModel):
    """Parámetros para calcular PV de una anualidad."""
    type: str           # "immediate" | "due" | "deferred"
    pmt: float          # pago periódico
    i: float            # tasa efectiva anual (decimal)
    n: float            # número de pagos
    d: float = 0        # períodos de diferimiento (solo si type="deferred")

class AnnuityResponse(BaseModel):
    pv: float
    fv: float
    label: str
    formula: str
    total_pmts: float
    total_interest: float

# ── Amortización ──
class AmortRow(BaseModel):
    """Una fila de la tabla de amortización."""
    t: int
    pmt: float
    interest: float
    principal: float
    extra: float
    balance: float
    is_cancelled: bool

class AmortRequest(BaseModel):
    """
    Parámetros para generar la tabla de amortización.

    El usuario ingresa:
      - i: tasa efectiva ANUAL (decimal), ej: 0.06 para 6% anual
      - freq: frecuencia de pagos por año (12=mensual, 4=trimestral, etc.)
      - n: número de períodos EN LA FRECUENCIA ELEGIDA (ej: 24 meses)

    El backend convierte i_anual → i_periodo antes de calcular:
      i_periodo = (1 + i_anual)^(1/freq) - 1
    """
    pv: float
    i: float            # tasa efectiva ANUAL (decimal)
    n: int              # número de períodos en la frecuencia elegida
    freq: int = 12      # pagos por año: 12=mensual, 4=trimestral, 2=semestral, 1=anual
    scheme: str = "french"
    extra_map: dict = {}

class AmortResponse(BaseModel):
    rows: list[AmortRow]
    total_interest: float
    total_pmt: float
    periods: int

# ── Conversión de Tasas ──
class RatesRequest(BaseModel):
    """Tasa de entrada para calcular equivalencias."""
    rate: float         # tasa en decimal
    m: int = 12         # capitalización
    conv: str = "nominal"

class RateEquivalent(BaseModel):
    label: str
    value: float        # en decimal

class RatesResponse(BaseModel):
    i_eff: float
    delta: float
    equivalents: list[RateEquivalent]


# =============================================================================
# SECCIÓN 3: FUNCIONES MATEMÁTICAS
# =============================================================================
# Exactamente las mismas fórmulas que teníamos en JavaScript/TypeScript,
# ahora en Python puro. La ventaja: podemos usar math, numpy, scipy, etc.
# =============================================================================

def effective_rate(rate: float, m: int, conv: str = "nominal") -> float:
    """
    Convierte cualquier convención de tasa a tasa efectiva anual.
      - "nominal":    (1 + rate/m)^m − 1
      - "effective":  rate (sin transformación)
      - "continuous": e^rate − 1
    """
    if conv == "continuous":
        return math.exp(rate) - 1
    elif conv == "effective":
        return rate
    return (1 + rate / m) ** m - 1


def calc_fv(pv: float, i: float, n: float) -> float:
    """FV = PV · (1 + i)^n"""
    return pv * (1 + i) ** n


def calc_pv(fv: float, i: float, n: float) -> float:
    """PV = FV · v^n   donde v = 1/(1+i)"""
    return fv / (1 + i) ** n


def annuity_immediate(pmt: float, i: float, n: float) -> float:
    """PV de anualidad vencida: PMT · (1 − v^n) / i"""
    if i == 0:
        return pmt * n
    return pmt * (1 - (1 + i) ** (-n)) / i


def annuity_due(pmt: float, i: float, n: float) -> float:
    """PV de anualidad anticipada: (1+i) · a⌐n|i"""
    return annuity_immediate(pmt, i, n) * (1 + i)


def annuity_deferred(pmt: float, i: float, n: float, d: float) -> float:
    """PV de anualidad diferida: v^d · a⌐n|i"""
    return annuity_immediate(pmt, i, n) * (1 + i) ** (-d)


def build_amort_schedule(
    pv: float,
    i: float,
    n: int,
    scheme: str = "french",
    extra_map: dict = {}
) -> list[dict]:
    """
    Genera la tabla de amortización con soporte para abonos extraordinarios.

    extra_map: las claves vienen como strings desde JSON ("3": 500)
               por eso convertimos a int antes de usarlas.
    """
    # Convertir claves de string a int (JSON siempre manda strings como claves)
    extras = {int(k): float(v) for k, v in extra_map.items()}
    rows = []

    if scheme == "french":
        pmt       = (pv * i) / (1 - (1 + i) ** (-n))
        balance   = pv
        remaining = n
        t         = 1

        while balance > 0.005 and t <= n + len(extras) + 1:
            interest  = balance * i
            principal = min(pmt - interest, balance)
            extra     = min(extras.get(t, 0), max(0, balance - principal))
            balance   = max(0, balance - principal - extra)

            rows.append({
                "t": t, "pmt": round(principal + interest, 2),
                "interest": round(interest, 2), "principal": round(principal, 2),
                "extra": round(extra, 2), "balance": round(balance, 2),
                "is_cancelled": balance < 0.005,
            })

            remaining -= 1
            if extra > 0 and balance > 0.005 and remaining > 0:
                pmt = (balance * i) / (1 - (1 + i) ** (-remaining))
            if balance < 0.005:
                break
            t += 1

    elif scheme == "german":
        base_principal = pv / n
        balance = pv
        for t in range(1, n + 1):
            if balance < 0.005:
                break
            interest  = balance * i
            principal = min(base_principal, balance)
            extra     = min(extras.get(t, 0), max(0, balance - principal))
            balance   = max(0, balance - principal - extra)
            rows.append({
                "t": t, "pmt": round(principal + interest, 2),
                "interest": round(interest, 2), "principal": round(principal, 2),
                "extra": round(extra, 2), "balance": round(balance, 2),
                "is_cancelled": balance < 0.005,
            })
            if balance < 0.005:
                break

    else:  # american / bullet
        balance = pv
        for t in range(1, n + 1):
            if balance < 0.005:
                break
            interest  = balance * i
            principal = balance if t == n else 0
            extra     = min(extras.get(t, 0), balance) if t < n else 0
            balance   = max(0, balance - principal - extra)
            rows.append({
                "t": t, "pmt": round(principal + interest, 2),
                "interest": round(interest, 2), "principal": round(principal, 2),
                "extra": round(extra, 2), "balance": round(balance, 2),
                "is_cancelled": balance < 0.005,
            })
            if balance < 0.005:
                break

    return rows


# =============================================================================
# SECCIÓN 4: ENDPOINTS
# =============================================================================
# Cada endpoint recibe un JSON del frontend, hace el cálculo y devuelve
# un JSON con el resultado. FastAPI maneja automáticamente la
# serialización/deserialización gracias a Pydantic.
# =============================================================================

@app.get("/health")
def health_check():
    """
    Health check para Railway.
    Railway llama a este endpoint para saber si el servidor está vivo.
    """
    return {"status": "ok", "service": "Fixed Income Risk Analyzer API"}


@app.post("/tvm", response_model=TVMResponse)
def calculate_tvm(req: TVMRequest):
    """
    Calcula FV o PV de un capital único.
    React llama: POST /tvm con { mode, pv/fv, rate, n, conv, m }
    """
    i_eff = effective_rate(req.rate, req.m, req.conv)
    delta = math.log(1 + i_eff)

    if req.mode == "fv":
        value  = calc_fv(req.pv or 0, i_eff, req.n)
        label  = "Future Value"
        detail = f"PV ${req.pv:,.2f} crece durante {req.n} períodos"
    else:
        value  = calc_pv(req.fv or 0, i_eff, req.n)
        label  = "Present Value"
        detail = f"FV ${req.fv:,.2f} descontado {req.n} períodos"

    return TVMResponse(
        value=round(value, 2),
        label=label,
        detail=detail,
        i_eff=i_eff,
        i_nom=req.rate,
        delta=delta,
    )


@app.post("/annuity", response_model=AnnuityResponse)
def calculate_annuity(req: AnnuityRequest):
    """
    Calcula PV y valor acumulado de una anualidad.
    React llama: POST /annuity con { type, pmt, i, n, d }
    """
    if req.type == "immediate":
        pv      = annuity_immediate(req.pmt, req.i, req.n)
        label   = "Anualidad Vencida — a⌐n|i"
        formula = "PV = PMT · (1 − vⁿ) / i"
    elif req.type == "due":
        pv      = annuity_due(req.pmt, req.i, req.n)
        label   = "Anualidad Anticipada — ä⌐n|i"
        formula = "PV = PMT · (1+i) · a⌐n|i"
    else:
        pv      = annuity_deferred(req.pmt, req.i, req.n, req.d)
        label   = f"Anualidad Diferida {int(req.d)} períodos"
        formula = f"PV = v^{int(req.d)} · PMT · a⌐{int(req.n)}|i"

    fv = calc_fv(pv, req.i, req.n + (req.d if req.type == "deferred" else 0))

    return AnnuityResponse(
        pv=round(pv, 2),
        fv=round(fv, 2),
        label=label,
        formula=formula,
        total_pmts=round(req.pmt * req.n, 2),
        total_interest=round(fv - req.pmt * req.n, 2),
    )


@app.post("/amortization", response_model=AmortResponse)
def calculate_amortization(req: AmortRequest):
    """
    Genera la tabla de amortización completa con abonos extraordinarios.
    React llama: POST /amortization con { pv, i, n, freq, scheme, extra_map }

    Conversión de tasa:
      i_anual  = 6%  →  i_mensual = (1.06)^(1/12) − 1 = 0.4868% por mes
    Esta conversión es la correcta para tasas efectivas (no nominales).
    """
    # Convertir tasa efectiva anual → tasa efectiva por período
    i_periodo = (1 + req.i) ** (1 / req.freq) - 1
    rows_raw = build_amort_schedule(req.pv, i_periodo, req.n, req.scheme, req.extra_map)

    rows = [AmortRow(**r) for r in rows_raw]

    return AmortResponse(
        rows=rows,
        total_interest=round(sum(r.interest for r in rows), 2),
        total_pmt=round(sum(r.pmt for r in rows), 2),
        periods=len(rows),
    )


@app.post("/rates", response_model=RatesResponse)
def calculate_rates(req: RatesRequest):
    """
    Calcula todas las tasas equivalentes a la tasa ingresada.
    React llama: POST /rates con { rate, m, conv }
    """
    i_eff = effective_rate(req.rate, req.m, req.conv)
    delta = math.log(1 + i_eff)

    equivalents = [
        RateEquivalent(label="Efectiva anual (m=1)",     value=(1 + i_eff) ** (1/1)   - 1),
        RateEquivalent(label="Nominal semestral (m=2)",  value=2   * ((1 + i_eff) ** (1/2)   - 1)),
        RateEquivalent(label="Nominal trimestral (m=4)", value=4   * ((1 + i_eff) ** (1/4)   - 1)),
        RateEquivalent(label="Nominal mensual (m=12)",   value=12  * ((1 + i_eff) ** (1/12)  - 1)),
        RateEquivalent(label="Nominal diaria (m=365)",   value=365 * ((1 + i_eff) ** (1/365) - 1)),
        RateEquivalent(label="Fuerza de interés δ",      value=delta),
    ]

    return RatesResponse(i_eff=i_eff, delta=delta, equivalents=equivalents)

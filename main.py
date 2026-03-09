# =============================================================================
# FIXED INCOME RISK ANALYZER — FastAPI Backend
# =============================================================================
# Run locally:  uvicorn main:app --reload
# Deploy:       Railway picks this up automatically via uvicorn main:app
#
# Endpoints:
#   POST /tvm           — Time Value of Money (PV / FV)
#   POST /annuity       — Annuities (immediate, due, deferred)
#   POST /amortization  — Amortization schedule with extra payments
#   POST /rates         — Interest rate conversions
# =============================================================================

import math
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Fixed Income Risk Analyzer — FM Exam")

# ---------------------------------------------------------------------------
# CORS — allows the Vercel frontend to call this API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, replace with your Vercel URL
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# SECTION 1: PURE MATH FUNCTIONS
# (identical logic to the Streamlit version, no UI dependencies)
# =============================================================================

def effective_rate(rate: float, m: int, conv: str = "nominal") -> float:
    """
    Converts any rate convention to the effective annual rate.

    FM defines three conventions:
      - "nominal":    i^(m) compounded m times/year → (1 + i/m)^m − 1
      - "effective":  already an effective annual rate → no transformation
      - "continuous": force of interest δ → e^δ − 1
    """
    if conv == "continuous":
        return math.exp(rate) - 1
    elif conv == "effective":
        return rate
    else:  # nominal
        return (1 + rate / m) ** m - 1


def annuity_immediate(pmt: float, i: float, n: float) -> float:
    """PV of annuity-immediate (payments at END of period). FM notation: a⌐n|i"""
    if i == 0:
        return pmt * n
    return pmt * (1 - (1 + i) ** (-n)) / i


def annuity_due(pmt: float, i: float, n: float) -> float:
    """PV of annuity-due (payments at START of period). FM notation: ä⌐n|i"""
    return annuity_immediate(pmt, i, n) * (1 + i)


def annuity_deferred(pmt: float, i: float, n: float, d: float) -> float:
    """PV of deferred annuity. FM formula: d|a⌐n|i = v^d · a⌐n|i"""
    return annuity_immediate(pmt, i, n) * (1 + i) ** (-d)


def build_amortization(
    pv: float,
    i: float,          # effective rate per period (already converted)
    n: int,
    scheme: str = "french",
    extra_map: dict = None
) -> list[dict]:
    """
    Builds the amortization table row by row.

    Schemes:
      french   — constant payment  PMT = PV·i / (1−(1+i)^−n)
      german   — constant principal K = PV/n
      american — interest-only; full principal at maturity (bullet)

    Extra payments reduce principal directly; French scheme recalculates
    the payment on the remaining balance after each extra payment.
    """
    if extra_map is None:
        extra_map = {}

    rows = []

    if scheme == "french":
        pmt       = (pv * i) / (1 - (1 + i) ** (-n))
        balance   = pv
        remaining = n
        t         = 1

        while balance > 0.005 and t <= n + len(extra_map) + 1:
            interest  = balance * i
            principal = min(pmt - interest, balance)
            extra     = min(extra_map.get(t, 0), max(0, balance - principal))
            balance   = max(0, balance - principal - extra)

            rows.append({
                "t":            t,
                "pmt":          round(principal + interest, 2),
                "interest":     round(interest, 2),
                "principal":    round(principal, 2),
                "extra":        round(extra, 2),
                "balance":      round(balance, 2),
                "is_cancelled": balance < 0.005,
            })

            remaining -= 1

            if extra > 0 and balance > 0.005 and remaining > 0:
                pmt = (balance * i) / (1 - (1 + i) ** (-remaining))

            if balance < 0.005:
                break
            t += 1

    elif scheme == "german":
        base_k  = pv / n
        balance = pv

        for t in range(1, n + 1):
            if balance < 0.005:
                break
            interest  = balance * i
            principal = min(base_k, balance)
            extra     = min(extra_map.get(t, 0), max(0, balance - principal))
            balance   = max(0, balance - principal - extra)

            rows.append({
                "t":            t,
                "pmt":          round(principal + interest, 2),
                "interest":     round(interest, 2),
                "principal":    round(principal, 2),
                "extra":        round(extra, 2),
                "balance":      round(balance, 2),
                "is_cancelled": balance < 0.005,
            })

    else:  # american / bullet
        balance = pv

        for t in range(1, n + 1):
            if balance < 0.005:
                break
            interest  = balance * i
            principal = balance if t == n else 0
            extra     = min(extra_map.get(t, 0), balance) if t < n else 0
            balance   = max(0, balance - principal - extra)

            rows.append({
                "t":            t,
                "pmt":          round(principal + interest, 2),
                "interest":     round(interest, 2),
                "principal":    round(principal, 2),
                "extra":        round(extra, 2),
                "balance":      round(balance, 2),
                "is_cancelled": balance < 0.005,
            })

    return rows


# =============================================================================
# SECTION 2: PYDANTIC REQUEST MODELS
# =============================================================================

class TVMRequest(BaseModel):
    mode: str               # "fv" | "pv"
    pv:   Optional[float] = None
    fv:   Optional[float] = None
    rate: float             # decimal (e.g. 0.05 for 5%)
    n:    float
    conv: str = "nominal"   # "nominal" | "effective" | "continuous"
    m:    int  = 12


class AnnuityRequest(BaseModel):
    type: str               # "immediate" | "due" | "deferred"
    pmt:  float
    i:    float             # effective annual rate in decimal
    n:    float
    freq: int  = 1          # payments per year (1=annual, 12=monthly, etc.)
    d:    float = 0         # deferral periods (only for deferred annuities)


class AmortRequest(BaseModel):
    pv:        float
    i:         float        # effective annual rate in decimal
    n:         int
    freq:      int  = 12    # payments per year
    scheme:    str  = "french"
    extra_map: dict = {}


class RatesRequest(BaseModel):
    rate: float             # decimal
    m:    int  = 12
    conv: str  = "nominal"


# =============================================================================
# SECTION 3: ENDPOINTS
# =============================================================================

@app.get("/")
def root():
    return {"status": "ok", "api": "Fixed Income Risk Analyzer — FM"}


# ---------------------------------------------------------------------------
# POST /tvm — Time Value of Money
# ---------------------------------------------------------------------------
@app.post("/tvm")
def tvm(req: TVMRequest):
    try:
        i_eff = effective_rate(req.rate, req.m, req.conv)
        delta = math.log(1 + i_eff)
        i_nom = req.m * ((1 + i_eff) ** (1 / req.m) - 1)  # nominal m=12

        if req.mode == "fv":
            if req.pv is None:
                raise HTTPException(status_code=422, detail="pv is required when mode='fv'")
            value = req.pv * (1 + i_eff) ** req.n
            label = "Future Value"
            detail = f"PV {req.pv:,.2f} grows over {req.n} periods"
        else:
            if req.fv is None:
                raise HTTPException(status_code=422, detail="fv is required when mode='pv'")
            value = req.fv / (1 + i_eff) ** req.n
            label = "Present Value"
            detail = f"FV {req.fv:,.2f} discounted {req.n} periods"

        return {
            "value":  round(value, 6),
            "label":  label,
            "detail": detail,
            "i_eff":  i_eff,
            "i_nom":  i_nom,
            "delta":  delta,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# POST /annuity — Annuity valuation
# ---------------------------------------------------------------------------

# Maps payments-per-year → human-readable label
FREQ_LABEL = {12: "mensual", 6: "bimestral", 4: "trimestral", 2: "semestral", 1: "anual"}

@app.post("/annuity")
def annuity(req: AnnuityRequest):
    try:
        # Convert effective annual rate to the rate per payment period
        i_periodo = (1 + req.i) ** (1 / req.freq) - 1

        if req.type == "immediate":
            pv    = annuity_immediate(req.pmt, i_periodo, req.n)
            label = "Annuity-Immediate (vencida)"
            formula = f"PV = PMT · (1 − v^n) / i_periodo  [{req.n} pagos]"
        elif req.type == "due":
            pv    = annuity_due(req.pmt, i_periodo, req.n)
            label = "Annuity-Due (anticipada)"
            formula = f"PV = PMT · ä⌐n|i = a⌐n|i · (1 + i_periodo)  [{req.n} pagos]"
        elif req.type == "deferred":
            pv    = annuity_deferred(req.pmt, i_periodo, req.n, req.d)
            label = f"Deferred Annuity (diferida {int(req.d)} períodos)"
            formula = f"PV = v^d · PMT · a⌐n|i  [d={req.d}, n={req.n}]"
        else:
            raise HTTPException(status_code=422, detail=f"Unknown annuity type: {req.type}")

        fv             = pv * (1 + i_periodo) ** req.n
        total_pmts     = req.pmt * req.n
        total_interest = fv - total_pmts
        freq_label     = FREQ_LABEL.get(req.freq, f"cada {12 // req.freq} meses")

        return {
            "pv":             round(pv, 6),
            "fv":             round(fv, 6),
            "label":          label,
            "formula":        formula,
            "total_pmts":     round(total_pmts, 2),
            "total_interest": round(total_interest, 2),
            "freq_label":     freq_label,
            "i_periodo":      i_periodo,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# POST /amortization — Amortization schedule
# ---------------------------------------------------------------------------
@app.post("/amortization")
def amortization(req: AmortRequest):
    try:
        # Convert effective annual rate → rate per payment period
        i_periodo = (1 + req.i) ** (1 / req.freq) - 1

        rows = build_amortization(req.pv, i_periodo, req.n, req.scheme, req.extra_map)

        total_interest = sum(r["interest"] for r in rows)
        total_pmt      = sum(r["pmt"] + r["extra"] for r in rows)

        return {
            "rows":           rows,
            "total_interest": round(total_interest, 2),
            "total_pmt":      round(total_pmt, 2),
            "periods":        len(rows),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# POST /rates — Interest rate conversions
# ---------------------------------------------------------------------------
@app.post("/rates")
def rates(req: RatesRequest):
    try:
        i_eff = effective_rate(req.rate, req.m, req.conv)
        delta = math.log(1 + i_eff)

        equivalents = [
            {"label": "Efectiva anual (m=1)",      "value": i_eff},
            {"label": "Nominal semestral (m=2)",    "value": 2   * ((1 + i_eff) ** (1/2)   - 1)},
            {"label": "Nominal trimestral (m=4)",   "value": 4   * ((1 + i_eff) ** (1/4)   - 1)},
            {"label": "Nominal mensual (m=12)",     "value": 12  * ((1 + i_eff) ** (1/12)  - 1)},
            {"label": "Nominal diaria (m=365)",     "value": 365 * ((1 + i_eff) ** (1/365) - 1)},
            {"label": "Fuerza de interés δ",        "value": delta},
        ]

        return {
            "i_eff":       i_eff,
            "delta":       delta,
            "equivalents": equivalents,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

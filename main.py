# =============================================================================
# FIXED INCOME RISK ANALYZER — FastAPI Backend
# =============================================================================
# Run locally:  uvicorn main:app --reload
# Deploy:       Railway picks this up automatically via uvicorn main:app
#
# Endpoints:
#   POST /tvm           — Time Value of Money (PV / FV)
#   POST /annuity       — Annuities: solve for PV, FV, PMT, i, or n
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
# =============================================================================

def effective_rate(rate: float, m: int, conv: str = "nominal") -> float:
    """
    Converts any rate convention to the effective annual rate.
      - "nominal":    i^(m) compounded m times/year → (1 + i/m)^m − 1
      - "effective":  already effective annual  → no transformation
      - "continuous": force of interest δ       → e^δ − 1
    """
    if conv == "continuous":
        return math.exp(rate) - 1
    elif conv == "effective":
        return rate
    else:
        return (1 + rate / m) ** m - 1


# ── Annuity PV helpers ───────────────────────────────────────────────────────

def annuity_immediate(pmt: float, i: float, n: float) -> float:
    """PV of annuity-immediate (end-of-period payments). FM notation: a⌐n|i"""
    if i == 0:
        return pmt * n
    return pmt * (1 - (1 + i) ** (-n)) / i


def annuity_due(pmt: float, i: float, n: float) -> float:
    """PV of annuity-due (beginning-of-period payments). FM notation: ä⌐n|i"""
    return annuity_immediate(pmt, i, n) * (1 + i)


def annuity_deferred(pmt: float, i: float, n: float, d: float) -> float:
    """PV of deferred annuity. FM formula: d|a⌐n|i = v^d · a⌐n|i"""
    return annuity_immediate(pmt, i, n) * (1 + i) ** (-d)


# ── Annuity FV helpers ───────────────────────────────────────────────────────

def fv_immediate(pmt: float, i: float, n: float) -> float:
    """FV of annuity-immediate at time n. FM notation: s⌐n|i = ((1+i)^n − 1)/i"""
    if i == 0:
        return pmt * n
    return pmt * ((1 + i) ** n - 1) / i


def fv_due(pmt: float, i: float, n: float) -> float:
    """FV of annuity-due at time n. FM notation: s̈⌐n|i = s⌐n|i · (1+i)"""
    return fv_immediate(pmt, i, n) * (1 + i)


# ── Unified dispatch ─────────────────────────────────────────────────────────

def _pv(pmt: float, i_p: float, n: float, ann_type: str, d: float) -> float:
    """PV at t=0 for any annuity type using the per-period rate i_p."""
    if ann_type == "immediate":
        return annuity_immediate(pmt, i_p, n)
    elif ann_type == "due":
        return annuity_due(pmt, i_p, n)
    else:
        return annuity_deferred(pmt, i_p, n, d)


def _fv(pmt: float, i_p: float, n: float, ann_type: str) -> float:
    """
    FV at the end of the payment stream using the per-period rate i_p.
    For a deferred annuity the FV is at time d+n (value at last payment).
    """
    if ann_type == "immediate":
        return fv_immediate(pmt, i_p, n)
    elif ann_type == "due":
        return fv_due(pmt, i_p, n)
    else:
        return fv_immediate(pmt, i_p, n)   # deferred: same FV formula as immediate


# ── Numerical solver: interest rate (bisection) ──────────────────────────────

def _solve_rate(
    target: float, pmt: float, n: float, ann_type: str, d: float, ref: str
) -> float:
    """
    Find i_period such that PV(i) = target  (ref='pv')
                         or FV(i) = target  (ref='fv').

    Method: bisection over [1e-8, 50] per period (up to ~5000 %/period).
    Raises ValueError when no solution exists in that range.
    """
    def f(i_p: float) -> float:
        if ref == "pv":
            return _pv(pmt, i_p, n, ann_type, d) - target
        else:
            return _fv(pmt, i_p, n, ann_type) - target

    lo, hi = 1e-8, 50.0
    f_lo, f_hi = f(lo), f(hi)

    if f_lo * f_hi > 0:
        raise ValueError(
            "No valid interest rate exists for the given inputs. "
            "Verify that the reference PV/FV is achievable with the provided PMT and n."
        )

    for _ in range(400):
        mid = (lo + hi) / 2.0
        fm  = f(mid)
        if abs(fm) < 1e-12 or (hi - lo) < 1e-14:
            break
        if f_lo * fm < 0:
            hi = mid; f_hi = fm
        else:
            lo = mid; f_lo = fm

    return (lo + hi) / 2.0


# ── Analytical solver: number of periods ────────────────────────────────────

def _solve_n(
    target: float, pmt: float, i_p: float, ann_type: str, d: float, ref: str
) -> float:
    """
    Solve for n analytically.

    From PV, annuity-immediate:  n = −ln(1 − PV·i/PMT) / ln(1+i)
    From FV, annuity-immediate:  n =  ln(1 + FV·i/PMT) / ln(1+i)

    Annuity-due and deferred cases adjust the target before applying the
    same log formula. Raises ValueError when no finite n exists.
    """
    if i_p == 0:
        return target / pmt

    if ref == "pv":
        if ann_type == "immediate":
            x = 1.0 - target * i_p / pmt
        elif ann_type == "due":
            # PV_due = PMT · a⌐n|i · (1+i)  →  a⌐n|i_factor = PV / (PMT·(1+i))
            x = 1.0 - (target / (pmt * (1 + i_p))) * i_p
        else:   # deferred
            # PV_def = v^d · PMT · a⌐n|i  →  a_factor = PV·(1+i)^d / PMT
            x = 1.0 - (target * (1 + i_p) ** d / pmt) * i_p
        if x <= 0:
            raise ValueError(
                "PMT does not cover the periodic interest charge; no finite n exists. "
                "Increase PMT or decrease the interest rate."
            )
        return -math.log(x) / math.log(1 + i_p)

    else:   # ref == "fv"
        if ann_type == "immediate":
            x = 1.0 + target * i_p / pmt
        elif ann_type == "due":
            # FV_due = PMT · s⌐n|i · (1+i)  →  s⌐n|i_factor = FV / (PMT·(1+i))
            x = 1.0 + (target / (pmt * (1 + i_p))) * i_p
        else:   # deferred: same FV formula as immediate
            x = 1.0 + target * i_p / pmt
        if x <= 0:
            raise ValueError("Invalid FV / PMT combination.")
        return math.log(x) / math.log(1 + i_p)


# ── Validation helpers ───────────────────────────────────────────────────────

def _require(val, name: str):
    if val is None:
        raise HTTPException(
            status_code=422,
            detail=f"'{name}' is required for this calculation."
        )


def _label_formula(
    solve: str, ann_type: str, ref: str, n: float, d: float
) -> tuple[str, str]:
    """Return a human-readable (label, FM formula string) pair."""
    base = {"immediate": "Annuity-Immediate",
            "due":       "Annuity-Due",
            "deferred":  "Deferred Annuity"}.get(ann_type, ann_type)

    pv_fmls = {
        "immediate": "PV = PMT · a⌐n|i = PMT · (1 − vⁿ) / i",
        "due":       "PV = PMT · ä⌐n|i = PMT · (1 − vⁿ) · (1+i) / i",
        "deferred":  f"PV = v^d · PMT · a⌐n|i   [d={int(d)}, n={int(round(n))}]",
    }
    fv_fmls = {
        "immediate": "FV = PMT · s⌐n|i = PMT · ((1+i)ⁿ − 1) / i",
        "due":       "FV = PMT · s̈⌐n|i = PMT · ((1+i)ⁿ − 1) · (1+i) / i",
        "deferred":  "FV = PMT · s⌐n|i  (accumulated value at end of payment stream)",
    }

    if solve == "pv":
        return f"{base} — PV", pv_fmls.get(ann_type, "")
    if solve == "fv":
        return f"{base} — FV", fv_fmls.get(ann_type, "")

    src    = "PV" if ref == "pv" else "FV"
    a_sym  = ("ä⌐n|i" if ann_type == "due" else "a⌐n|i") if ref == "pv" else \
             ("s̈⌐n|i" if ann_type == "due" else "s⌐n|i")

    if solve == "pmt":
        return f"{base} — solve PMT", f"PMT = {src} / {a_sym}"
    if solve == "i":
        return (f"{base} — solve i",
                f"Bisection: find i s.t. {src}(PMT, i, n) = {src}_given")
    if solve == "n":
        if ref == "pv":
            return (f"{base} — solve n",
                    "n = −ln(1 − PV·i / PMT) / ln(1+i)")
        else:
            return (f"{base} — solve n",
                    "n = ln(1 + FV·i / PMT) / ln(1+i)")

    return base, ""


# ── Amortization table ───────────────────────────────────────────────────────

def build_amortization(
    pv: float, i: float, n: int,
    scheme: str = "french", extra_map: dict = None
) -> list[dict]:
    if extra_map is None:
        extra_map = {}
    extra_map = {int(k): v for k, v in extra_map.items()}
    rows = []

    if scheme == "french":
        pmt = (pv * i) / (1 - (1 + i) ** (-n))
        balance = pv
        t = 1
        while balance > 0.005 and t <= n + len(extra_map) + 1:
            interest  = balance * i
            principal = min(pmt - interest, balance)
            extra     = min(extra_map.get(t, 0), max(0, balance - principal))
            balance   = max(0, balance - principal - extra)
            rows.append({"t": t, "pmt": round(principal + interest, 2),
                         "interest": round(interest, 2), "principal": round(principal, 2),
                         "extra": round(extra, 2), "balance": round(balance, 2),
                         "is_cancelled": balance < 0.005})
            if balance < 0.005:
                break
            t += 1

    elif scheme == "german":
        base_k = pv / n
        balance = pv
        for t in range(1, n + 1):
            if balance < 0.005:
                break
            interest  = balance * i
            principal = min(base_k, balance)
            extra     = min(extra_map.get(t, 0), max(0, balance - principal))
            balance   = max(0, balance - principal - extra)
            rows.append({"t": t, "pmt": round(principal + interest, 2),
                         "interest": round(interest, 2), "principal": round(principal, 2),
                         "extra": round(extra, 2), "balance": round(balance, 2),
                         "is_cancelled": balance < 0.005})

    else:  # american / bullet
        balance = pv
        for t in range(1, n + 1):
            if balance < 0.005:
                break
            interest  = balance * i
            principal = balance if t == n else 0
            extra     = min(extra_map.get(t, 0), balance) if t < n else 0
            balance   = max(0, balance - principal - extra)
            rows.append({"t": t, "pmt": round(principal + interest, 2),
                         "interest": round(interest, 2), "principal": round(principal, 2),
                         "extra": round(extra, 2), "balance": round(balance, 2),
                         "is_cancelled": balance < 0.005})
    return rows


# =============================================================================
# SECTION 2: PYDANTIC REQUEST MODELS
# =============================================================================

class TVMRequest(BaseModel):
    mode: str
    pv:   Optional[float] = None
    fv:   Optional[float] = None
    rate: float
    n:    float
    conv: str = "nominal"
    m:    int  = 12


class AnnuityRequest(BaseModel):
    type:  str                     # "immediate" | "due" | "deferred"
    solve: str        = "pv"      # "pv" | "fv" | "pmt" | "i" | "n"
    ref:   str        = "pv"      # reference variable when solving pmt / i / n
    # Inputs (None = being solved for or not needed)
    pmt:   Optional[float] = None
    i:     Optional[float] = None  # effective annual rate (decimal)
    n:     Optional[float] = None
    freq:  int        = 1          # payments per year
    d:     float      = 0.0        # deferral periods
    pv:    Optional[float] = None  # reference PV
    fv:    Optional[float] = None  # reference FV


class AmortRequest(BaseModel):
    pv:        float
    i:         float
    n:         int
    freq:      int  = 12
    scheme:    str  = "french"
    extra_map: dict = {}


class RatesRequest(BaseModel):
    rate: float
    m:    int  = 12
    conv: str  = "nominal"


# =============================================================================
# SECTION 3: ENDPOINTS
# =============================================================================

@app.get("/")
def root():
    return {"status": "ok", "api": "Fixed Income Risk Analyzer — FM"}


@app.post("/tvm")
def tvm(req: TVMRequest):
    try:
        i_eff = effective_rate(req.rate, req.m, req.conv)
        delta = math.log(1 + i_eff)
        i_nom = req.m * ((1 + i_eff) ** (1 / req.m) - 1)

        if req.mode == "fv":
            if req.pv is None:
                raise HTTPException(422, "pv is required when mode='fv'")
            value = req.pv * (1 + i_eff) ** req.n
            label, detail = "Future Value", f"PV {req.pv:,.2f} grows over {req.n} periods"
        else:
            if req.fv is None:
                raise HTTPException(422, "fv is required when mode='pv'")
            value = req.fv / (1 + i_eff) ** req.n
            label, detail = "Present Value", f"FV {req.fv:,.2f} discounted {req.n} periods"

        return {"value": round(value, 6), "label": label, "detail": detail,
                "i_eff": i_eff, "i_nom": i_nom, "delta": delta}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------------------
# POST /annuity  — Full 5-variable solver
# ---------------------------------------------------------------------------
FREQ_LABEL = {12: "mensual", 6: "bimestral", 4: "trimestral", 2: "semestral", 1: "anual"}

@app.post("/annuity")
def annuity(req: AnnuityRequest):
    try:
        solve    = req.solve
        ann_type = req.type
        freq     = req.freq
        d        = req.d

        # ── Compute all five variables ───────────────────────────────────────

        if solve == "pv":
            _require(req.pmt, "pmt"); _require(req.i, "i"); _require(req.n, "n")
            i_p = (1 + req.i) ** (1 / freq) - 1
            pmt, n = req.pmt, req.n
            pv = _pv(pmt, i_p, n, ann_type, d)
            fv = _fv(pmt, i_p, n, ann_type)

        elif solve == "fv":
            _require(req.pmt, "pmt"); _require(req.i, "i"); _require(req.n, "n")
            i_p = (1 + req.i) ** (1 / freq) - 1
            pmt, n = req.pmt, req.n
            fv = _fv(pmt, i_p, n, ann_type)
            pv = _pv(pmt, i_p, n, ann_type, d)

        elif solve == "pmt":
            _require(req.i, "i"); _require(req.n, "n")
            i_p = (1 + req.i) ** (1 / freq) - 1
            n   = req.n
            if req.ref == "pv":
                _require(req.pv, "pv")
                factor = _pv(1.0, i_p, n, ann_type, d)
                if abs(factor) < 1e-14:
                    raise ValueError("Annuity factor is zero; check inputs.")
                pmt = req.pv / factor
            else:
                _require(req.fv, "fv")
                factor = _fv(1.0, i_p, n, ann_type)
                if abs(factor) < 1e-14:
                    raise ValueError("Annuity factor is zero; check inputs.")
                pmt = req.fv / factor
            pv = _pv(pmt, i_p, n, ann_type, d)
            fv = _fv(pmt, i_p, n, ann_type)

        elif solve == "i":
            _require(req.pmt, "pmt"); _require(req.n, "n")
            pmt, n = req.pmt, req.n
            target = req.pv if req.ref == "pv" else req.fv
            if target is None:
                _require(None, "pv" if req.ref == "pv" else "fv")
            i_p = _solve_rate(target, pmt, n, ann_type, d, req.ref)
            pv  = _pv(pmt, i_p, n, ann_type, d)
            fv  = _fv(pmt, i_p, n, ann_type)

        elif solve == "n":
            _require(req.pmt, "pmt"); _require(req.i, "i")
            i_p = (1 + req.i) ** (1 / freq) - 1
            pmt = req.pmt
            target = req.pv if req.ref == "pv" else req.fv
            if target is None:
                _require(None, "pv" if req.ref == "pv" else "fv")
            n   = _solve_n(target, pmt, i_p, ann_type, d, req.ref)
            pv  = _pv(pmt, i_p, n, ann_type, d)
            fv  = _fv(pmt, i_p, n, ann_type)

        else:
            raise HTTPException(422, f"Unknown solve target: '{solve}'")

        # ── Derived quantities ───────────────────────────────────────────────
        i_annual       = (1 + i_p) ** freq - 1
        delta          = math.log(1 + i_annual) if i_annual > -1 else 0.0
        freq_label     = FREQ_LABEL.get(freq, f"cada {12 // max(freq, 1)} meses")
        total_pmts     = pmt * n
        total_interest = fv - total_pmts
        label, formula = _label_formula(solve, ann_type, req.ref, n, d)

        return {
            "solved":         solve,
            "pv":             round(pv, 6),
            "fv":             round(fv, 6),
            "pmt":            round(pmt, 6),
            "i_annual":       i_annual,
            "i_period":       i_p,
            "n":              round(n, 6),
            "delta":          delta,
            "label":          label,
            "formula":        formula,
            "freq_label":     freq_label,
            "total_pmts":     round(total_pmts, 2),
            "total_interest": round(total_interest, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/amortization")
def amortization(req: AmortRequest):
    try:
        i_periodo = (1 + req.i) ** (1 / req.freq) - 1
        rows = build_amortization(req.pv, i_periodo, req.n, req.scheme, req.extra_map)
        return {
            "rows":           rows,
            "total_interest": round(sum(r["interest"] for r in rows), 2),
            "total_pmt":      round(sum(r["pmt"] + r["extra"] for r in rows), 2),
            "periods":        len(rows),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/rates")
def rates(req: RatesRequest):
    try:
        i_eff = effective_rate(req.rate, req.m, req.conv)
        delta = math.log(1 + i_eff)
        return {
            "i_eff": i_eff,
            "delta": delta,
            "equivalents": [
                {"label": "Efectiva anual (m=1)",      "value": i_eff},
                {"label": "Nominal semestral (m=2)",    "value": 2   * ((1 + i_eff) ** (1/2)   - 1)},
                {"label": "Nominal trimestral (m=4)",   "value": 4   * ((1 + i_eff) ** (1/4)   - 1)},
                {"label": "Nominal mensual (m=12)",     "value": 12  * ((1 + i_eff) ** (1/12)  - 1)},
                {"label": "Nominal diaria (m=365)",     "value": 365 * ((1 + i_eff) ** (1/365) - 1)},
                {"label": "Fuerza de interés δ",        "value": delta},
            ],
        }
    except Exception as e:
        raise HTTPException(400, str(e))

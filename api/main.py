"""
Project Sentinel — SOFR Pricing Engine REST API
================================================
FastAPI server exposing the core pricing and analytics modules.

Usage
-----
    uvicorn api.main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
OpenAPI spec:     http://localhost:8000/openapi.json

Endpoints
---------
GET  /health                      — liveness check
GET  /curve/zero                  — zero curve for a flat SOFR level
POST /curve/bootstrap             — full curve from market quotes
POST /price/swap                  — price a SOFR OIS swap
POST /price/fra                   — price a forward rate agreement
POST /risk/scenario               — parallel / twist shift P&L
POST /risk/dv01                   — DV01 ladder for a swap portfolio
GET  /carry/table                 — carry + roll-down table
POST /signals/composite           — composite macro trading signal
POST /taylor/rate                 — Taylor Rule recommended rate
GET  /fomc/probabilities          — FOMC meeting outcome probabilities
POST /pca/scores                  — project yield curve into PC space
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Engine imports ─────────────────────────────────────────────────────────────
from sofr_engine.bootstrap import flat_sofr_curve, SOFRCurveBootstrapper
from sofr_engine.instruments import SOFRSwap, ForwardRateAgreement
from sofr_engine.risk import ScenarioEngine, RiskReport
from sofr_engine.convexity import hull_white_convexity_adjustment
from models.taylor_rule import compute_taylor_rule, TaylorRuleConfig
from models.fomc_probability import fedwatch_probabilities, fomc_prob_summary
from models.macro_signals import composite_signal
from models.carry_rolldown import carry_rolldown_table, steepener_carry
from dateutil.relativedelta import relativedelta

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Project Sentinel — SOFR Pricing Engine",
    description=(
        "Production-grade SOFR forward curve bootstrapper, OIS swap pricer, "
        "risk analytics, and macro signal API. "
        "Built with Hull-White convexity adjustment and walk-forward validation."
    ),
    version="0.2.0",
    contact={
        "name": "Bhavesh Anchalia",
        "email": "anchaliabhavesh1@gmail.com",
    },
)


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "version": "0.2.0", "engine": "Project Sentinel"}


# ── Pydantic models ────────────────────────────────────────────────────────────

class OISQuote(BaseModel):
    tenor_years: float = Field(..., ge=0.1, le=50.0, description="Swap tenor in years")
    rate:        float = Field(..., ge=0.0, le=0.20, description="Par OIS rate (decimal)")


class FuturesQuote(BaseModel):
    expiry_iso:      str   = Field(..., description="Futures expiry date (YYYY-MM-DD)")
    accrual_end_iso: str   = Field(..., description="End of accrual period (YYYY-MM-DD)")
    implied_rate:    float = Field(..., ge=0.0, le=0.20, description="Implied futures rate (decimal)")


class BootstrapRequest(BaseModel):
    ref_date_iso:  str            = Field(..., description="Curve date (YYYY-MM-DD)")
    sofr_on:       float          = Field(..., ge=0.0, le=0.20, description="Overnight SOFR (decimal)")
    futures:       list[FuturesQuote] = Field(default_factory=list)
    ois_quotes:    list[OISQuote]     = Field(default_factory=list)
    hw_sigma:      float          = Field(0.010, description="Hull-White σ for convexity adj")
    hw_mean_rev:   float          = Field(0.050, description="Hull-White mean reversion (a)")


class SwapRequest(BaseModel):
    effective_date_iso: str   = Field(..., description="Swap effective date (YYYY-MM-DD)")
    maturity_date_iso:  str   = Field(..., description="Swap maturity date (YYYY-MM-DD)")
    fixed_rate:         float = Field(..., ge=0.0, le=0.20, description="Fixed coupon (decimal)")
    notional:           float = Field(10_000_000.0, description="Notional in USD")
    pay_fixed:          bool  = Field(True, description="True = payer swap")
    sofr_on:            float = Field(0.0358, description="Curve level for flat curve bootstrap")


class FRARequest(BaseModel):
    start_date_iso: str   = Field(..., description="FRA start date")
    end_date_iso:   str   = Field(..., description="FRA end date")
    fixed_rate:     float = Field(..., ge=0.0, le=0.20)
    notional:       float = Field(10_000_000.0)
    pay_receive:    Literal["pay", "receive"] = "receive"
    sofr_on:        float = Field(0.0358)


class ScenarioRequest(BaseModel):
    sofr_on:        float = Field(0.0358, description="Base curve level")
    swap_tenor_yrs: float = Field(10.0, description="Tenor of test swap")
    swap_notional:  float = Field(10_000_000.0)
    shift_type:     Literal["parallel", "twist", "key_rate"] = "parallel"
    shift1_bps:     float = Field(0.0, description="Primary shift (parallel/short)")
    shift2_bps:     float = Field(0.0, description="Long-end shift (twist only)")
    key_tenor:      float = Field(5.0, description="Key-rate tenor (key_rate only)")


class DV01Request(BaseModel):
    sofr_on:    float             = Field(0.0358)
    swaps:      list[dict]        = Field(..., description="[{tenor_y, notional, pay_fixed}]")


class SignalRequest(BaseModel):
    core_pce_yoy:  float = Field(2.11, description="Core PCE YoY (%)")
    unemployment:  float = Field(4.30, description="Unemployment rate (%)")
    tsy_2y:        float = Field(4.50, description="2Y Treasury yield (%)")
    tsy_10y:       float = Field(4.35, description="10Y Treasury yield (%)")
    effr:          float = Field(4.33, description="Effective Federal Funds Rate (%)")
    taylor_rate:   float = Field(2.26, description="Taylor Rule recommended rate (%)")


class TaylorRequest(BaseModel):
    inflation:      float = Field(2.11, description="Core PCE inflation (%)")
    unemployment:   float = Field(4.30, description="Unemployment rate (%)")
    neutral_rate:   float = Field(2.50, description="Neutral rate r* (%)")
    inflation_target: float = Field(2.0)
    nairu:          float = Field(4.0)
    variant:        Literal["standard", "balanced", "inertial"] = "standard"


class FOMCRequest(BaseModel):
    current_rate:  float = Field(0.0433, description="Current EFFR (decimal)")
    implied_rate:  float = Field(0.0408, description="Futures-implied rate (decimal)")


class CarryRequest(BaseModel):
    sofr_on: float = Field(0.0358, description="Flat curve level")
    dt_months: float = Field(1.0, description="Holding period in months")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _flat_curve(sofr_on: float):
    return flat_sofr_curve(date.today(), sofr_on)


def _sanitize(v):
    """Replace NaN/Inf floats with None for JSON safety."""
    import math
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _sanitize_row(row: dict) -> dict:
    return {k: _sanitize(v) for k, v in row.items()}


def _bootstrap_curve(req: BootstrapRequest):
    ref   = date.fromisoformat(req.ref_date_iso)
    futs  = pd.DataFrame([{
        "expiry":       date.fromisoformat(f.expiry_iso),
        "accrual_end":  date.fromisoformat(f.accrual_end_iso),
        "implied_rate": f.implied_rate,
    } for f in req.futures])
    ois   = [(q.tenor_years, q.rate) for q in req.ois_quotes]
    return SOFRCurveBootstrapper.from_market_data(
        ref, req.sofr_on, futs, ois,
        sigma=req.hw_sigma,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/curve/zero", tags=["Curve"])
def zero_curve(
    rate: float = 0.0358,
    tenors: str = "0.5,1,2,3,5,7,10,15,20,30",
):
    """
    Return zero rates and discount factors for a flat SOFR curve.

    Parameters
    ----------
    rate   : overnight SOFR level (decimal)
    tenors : comma-separated list of tenors in years
    """
    curve = _flat_curve(rate)
    t_list = [float(t) for t in tenors.split(",")]
    rows   = []
    for t in t_list:
        par = curve.par_ois_rate(t) * 100
        row = {
            "tenor_yrs":       t,
            "df":              round(float(curve.df(t)), 8),
            "zero_rate_pct":   round(float(curve.zero_rate(t) * 100), 5),
            "fwd_rate_pct":    round(float(curve.forward_rate(t, t + 1) * 100), 5)
                               if t + 1 <= curve._times[-1] else None,
            "par_ois_rate_pct": _sanitize(round(float(par), 5)),
        }
        rows.append(row)
    return {"ref_date": date.today().isoformat(), "sofr_on_pct": rate * 100, "curve": rows}


@app.post("/curve/bootstrap", tags=["Curve"])
def bootstrap_curve(req: BootstrapRequest):
    """Bootstrap a SOFR OIS curve from futures and OIS swap quotes."""
    try:
        curve = _bootstrap_curve(req)
        rows  = []
        for t in [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]:
            if t < curve._times[-1]:
                rows.append(_sanitize_row({
                    "tenor_yrs":      t,
                    "zero_rate_pct":  round(float(curve.zero_rate(t) * 100), 5),
                    "par_ois_pct":    round(float(curve.par_ois_rate(t) * 100), 5),
                    "df":             round(float(curve.df(t)), 8),
                }))
        return {"ref_date": req.ref_date_iso, "n_pillars": len(curve._times), "zero_curve": rows}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/price/swap", tags=["Pricing"])
def price_swap(req: SwapRequest):
    """Price a SOFR OIS swap and return PV, DV01, par rate, and carry."""
    try:
        curve = _flat_curve(req.sofr_on)
        eff   = date.fromisoformat(req.effective_date_iso)
        mat   = date.fromisoformat(req.maturity_date_iso)
        swap  = SOFRSwap(eff, mat, req.fixed_rate, req.notional, pay_fixed=req.pay_fixed)
        summ  = swap.summary(curve)
        return {k: round(v, 6) if isinstance(v, float) else v for k, v in summ.items()}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/price/fra", tags=["Pricing"])
def price_fra(req: FRARequest):
    """Price a SOFR Forward Rate Agreement."""
    try:
        curve = _flat_curve(req.sofr_on)
        start = date.fromisoformat(req.start_date_iso)
        end   = date.fromisoformat(req.end_date_iso)
        fra   = ForwardRateAgreement(start, end, req.fixed_rate, req.notional, req.pay_receive)
        return {
            "forward_rate_pct": round(fra.forward_rate(curve) * 100, 5),
            "fixed_rate_pct":   round(req.fixed_rate * 100, 5),
            "net_pv_usd":       round(fra.pv(curve), 2),
            "dv01_usd":         round(fra.dv01(curve), 2),
            "accrual_frac":     round(fra.accrual_frac, 6),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/risk/scenario", tags=["Risk"])
def scenario_pnl(req: ScenarioRequest):
    """Compute P&L of a single par swap under a yield curve scenario."""
    try:
        curve = _flat_curve(req.sofr_on)
        eff   = date.today()
        mat   = eff + relativedelta(years=int(req.swap_tenor_yrs))
        swap  = SOFRSwap(eff, mat, curve.par_ois_rate(req.swap_tenor_yrs), req.swap_notional)
        rpt   = RiskReport({"swap": swap}, curve)

        if req.shift_type == "parallel":
            shifted = ScenarioEngine.parallel_shift(curve, req.shift1_bps)
        elif req.shift_type == "twist":
            shifted = ScenarioEngine.twist_shift(curve, req.shift1_bps, req.shift2_bps)
        else:
            shifted = ScenarioEngine.key_rate_shift(curve, req.key_tenor, req.shift1_bps)

        pnl = swap.pv(shifted) - swap.pv(curve)
        return {
            "shift_type":   req.shift_type,
            "shift1_bps":   req.shift1_bps,
            "shift2_bps":   req.shift2_bps,
            "pnl_usd":      round(pnl, 2),
            "base_pv_usd":  round(swap.pv(curve), 2),
            "dv01_usd":     round(swap.dv01(curve), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/risk/dv01", tags=["Risk"])
def dv01_ladder(req: DV01Request):
    """Compute key-rate DV01 ladder for a portfolio of par swaps."""
    try:
        curve = _flat_curve(req.sofr_on)
        instruments = {}
        for i, s in enumerate(req.swaps):
            t   = float(s.get("tenor_y", 10.0))
            n   = float(s.get("notional", 10_000_000))
            pf  = bool(s.get("pay_fixed", True))
            eff = date.today()
            mat = eff + relativedelta(years=int(t))
            swap = SOFRSwap(eff, mat, curve.par_ois_rate(t), n, pay_fixed=pf)
            instruments[f"{t}Y"] = swap
        rpt    = RiskReport(instruments, curve)
        ladder = rpt.dv01_ladder()
        return {
            "total_dv01_usd": round(float(ladder["total_dv01_usd"].sum()), 2),
            "ladder":         ladder[["tenor_yrs", "total_dv01_usd"]].round(2).to_dict("records"),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/carry/table", tags=["Analytics"])
def carry_table(sofr_on: float = 0.0358, dt_months: float = 1.0):
    """Carry + roll-down table across standard tenors for a flat SOFR curve."""
    try:
        curve = _flat_curve(sofr_on)
        dt    = dt_months / 12.0
        tbl   = carry_rolldown_table(curve, dt_years=dt)
        tbl   = tbl.dropna()   # remove tenors where par_ois_rate is undefined
        return {
            "sofr_on_pct": sofr_on * 100,
            "horizon_months": dt_months,
            "table": [_sanitize_row(r) for r in tbl.reset_index().round(4).to_dict("records")],
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/signals/composite", tags=["Signals"])
def composite_signal_endpoint(req: SignalRequest):
    """Compute the composite macro trading signal for a single snapshot."""
    try:
        n   = 10   # small synthetic history to build signal
        idx = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
        df  = pd.DataFrame({
            "core_pce_yoy":  req.core_pce_yoy,
            "unemployment":  req.unemployment,
            "tsy_2y":        req.tsy_2y,
            "tsy_10y":       req.tsy_10y,
            "effr":          req.effr,
            "taylor_rate":   req.taylor_rate,
        }, index=idx)
        result = composite_signal(df)
        pos = int(result["position"].iloc[-1])
        return {
            "position":        pos,
            "position_label":  {1: "LONG", 0: "FLAT", -1: "SHORT"}[pos],
            "composite_score": round(float(result["composite_score"].iloc[-1]), 4)
                               if "composite_score" in result.columns else None,
            "signal_inputs": {
                "policy_gap_bps":    round((req.effr - req.taylor_rate) * 100, 1),
                "slope_2s10s_bps":   round((req.tsy_10y - req.tsy_2y) * 100, 1),
                "core_pce_pct":      req.core_pce_yoy,
                "unemployment_pct":  req.unemployment,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/taylor/rate", tags=["Signals"])
def taylor_rate_endpoint(req: TaylorRequest):
    """Compute Taylor Rule recommended rate for a given macro snapshot."""
    try:
        cfg = TaylorRuleConfig(
            neutral_rate=req.neutral_rate,
            inflation_target=req.inflation_target,
            nairu=req.nairu,
            variant=req.variant,
        )
        idx = pd.date_range("2025-01-01", periods=1, freq="MS")
        res = compute_taylor_rule(
            pd.Series([req.inflation],    index=idx),
            pd.Series([req.unemployment], index=idx),
            cfg,
        )
        taylor_r  = float(res["taylor_rate"].iloc[0])
        infl_gap  = float(res["inflation_gap"].iloc[0])
        unem_gap  = float(res["unemployment_gap"].iloc[0])
        return {
            "taylor_rate_pct":     round(taylor_r, 4),
            "inflation_gap_pct":   round(infl_gap, 4),
            "unemployment_gap_pct": round(unem_gap, 4),
            "inflation_contribution_pct":  round(0.5 * infl_gap, 4),
            "output_contribution_pct":     round(unem_gap, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/fomc/probabilities", tags=["Signals"])
def fomc_probabilities_endpoint(req: FOMCRequest):
    """Compute FOMC outcome probabilities from futures-implied rates."""
    try:
        probs   = fedwatch_probabilities(req.implied_rate, req.current_rate)
        summary = fomc_prob_summary(req.implied_rate, req.current_rate)
        return {
            "current_rate_pct": req.current_rate * 100,
            "implied_rate_pct": req.implied_rate * 100,
            "implied_move_bps": round((req.implied_rate - req.current_rate) * 10000, 1),
            "probabilities":    {f"{k:+d}bp": round(v, 4) for k, v in sorted(probs.items())},
            "summary": {
                "p_cut":  round(summary["p_cut"], 4),
                "p_hold": round(summary["p_hold"], 4),
                "p_hike": round(summary["p_hike"], 4),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/convexity/schedule", tags=["Analytics"])
def convexity_schedule(
    sigma: float = 0.010,
    mean_rev: float = 0.050,
    n_contracts: int = 8,
):
    """Hull-White convexity adjustment schedule for SR3 futures strip."""
    rows = []
    for i in range(n_contracts):
        t1 = 0.25 * (i + 1)
        t2 = t1 + 0.25
        ca = hull_white_convexity_adjustment(t1, t2, sigma=sigma, mean_reversion=mean_rev)
        rows.append({
            "contract":    i + 1,
            "t1_years":   t1,
            "t2_years":   t2,
            "ca_bps":     round(ca * 10000, 4),
        })
    return {
        "sigma_pct": sigma * 100,
        "mean_reversion_pct": mean_rev * 100,
        "schedule": rows,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

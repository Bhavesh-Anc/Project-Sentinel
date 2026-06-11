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
from sofr_engine.swaption import Swaption, SwaptionVolSurface, price_swaption
from sofr_engine.sabr import (
    SABRParams, SABRSurface, sabr_implied_vol, sabr_vol_smile, calibrate_sabr,
)
from models.return_attribution import (
    attribute_single_period as _attr_single,
    steepener_attribution as _attr_steepener,
)
from sofr_engine.cap_floor import (
    Cap, Floor, CapFloorVolSurface, strip_caplet_vols, price_cap_floor,
)
from sofr_engine.monte_carlo import (
    HullWhiteParams, simulate_hw, price_zcb_mc, price_caplet_mc,
    portfolio_var_hw, parametric_var, convergence_diagnostics,
)
from sofr_engine.bermudan import (
    price_bermudan_swaption, price_european_swaption_hw,
)
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


class SwaptionRequest(BaseModel):
    sofr_on:          float = Field(0.0433, description="Flat curve level (decimal)")
    expiry_years:     float = Field(1.0,  ge=0.01, le=30.0, description="Option expiry in years")
    swap_tenor_years: float = Field(5.0,  ge=0.25, le=30.0, description="Underlying swap tenor in years")
    strike:           float | None = Field(None, description="Fixed rate (decimal); None = ATM")
    notional:         float = Field(10_000_000.0, description="Notional in USD")
    swaption_type:    Literal["payer", "receiver"] = "payer"
    vol:              float = Field(0.20, ge=0.001, le=2.0, description="Black-76 implied vol (decimal)")


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


@app.post("/price/swaption", tags=["Pricing"])
def price_swaption_endpoint(req: SwaptionRequest):
    """Price a European SOFR swaption via Black-76 and return full analytics."""
    try:
        curve = _flat_curve(req.sofr_on)
        result = price_swaption(
            curve,
            expiry_years=req.expiry_years,
            swap_tenor_years=req.swap_tenor_years,
            strike=req.strike,
            notional=req.notional,
            swaption_type=req.swaption_type,
            vol=req.vol,
        )
        return {k: round(v, 6) if isinstance(v, float) else v for k, v in result.items()}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/vol/surface", tags=["Analytics"])
def vol_surface():
    """Return a typical 2024-vintage ATM swaption vol surface (Black-76 implied vols)."""
    surf = SwaptionVolSurface.typical_market()
    df   = surf.to_dataframe()
    return {
        "description": "ATM Black-76 implied vols (%). Rows = option expiry, cols = swap tenor.",
        "surface": df.reset_index().to_dict("records"),
    }


# ── SABR smile ─────────────────────────────────────────────────────────────────

class MCVaRRequest(BaseModel):
    sofr_on:        float = Field(0.0433, ge=0.0, le=0.20, description="Flat curve SOFR (decimal)")
    hw_a:           float = Field(0.05,  ge=0.0, le=2.0,   description="HW mean reversion a")
    hw_sigma:       float = Field(0.010, gt=0.0, le=0.20,  description="HW short-rate vol σ")
    portfolio_dv01: float = Field(10_000.0, description="Portfolio DV01 in $/bp (positive=long)")
    horizon_days:   int   = Field(1, ge=1, le=252,          description="VaR horizon in days")
    confidence:     float = Field(0.99, gt=0.5, lt=1.0,    description="Confidence level")
    n_paths:        int   = Field(10_000, ge=100, le=200_000)


class BermudanRequest(BaseModel):
    sofr_on:          float = Field(0.0433, ge=0.0, le=0.20)
    hw_a:             float = Field(0.05, ge=0.0, le=2.0)
    hw_sigma:         float = Field(0.010, gt=0.0, le=0.20)
    first_exercise:   float = Field(1.0, ge=0.25, le=20.0, description="First exercise date (years)")
    swap_maturity:    float = Field(6.0, ge=1.0, le=30.0,  description="Swap maturity (years)")
    strike:           float | None = Field(None, description="Fixed rate; None = ATM")
    notional:         float = Field(1_000_000.0)
    pay_receive:      Literal["payer", "receiver"] = "payer"
    exercise_freq:    int   = Field(2, ge=1, le=4)
    n_paths:          int   = Field(5_000, ge=500, le=50_000)


class MCCapletRequest(BaseModel):
    sofr_on:      float = Field(0.0433, ge=0.0, le=0.20)
    hw_a:         float = Field(0.05,  ge=0.0, le=2.0)
    hw_sigma:     float = Field(0.010, gt=0.0, le=0.20)
    strike:       float = Field(0.04,  ge=0.0, le=0.20)
    t_reset:      float = Field(1.0,   ge=0.01, le=30.0)
    t_pay:        float = Field(1.25,  ge=0.01, le=30.0)
    notional:     float = Field(1_000_000.0)
    n_paths:      int   = Field(10_000, ge=500, le=100_000)


class CapFloorRequest(BaseModel):
    sofr_on:         float = Field(0.0433, ge=0.0, le=0.20, description="Flat curve overnight SOFR (decimal)")
    maturity_years:  float = Field(5.0, ge=0.25, le=30.0, description="Cap/floor maturity in years")
    strike:          float | None = Field(None, description="Strike (decimal); None = ATM")
    notional:        float = Field(10_000_000.0, description="Notional in USD")
    vol:             float = Field(0.30, ge=0.001, le=5.0, description="Black-76 flat vol (decimal)")
    instrument:      Literal["cap", "floor"] = "cap"
    freq:            int   = Field(4, ge=1, le=12, description="Reset frequency per year (4=quarterly)")


class AttributionSingleRequest(BaseModel):
    sofr_start:      float = Field(0.0433, ge=0.0, le=0.20, description="Curve level at period start (decimal)")
    sofr_end:        float = Field(0.0408, ge=0.0, le=0.20, description="Curve level at period end (decimal)")
    tenor_years:     float = Field(10.0, ge=0.25, le=50.0, description="Position tenor in years")
    dt_years:        float = Field(1/12, gt=0.0, le=5.0,   description="Holding period in years (default 1M)")
    financing_rate:  float | None = Field(None, description="Financing rate (decimal); None = overnight SOFR")


class AttributionSteepenerRequest(BaseModel):
    sofr_start:   float = Field(0.0433, ge=0.0, le=0.20)
    sofr_end:     float = Field(0.0408, ge=0.0, le=0.20)
    short_tenor:  float = Field(2.0, ge=0.25, le=10.0, description="Short leg tenor (years)")
    long_tenor:   float = Field(10.0, ge=1.0, le=50.0,  description="Long leg tenor (years)")
    dt_years:     float = Field(1/12, gt=0.0, le=5.0)


class SABRSmileRequest(BaseModel):
    forward_rate:     float = Field(0.0453, description="Forward swap rate (decimal)")
    expiry_years:     float = Field(1.0, ge=0.01, le=30.0, description="Option expiry in years")
    alpha:            float = Field(0.05, gt=0.0, description="SABR alpha (initial vol)")
    beta:             float = Field(0.5, ge=0.0, le=1.0, description="SABR beta (backbone)")
    rho:              float = Field(-0.25, gt=-1.0, lt=1.0, description="SABR rho (skew)")
    nu:               float = Field(0.40, ge=0.0, description="SABR nu (vol of vol)")
    n_strikes:        int   = Field(21, ge=3, le=101, description="Number of strike points")
    strike_range_bps: float = Field(200.0, gt=0.0, description="±range around ATM in bps")


class SABRCalibrateRequest(BaseModel):
    forward_rate:   float       = Field(0.0453, description="Forward swap rate (decimal)")
    expiry_years:   float       = Field(1.0, ge=0.01, le=30.0)
    strikes:        list[float] = Field(..., min_length=1, description="Strike rates (decimal)")
    market_vols:    list[float] = Field(..., min_length=1, description="Market Black-76 vols (decimal)")
    beta:           float       = Field(0.5, ge=0.0, le=1.0)


@app.post("/sabr/smile", tags=["SABR Volatility"])
def sabr_smile_endpoint(req: SABRSmileRequest):
    """
    Compute SABR vol smile across a strike grid given SABR parameters.
    Returns strike_pct, moneyness_bps, sabr_vol_pct, and normal_vol_bps columns.
    """
    try:
        params = SABRParams(alpha=req.alpha, beta=req.beta, rho=req.rho, nu=req.nu)
        df = sabr_vol_smile(
            F=req.forward_rate,
            T=req.expiry_years,
            params=params,
            n_strikes=req.n_strikes,
            strike_range_bps=req.strike_range_bps,
        )
        return {
            "forward_rate_pct": round(req.forward_rate * 100, 6),
            "expiry_years": req.expiry_years,
            "atm_vol_pct": round(float(df.loc[df["moneyness_bps"].abs().idxmin(), "sabr_vol_pct"]), 4),
            "smile": df.round(6).to_dict("records"),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/sabr/calibrate", tags=["SABR Volatility"])
def sabr_calibrate_endpoint(req: SABRCalibrateRequest):
    """
    Calibrate SABR parameters {alpha, rho, nu} to a set of (strike, vol) market quotes
    with beta fixed. Returns calibrated parameters and vol fit quality (RMSE in bps).
    """
    if len(req.strikes) != len(req.market_vols):
        raise HTTPException(status_code=422, detail="strikes and market_vols must have equal length")
    try:
        params = calibrate_sabr(
            F=req.forward_rate,
            T=req.expiry_years,
            strikes=req.strikes,
            market_vols=req.market_vols,
            beta=req.beta,
        )
        # Compute fit quality
        fitted_vols = [sabr_implied_vol(req.forward_rate, k, req.expiry_years, params) for k in req.strikes]
        residuals   = [abs(fv - mv) * 10000 for fv, mv in zip(fitted_vols, req.market_vols)]
        rmse_bps    = float(np.sqrt(np.mean([r**2 for r in residuals])))
        return {
            "alpha": round(params.alpha, 8),
            "beta":  round(params.beta,  6),
            "rho":   round(params.rho,   8),
            "nu":    round(params.nu,    8),
            "rmse_bps": round(rmse_bps, 4),
            "max_error_bps": round(max(residuals), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/sabr/surface", tags=["SABR Volatility"])
def sabr_surface_endpoint():
    """
    Return the SABR-calibrated vol surface anchored to the typical ATM grid.
    Uses USD convention: beta=0.5, rho=-0.25, nu=0.40. Returns calibrated alpha
    per grid node and the ATM vol reproduced by SABR.
    """
    atm_surf  = SwaptionVolSurface.typical_market()
    curve     = _flat_curve(0.0453)
    sabr_surf = SABRSurface.calibrate_from_atm_surface(atm_surf, curve)
    rows = []
    for i, exp in enumerate(sabr_surf._expiries):
        for j, ten in enumerate(sabr_surf._tenors):
            p = sabr_surf._params[i][j]
            # Approximate ATM forward from curve; SABR ATM vol is self-consistent
            F = float(curve.par_ois_rate(float(exp) + float(ten) / 2.0))
            if F <= 0:
                F = 0.045
            atm_vol = sabr_implied_vol(F, F, float(exp), p) * 100.0
            rows.append({
                "expiry_years": float(exp),
                "tenor_years":  float(ten),
                "alpha":      round(p.alpha, 6),
                "beta":       round(p.beta,  4),
                "rho":        round(p.rho,   4),
                "nu":         round(p.nu,    4),
                "atm_vol_pct": round(atm_vol, 4),
            })
    return {
        "description": "SABR params calibrated to ATM surface (USD convention: beta=0.5, rho=-0.25, nu=0.40)",
        "nodes": rows,
    }


# ── Bermudan swaption (LSM) ────────────────────────────────────────────────────

@app.post("/bermudan/price", tags=["Bermudan Swaption"])
def bermudan_price_endpoint(req: BermudanRequest):
    """
    Price a Bermudan swaption via Longstaff-Schwartz (2001) Monte Carlo.

    The holder may exercise at any semi-annual (or quarterly) date between
    first_exercise and swap_maturity. Returns Bermudan price, European lower
    bound, early exercise premium, and per-date exercise probabilities.
    """
    if req.first_exercise >= req.swap_maturity:
        raise HTTPException(status_code=422,
                            detail="first_exercise must be < swap_maturity")
    try:
        curve  = _flat_curve(req.sofr_on)
        params = HullWhiteParams(a=req.hw_a, sigma=req.hw_sigma)
        result = price_bermudan_swaption(
            curve, params,
            first_exercise  = req.first_exercise,
            swap_maturity   = req.swap_maturity,
            strike          = req.strike,
            notional        = req.notional,
            pay_receive     = req.pay_receive,
            exercise_freq   = req.exercise_freq,
            n_paths         = req.n_paths,
            seed            = 42,
        )
        return {
            "price":                 round(result.price, 2),
            "european_lower_bound":  round(result.european_lower, 2),
            "early_exercise_premium": round(result.early_exercise_premium, 2),
            "strike_pct":            round(result.strike * 100, 4),
            "swap_maturity":         result.swap_maturity,
            "n_exercise_dates":      len(result.exercise_dates),
            "exercise_schedule": [
                {"date_years": round(d, 4), "exercise_prob": round(p, 4)}
                for d, p in zip(result.exercise_dates, result.exercise_probs)
            ],
            "n_paths": result.n_paths,
            "pay_receive": result.pay_receive,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Monte Carlo / VaR ─────────────────────────────────────────────────────────

@app.post("/mc/var", tags=["Monte Carlo"])
def mc_var_endpoint(req: MCVaRRequest):
    """
    Monte Carlo Value-at-Risk under Hull-White 1-factor model.

    Simulates the short-rate process exactly (no Euler discretization) and
    computes VaR / CVaR for a fixed-income portfolio characterised by its DV01.
    Returns the full P&L percentile distribution and 1-day yield vol estimate.
    """
    try:
        curve  = _flat_curve(req.sofr_on)
        params = HullWhiteParams(a=req.hw_a, sigma=req.hw_sigma)
        result = portfolio_var_hw(
            curve, params,
            portfolio_dv01 = req.portfolio_dv01,
            horizon        = req.horizon_days / 252.0,
            confidence     = req.confidence,
            n_paths        = req.n_paths,
            seed           = 42,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/mc/caplet", tags=["Monte Carlo"])
def mc_caplet_endpoint(req: MCCapletRequest):
    """
    Price a SOFR caplet by Hull-White Monte Carlo simulation.

    Returns MC price with 95% confidence interval and a Black-76 benchmark
    for comparison. Demonstrates convergence of MC to analytical pricing.
    """
    if req.t_pay <= req.t_reset:
        raise HTTPException(status_code=422, detail="t_pay must be > t_reset")
    try:
        curve  = _flat_curve(req.sofr_on)
        params = HullWhiteParams(a=req.hw_a, sigma=req.hw_sigma)
        result = price_caplet_mc(
            curve, params,
            strike   = req.strike,
            t_reset  = req.t_reset,
            t_pay    = req.t_pay,
            notional = req.notional,
            n_paths  = req.n_paths,
            seed     = 42,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/mc/zcb-convergence", tags=["Monte Carlo"])
def mc_zcb_convergence_endpoint(
    sofr_on:  float = 0.0433,
    maturity: float = 5.0,
    hw_a:     float = 0.05,
    hw_sigma: float = 0.010,
):
    """
    Hull-White MC vs analytical ZCB price convergence as n_paths increases.

    Shows how MC pricing error (in bps) decreases as 1/√n_paths.
    Useful for demonstrating MC convergence to the exact bond price.
    """
    try:
        curve   = _flat_curve(sofr_on)
        params  = HullWhiteParams(a=hw_a, sigma=hw_sigma)
        results = convergence_diagnostics(curve, params,
                                          test_maturity=maturity, seed=42)
        return {
            "maturity":   maturity,
            "sofr_on":    sofr_on,
            "analytical": round(float(curve.df(maturity)), 8),
            "convergence": results,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Cap / Floor pricing ────────────────────────────────────────────────────────

@app.post("/cap/price", tags=["Cap / Floor"])
def cap_floor_price_endpoint(req: CapFloorRequest):
    """
    Price a SOFR cap or floor strip (Black-76) and return full analytics.

    If strike is None, uses the ATM forward rate (annuity-weighted average
    of quarterly SOFR forward rates over the cap tenor).
    Returns PV, DV01, vega, ATM forward, moneyness, and per-caplet count.
    """
    try:
        curve = _flat_curve(req.sofr_on)
        result = price_cap_floor(
            curve,
            maturity_years=req.maturity_years,
            strike=req.strike,
            notional=req.notional,
            vol=req.vol,
            instrument=req.instrument,
            freq=req.freq,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/cap/vol-surface", tags=["Cap / Floor"])
def cap_vol_surface_endpoint(sofr_on: float = 0.0433):
    """
    Return a typical 2024-vintage USD SOFR cap vol surface (Black-76 flat vols).

    Grid: tenors [1Y, 2Y, 3Y, 5Y, 7Y, 10Y] × strikes [ATM-150bps … ATM+150bps].
    """
    try:
        surf = CapFloorVolSurface.typical_market(sofr_on=sofr_on)
        rows = []
        for i, T in enumerate(surf._tenors):
            for j, K in enumerate(surf._strikes):
                rows.append({
                    "tenor_years": T,
                    "strike_pct":  round(K * 100, 4),
                    "vol_pct":     round(surf._vols[i, j] * 100, 4),
                })
        return {
            "description": "Black-76 flat cap vols (%). 2024 USD market approximation.",
            "sofr_on_pct": round(sofr_on * 100, 4),
            "nodes":       rows,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/cap/strip-vols", tags=["Cap / Floor"])
def strip_caplet_vols_endpoint(
    sofr_on:  float = 0.0433,
    strike:   float = 0.04,
    tenors:   str   = "1,2,3,5,7,10",
):
    """
    Bootstrap per-caplet (forward) vols from market cap term vols.

    Uses typical 2024 USD market vol levels anchored to the provided SOFR curve.
    Returns expiry_years and forward caplet Black-76 vol for each stripped period.
    """
    try:
        curve = _flat_curve(sofr_on)
        surf  = CapFloorVolSurface.typical_market(sofr_on=sofr_on)
        t_list = [float(t) for t in tenors.split(",")]
        term_vols = {T: surf.vol(T, strike) for T in t_list}
        caplet_vols = strip_caplet_vols(term_vols, curve, strike=strike)
        return {
            "sofr_on_pct":    round(sofr_on * 100, 4),
            "strike_pct":     round(strike * 100, 4),
            "term_vols":      {f"{T}Y": round(v * 100, 4) for T, v in term_vols.items()},
            "caplet_vols":    [
                {"expiry_years": round(e, 4), "vol_pct": round(v * 100, 4)}
                for e, v in caplet_vols
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Return attribution ─────────────────────────────────────────────────────────

@app.post("/attribution/single", tags=["Attribution"])
def attribution_single(req: AttributionSingleRequest):
    """
    Campisi return attribution for a single par-bond position over one holding period.

    Decomposes total P&L into: Carry, Roll-Down, Duration, Convexity, Residual.
    Both curves are flat SOFR curves built from the supplied overnight rates.
    """
    try:
        curve_s = _flat_curve(req.sofr_start)
        curve_e = _flat_curve(req.sofr_end)
        r = _attr_single(
            curve_start    = curve_s,
            curve_end      = curve_e,
            tenor_years    = req.tenor_years,
            dt_years       = req.dt_years,
            financing_rate = req.financing_rate,
        )
        return {
            "tenor_years":       r.tenor_years,
            "dt_years":          round(r.dt_years, 6),
            "yield_start_pct":   round(r.yield_start_pct, 5),
            "yield_end_pct":     round(r.yield_end_pct, 5),
            "delta_y_bps":       round(r.delta_y_bps, 4),
            "carry_bps":         round(r.carry_bps, 4),
            "rolldown_bps":      round(r.rolldown_bps, 4),
            "duration_bps":      round(r.duration_bps, 4),
            "convexity_bps":     round(r.convexity_bps, 6),
            "total_approx_bps":  round(r.total_approx_bps, 4),
            "total_actual_bps":  round(r.total_actual_bps, 4),
            "residual_bps":      round(r.residual_bps, 4),
            "modified_duration": round(r.modified_duration, 4),
            "convexity_years2":  round(r.convexity_years2, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/attribution/steepener", tags=["Attribution"])
def attribution_steepener(req: AttributionSteepenerRequest):
    """
    DV01-neutral 2s10s steepener attribution over one holding period.

    Receive fixed short_tenor / pay fixed long_tenor with DV01-neutral sizing.
    Returns leg-level and net attribution across all five Campisi components.
    """
    try:
        curve_s = _flat_curve(req.sofr_start)
        curve_e = _flat_curve(req.sofr_end)
        res = _attr_steepener(
            curve_start  = curve_s,
            curve_end    = curve_e,
            short_tenor  = req.short_tenor,
            long_tenor   = req.long_tenor,
            dt_years     = req.dt_years,
        )
        short_r = res["short_leg"]
        long_r  = res["long_leg"]
        return {
            "dv01_ratio":        round(res["dv01_ratio"], 6),
            "net_carry_bps":     round(res["net_carry_bps"], 4),
            "net_rolldown_bps":  round(res["net_rolldown_bps"], 4),
            "net_duration_bps":  round(res["net_duration_bps"], 4),
            "net_convexity_bps": round(res["net_convexity_bps"], 6),
            "net_total_bps":     round(res["net_total_bps"], 4),
            "dominant_component": res["net"]["dominant_component"],
            "short_leg": {
                "tenor_years":      short_r.tenor_years,
                "yield_start_pct":  round(short_r.yield_start_pct, 5),
                "delta_y_bps":      round(short_r.delta_y_bps, 4),
                "carry_bps":        round(short_r.carry_bps, 4),
                "rolldown_bps":     round(short_r.rolldown_bps, 4),
                "duration_bps":     round(short_r.duration_bps, 4),
                "total_actual_bps": round(short_r.total_actual_bps, 4),
                "modified_duration": round(short_r.modified_duration, 4),
            },
            "long_leg": {
                "tenor_years":      long_r.tenor_years,
                "yield_start_pct":  round(long_r.yield_start_pct, 5),
                "delta_y_bps":      round(long_r.delta_y_bps, 4),
                "carry_bps":        round(long_r.carry_bps, 4),
                "rolldown_bps":     round(long_r.rolldown_bps, 4),
                "duration_bps":     round(long_r.duration_bps, 4),
                "total_actual_bps": round(long_r.total_actual_bps, 4),
                "modified_duration": round(long_r.modified_duration, 4),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

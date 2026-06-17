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
from sofr_engine.credit import (
    HazardRateCurve as _HazardCurve,
    CDSContract as _CDSContract,
    bootstrap_hazard_curve as _bootstrap_haz,
    cds_pv as _cds_pv,
    cds_par_spread as _cds_par,
)
from sofr_engine.g2pp import (
    G2ppParams,
    simulate_g2pp as _sim_g2pp,
    g2pp_swaption_mc as _g2pp_sw_mc,
    g2pp_portfolio_var as _g2pp_var,
    g2pp_zcb as _g2pp_zcb,
)
from sofr_engine.cms import (
    CMSCaplet as _CMSCaplet,
    CMSSpreadOption as _CMSSpreadOption,
    CMSSwap as _CMSSwap,
    cms_convexity_adj, cms_caplet_pv, cms_floorlet_pv, cms_spread_option_pv,
)
from sofr_engine.lmm import (
    LMMParams as _LMMParams,
    initial_forwards as _lmm_fwds,
    simulate_lmm as _sim_lmm,
    caplet_black76 as _caplet_b76,
    cap_black76 as _cap_b76,
    cap_implied_vol as _cap_iv,
    swaption_lmm_mc as _sw_lmm_mc,
    rebonato_swaption_vol as _rebonato,
    calibrate_caplet_vols as _cal_cap_vols,
    calibrate_corr_decay as _cal_corr,
)
from sofr_engine.xva import (
    XVAParams as _XVAParams,
    compute_epe_profile as _compute_epe,
    full_xva as _full_xva,
)
from sofr_engine.xccy import (
    FXForwardCurve as _FXFwdCurve,
    CrossCurrencySwap as _XCCYSwap,
    fx_forward as _fx_fwd,
    xccy_par_basis as _xccy_par,
    xccy_swap_pv as _xccy_pv,
    xccy_dv01 as _xccy_dv01,
    xccy_basis_term_structure as _xccy_term,
)
from sofr_engine.inflation import (
    InflationCurve as _InflCurve,
    ZCInflationSwap as _ZCSwap,
    YoYInflationSwap as _YoYSwap,
    InflationCapFloor as _InflCapFloor,
    zc_inflation_pv as _zc_pv,
    yoy_inflation_pv as _yoy_pv,
    inflation_cap_floor_pv as _infl_cf_pv,
    breakeven_inflation as _breakeven,
    calibrate_inflation_curve as _cal_infl,
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


class CDSRequest(BaseModel):
    sofr_on:        float = Field(0.0433, ge=0.0, le=0.20, description="Risk-free SOFR rate")
    maturity_years: float = Field(5.0, ge=0.25, le=20.0)
    coupon:         float = Field(0.01, ge=0.0, le=0.20, description="Running coupon (decimal)")
    notional:       float = Field(10_000_000.0)
    recovery:       float = Field(0.40, ge=0.0, lt=1.0)
    hazard_rate:    float = Field(0.02, gt=0.0, le=0.50, description="Flat hazard rate (decimal)")
    buy_protection: bool  = True
    freq:           int   = Field(4, ge=1, le=12)


class CDSBootstrapRequest(BaseModel):
    sofr_on:    float          = Field(0.0433, ge=0.0, le=0.20)
    maturities: list[float]    = Field([1.0, 3.0, 5.0, 7.0, 10.0])
    spreads:    list[float]    = Field([0.005, 0.010, 0.015, 0.020, 0.030],
                                        description="Par CDS spreads (decimal)")
    recovery:   float          = Field(0.40, ge=0.0, lt=1.0)
    coupon:     float          = Field(0.01, ge=0.0, le=0.20)
    notional:   float          = Field(10_000_000.0)


class G2ppSwaptionRequest(BaseModel):
    sofr_on:     float = Field(0.0433, ge=0.0, le=0.20)
    a:           float = Field(0.05, ge=1e-4, le=2.0, description="x mean-reversion speed")
    b:           float = Field(0.10, ge=1e-4, le=2.0, description="y mean-reversion speed")
    sigma:       float = Field(0.010, gt=0.0, le=0.20, description="x vol")
    eta:         float = Field(0.008, gt=0.0, le=0.20, description="y vol")
    rho:         float = Field(-0.30, ge=-0.99, le=0.99, description="W1-W2 correlation")
    expiry:      float = Field(1.0, ge=0.01, le=20.0)
    swap_tenor:  float = Field(5.0, ge=0.25, le=30.0)
    strike:      float | None = Field(None, description="Strike; None = ATM")
    notional:    float = Field(1_000_000.0)
    pay_receive: Literal["payer", "receiver"] = "payer"
    freq:        int   = Field(2, ge=1, le=4)
    n_paths:     int   = Field(10_000, ge=500, le=100_000)


class G2ppVaRRequest(BaseModel):
    sofr_on:        float = Field(0.0433, ge=0.0, le=0.20)
    a:              float = Field(0.05, ge=1e-4, le=2.0)
    b:              float = Field(0.10, ge=1e-4, le=2.0)
    sigma:          float = Field(0.010, gt=0.0, le=0.20)
    eta:            float = Field(0.008, gt=0.0, le=0.20)
    rho:            float = Field(-0.30, ge=-0.99, le=0.99)
    portfolio_dv01: float = Field(10_000.0, description="Portfolio DV01 ($ per bp)")
    horizon_days:   int   = Field(1, ge=1, le=250)
    confidence:     float = Field(0.99, ge=0.90, le=0.9999)
    n_paths:        int   = Field(10_000, ge=1000, le=100_000)


class CMSConvexityRequest(BaseModel):
    sofr_on:     float = Field(0.0433, ge=0.0, le=0.20, description="Flat SOFR curve level (decimal)")
    expiry:      float = Field(1.0,    ge=0.01, le=30.0, description="Option expiry (years)")
    swap_tenor:  float = Field(10.0,   ge=0.25, le=30.0, description="CMS swap tenor (years)")
    vol:         float = Field(0.30,   gt=0.0,  le=5.0,  description="Black-76 swaption vol")
    model:       Literal["linear_tsr", "replication"] = "linear_tsr"
    freq:        int   = Field(2, ge=1, le=4, description="CMS swap reset frequency")


class CMSCapletRequest(BaseModel):
    sofr_on:     float = Field(0.0433, ge=0.0, le=0.20)
    t_fix:       float = Field(1.0,    ge=0.01, le=30.0, description="Rate-fixing date (years)")
    t_pay:       float = Field(1.25,   ge=0.01, le=30.0, description="Payment date (years)")
    swap_tenor:  float = Field(10.0,   ge=0.25, le=30.0)
    strike:      float = Field(0.04,   ge=0.0,  le=0.30)
    notional:    float = Field(1_000_000.0)
    cap_floor:   Literal["cap", "floor"] = "cap"
    vol:         float = Field(0.30, gt=0.0, le=5.0)
    freq:        int   = Field(2, ge=1, le=4)
    model:       Literal["linear_tsr", "replication"] = "linear_tsr"


class CMSSpreadRequest(BaseModel):
    sofr_on:        float = Field(0.0433, ge=0.0, le=0.20)
    t_fix:          float = Field(1.0,    ge=0.01, le=30.0)
    t_pay:          float = Field(1.25,   ge=0.01, le=30.0)
    long_tenor:     float = Field(10.0,   ge=0.25, le=30.0)
    short_tenor:    float = Field(2.0,    ge=0.25, le=30.0)
    spread_strike:  float = Field(0.005,  ge=-0.10, le=0.20, description="Spread strike (decimal)")
    notional:       float = Field(10_000_000.0)
    call_put:       Literal["call", "put"] = "call"
    vol_long:       float = Field(0.30, gt=0.0, le=5.0)
    vol_short:      float = Field(0.30, gt=0.0, le=5.0)
    rho:            float = Field(0.7,  ge=-1.0, le=1.0, description="Correlation between CMS rates")
    freq:           int   = Field(2, ge=1, le=4)
    model:          Literal["linear_tsr", "replication"] = "linear_tsr"


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


# ── CDS pricing ───────────────────────────────────────────────────────────────

@app.post("/cds/price", tags=["CDS"])
def cds_price_endpoint(req: CDSRequest):
    """
    Price a CDS under piecewise-constant flat hazard rate model.
    Returns fee leg, protection leg, par spread, CS01, DV01.
    """
    try:
        curve = _flat_curve(req.sofr_on)
        hc    = _HazardCurve.flat(req.hazard_rate, [req.maturity_years], recovery=req.recovery)
        cds   = _CDSContract(
            maturity_years = req.maturity_years,
            coupon         = req.coupon,
            notional       = req.notional,
            recovery       = req.recovery,
            freq           = req.freq,
            buy_protection = req.buy_protection,
        )
        res = _cds_pv(curve, hc, cds)
        return {
            "pv":               res.pv,
            "fee_leg_pv":       res.fee_leg_pv,
            "prot_leg_pv":      res.prot_leg_pv,
            "par_spread_bps":   res.par_spread_bps,
            "risky_annuity":    res.risky_annuity,
            "cs01":             res.cs01,
            "dv01":             res.dv01,
            "upfront":          res.upfront,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/cds/bootstrap", tags=["CDS"])
def cds_bootstrap_endpoint(req: CDSBootstrapRequest):
    """
    Bootstrap hazard rate curve from par CDS spreads and reprice the term CDS.
    """
    if len(req.maturities) != len(req.spreads):
        raise HTTPException(status_code=422,
                            detail="maturities and spreads must have the same length")
    try:
        curve = _flat_curve(req.sofr_on)
        hc    = _bootstrap_haz(curve, req.maturities, req.spreads, recovery=req.recovery)
        schedule = []
        for T, s_in in zip(req.maturities, req.spreads):
            s_model = float(_cds_par(curve, hc, T))
            cds_t   = _CDSContract(T, coupon=req.coupon, notional=req.notional,
                                    recovery=req.recovery)
            res_t   = _cds_pv(curve, hc, cds_t)
            schedule.append({
                "maturity_years":    round(T, 4),
                "market_spread_bps": round(s_in * 10_000, 3),
                "model_spread_bps":  round(s_model * 10_000, 6),
                "hazard_rate_bps":   round(float(hc.hazard_at(T)) * 10_000, 3),
                "survival_prob":     round(hc.survival(T), 6),
                "cds_pv":            round(res_t.pv, 2),
            })
        return {
            "schedule": schedule,
            "recovery": req.recovery,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── G2++ two-factor model ─────────────────────────────────────────────────────

@app.post("/g2pp/swaption", tags=["G2++"])
def g2pp_swaption_endpoint(req: G2ppSwaptionRequest):
    """
    Price a European swaption under the G2++ two-factor Gaussian model via MC.
    Supports both payer and receiver, ATM or fixed strike.
    """
    try:
        curve  = _flat_curve(req.sofr_on)
        params = G2ppParams(a=req.a, b=req.b, sigma=req.sigma, eta=req.eta, rho=req.rho)
        res    = _g2pp_sw_mc(
            curve, params,
            expiry      = req.expiry,
            swap_tenor  = req.swap_tenor,
            strike      = req.strike,
            notional    = req.notional,
            pay_receive = req.pay_receive,
            freq        = req.freq,
            n_paths     = req.n_paths,
            seed        = 42,
        )
        return {
            "pv":                     round(res["pv"], 2),
            "mc_stderr":              round(res["mc_stderr"], 2),
            "forward_swap_rate_pct":  round(res["forward_swap_rate_pct"], 5),
            "strike_pct":             round(res["strike_pct"], 5),
            "annuity":                round(res["annuity"], 6),
            "n_paths":                res["n_paths"],
            "pay_receive":            res["pay_receive"],
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/g2pp/var", tags=["G2++"])
def g2pp_var_endpoint(req: G2ppVaRRequest):
    """
    Portfolio VaR/CVaR under G2++ via scenario simulation.
    Simulates short-rate scenarios at the horizon and computes PnL distribution.
    """
    try:
        curve  = _flat_curve(req.sofr_on)
        params = G2ppParams(a=req.a, b=req.b, sigma=req.sigma, eta=req.eta, rho=req.rho)
        res    = _g2pp_var(
            curve, params,
            portfolio_dv01 = req.portfolio_dv01,
            horizon        = req.horizon_days / 250.0,
            confidence     = req.confidence,
            n_paths        = req.n_paths,
            seed           = 42,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── CMS pricing ───────────────────────────────────────────────────────────────

@app.post("/cms/convexity", tags=["CMS"])
def cms_convexity_endpoint(req: CMSConvexityRequest):
    """
    Compute the CMS convexity adjustment and CMS-adjusted rate.

    Returns the forward swap rate, convexity adjustment (bps), and the
    CMS rate (forward + adjustment) under Linear TSR or static replication.
    """
    try:
        curve  = _flat_curve(req.sofr_on)
        result = cms_convexity_adj(
            curve, req.expiry, req.swap_tenor, req.vol,
            freq=req.freq, model=req.model,
        )
        return {
            "forward_swap_rate_pct":  round(result.forward_swap_rate * 100, 5),
            "convexity_adj_bps":      round(result.convexity_adj_bps, 4),
            "cms_rate_pct":           round(result.cms_rate * 100, 5),
            "expiry":                 req.expiry,
            "swap_tenor":             req.swap_tenor,
            "vol_pct":                round(req.vol * 100, 2),
            "model":                  req.model,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/cms/caplet", tags=["CMS"])
def cms_caplet_endpoint(req: CMSCapletRequest):
    """
    Price a CMS caplet or floorlet using the convexity-adjusted forward rate
    with Black-76 model.

    Returns PV, convexity-adjusted CMS rate, and the forward swap rate.
    """
    if req.t_pay <= req.t_fix:
        raise HTTPException(status_code=422,
                            detail="t_pay must be greater than t_fix")
    try:
        curve  = _flat_curve(req.sofr_on)
        caplet = _CMSCaplet(
            t_fix=req.t_fix, t_pay=req.t_pay, swap_tenor=req.swap_tenor,
            strike=req.strike, notional=req.notional, cap_floor=req.cap_floor,
            freq=req.freq,
        )
        pv  = cms_caplet_pv(caplet, curve, req.vol, model=req.model)
        res = cms_convexity_adj(curve, req.t_fix, req.swap_tenor, req.vol,
                                freq=req.freq, model=req.model)
        return {
            "pv":                     round(pv, 2),
            "cms_rate_pct":           round(res.cms_rate * 100, 5),
            "forward_swap_rate_pct":  round(res.forward_swap_rate * 100, 5),
            "convexity_adj_bps":      round(res.convexity_adj_bps, 4),
            "strike_pct":             round(req.strike * 100, 4),
            "cap_floor":              req.cap_floor,
            "notional":               req.notional,
            "model":                  req.model,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/cms/spread-option", tags=["CMS"])
def cms_spread_option_endpoint(req: CMSSpreadRequest):
    """
    Price a CMS spread option (e.g. 10Y-2Y steepener call/put) using
    Kirk's bivariate-normal approximation with CMS convexity adjustments.
    """
    if req.t_pay <= req.t_fix:
        raise HTTPException(status_code=422,
                            detail="t_pay must be greater than t_fix")
    if req.long_tenor <= req.short_tenor:
        raise HTTPException(status_code=422,
                            detail="long_tenor must exceed short_tenor")
    try:
        curve  = _flat_curve(req.sofr_on)
        option = _CMSSpreadOption(
            t_fix=req.t_fix, t_pay=req.t_pay,
            long_tenor=req.long_tenor, short_tenor=req.short_tenor,
            spread_strike=req.spread_strike, notional=req.notional,
            call_put=req.call_put, freq=req.freq,
        )
        pv = cms_spread_option_pv(
            option, curve, req.vol_long, req.vol_short, req.rho,
            model=req.model,
        )
        res_l = cms_convexity_adj(curve, req.t_fix, req.long_tenor,
                                  req.vol_long, freq=req.freq, model=req.model)
        res_s = cms_convexity_adj(curve, req.t_fix, req.short_tenor,
                                  req.vol_short, freq=req.freq, model=req.model)
        return {
            "pv":                          round(pv, 2),
            "cms_rate_long_pct":           round(res_l.cms_rate * 100, 5),
            "cms_rate_short_pct":          round(res_s.cms_rate * 100, 5),
            "cms_spread_pct":              round((res_l.cms_rate - res_s.cms_rate) * 100, 5),
            "spread_strike_bps":           round(req.spread_strike * 10_000, 2),
            "correlation":                 req.rho,
            "call_put":                    req.call_put,
            "notional":                    req.notional,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/cms/convexity-schedule", tags=["CMS"])
def cms_convexity_schedule(
    sofr_on: float = 0.0433,
    swap_tenor: float = 10.0,
    vol: float = 0.30,
    model: str = "linear_tsr",
):
    """
    Return the CMS convexity adjustment across a schedule of expiries (1M–10Y).
    Useful for visualising how the adjustment grows with time to expiry.
    """
    try:
        curve   = _flat_curve(sofr_on)
        expiries = [1/12, 3/12, 6/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        m = model if model in ("linear_tsr", "replication") else "linear_tsr"
        schedule = []
        for T in expiries:
            res = cms_convexity_adj(curve, T, swap_tenor, vol, model=m)
            schedule.append({
                "expiry_years":       round(T, 4),
                "forward_swap_rate_pct": round(res.forward_swap_rate * 100, 5),
                "convexity_adj_bps":  round(res.convexity_adj_bps, 4),
                "cms_rate_pct":       round(res.cms_rate * 100, 5),
            })
        return {"schedule": schedule, "swap_tenor": swap_tenor, "model": m}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── LMM Pydantic models ────────────────────────────────────────────────────────

class LMMCapRequest(BaseModel):
    sofr_on      : float = Field(0.0433, description="Overnight SOFR rate")
    tenor_years  : float = Field(5.0,   ge=0.5, description="Cap maturity in years")
    n_periods    : int   = Field(10,    ge=1,   description="Number of caplet periods")
    flat_vol     : float = Field(0.25,  gt=0,   description="Flat Black vol")
    strike       : float = Field(0.04,  ge=0,   description="Cap strike rate")
    notional     : float = Field(1_000_000.0, gt=0)
    corr_decay   : float = Field(0.10,  ge=0,   description="Exponential correlation decay λ")
    is_cap       : bool  = Field(True,  description="True=cap, False=floor")


class LMMSwaptionRequest(BaseModel):
    sofr_on      : float = Field(0.0433)
    tenor_years  : float = Field(5.0,  ge=0.5, description="Total tenor span in years")
    n_periods    : int   = Field(10,   ge=2,   description="Number of forward rate periods")
    flat_vol     : float = Field(0.25, gt=0)
    expiry_period: int   = Field(4,    ge=0,   description="Swaption expiry (period index k_start)")
    swap_end_period: int = Field(10,   ge=1,   description="Swap end period index k_end")
    strike       : float = Field(-1.0, description="Strike rate; -1 = ATM")
    notional     : float = Field(1_000_000.0, gt=0)
    corr_decay   : float = Field(0.10, ge=0)
    is_payer     : bool  = Field(True)
    n_paths      : int   = Field(2_000, ge=100)
    n_steps      : int   = Field(50,    ge=10)
    seed         : int   = Field(42)


class LMMCalibrateRequest(BaseModel):
    sofr_on      : float = Field(0.0433)
    tenor_years  : float = Field(5.0, ge=0.5)
    n_periods    : int   = Field(10,  ge=2)
    cap_flat_vols: list[float] = Field(
        default=[0.25] * 10,
        description="Flat ATM cap implied vol for each successive cap (length = n_periods)"
    )
    corr_decay   : float = Field(0.10, ge=0)


def _build_lmm_params(
    sofr_on: float,
    tenor_years: float,
    n_periods: int,
    flat_vol: float,
    corr_decay: float,
):
    dt     = tenor_years / n_periods
    tenors = np.linspace(0.0, tenor_years, n_periods + 1)
    vols   = np.full(n_periods, flat_vol)
    curve  = _flat_curve(sofr_on)
    params = _LMMParams(tenors=tenors, vols=vols, corr_decay=corr_decay)
    return curve, params


# ── LMM endpoints ──────────────────────────────────────────────────────────────

@app.post("/lmm/cap", tags=["LMM"])
def lmm_cap_endpoint(req: LMMCapRequest):
    """
    Black-76 cap or floor price from the LMM.
    Returns per-caplet breakdown and total cap PV.
    """
    try:
        curve, params = _build_lmm_params(
            req.sofr_on, req.tenor_years, req.n_periods,
            req.flat_vol, req.corr_decay,
        )
        F0 = _lmm_fwds(curve, params.tenors)
        K  = req.strike

        total_pv = _cap_b76(curve, params, K, req.notional, req.is_cap)
        caplets  = [
            {
                "period"    : k,
                "T_fix"     : round(float(params.tenors[k]), 4),
                "T_pay"     : round(float(params.tenors[k + 1]), 4),
                "forward_pct": round(float(F0[k]) * 100, 5),
                "pv"        : round(_caplet_b76(curve, params, k, K, req.notional, req.is_cap), 2),
            }
            for k in range(params.N)
        ]

        iv = _cap_iv(curve, params, K, total_pv, req.notional, req.is_cap)

        return {
            "total_pv"          : round(total_pv, 2),
            "implied_flat_vol"  : round(iv, 6),
            "strike"            : K,
            "notional"          : req.notional,
            "is_cap"            : req.is_cap,
            "n_periods"         : params.N,
            "caplets"           : caplets,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/lmm/swaption", tags=["LMM"])
def lmm_swaption_endpoint(req: LMMSwaptionRequest):
    """
    European swaption priced via LMM Monte Carlo under Q^{T_N}.
    Also returns the Rebonato approximate implied vol.
    """
    if req.expiry_period >= req.swap_end_period:
        raise HTTPException(422, "expiry_period must be < swap_end_period")
    if req.swap_end_period > req.n_periods:
        raise HTTPException(422, "swap_end_period must be ≤ n_periods")

    try:
        curve, params = _build_lmm_params(
            req.sofr_on, req.tenor_years, req.n_periods,
            req.flat_vol, req.corr_decay,
        )
        k_s = req.expiry_period
        k_e = req.swap_end_period

        # Compute ATM strike if requested
        if req.strike < 0:
            F0  = _lmm_fwds(curve, params.tenors)
            alp = params.alpha
            P, A = 1.0, 0.0
            for i in range(k_s, k_e):
                P = P / (1.0 + alp[i] * F0[i])
                A += alp[i] * P
            K = float((1.0 - P) / max(A, 1e-15))
        else:
            K = req.strike

        sim = _sim_lmm(curve, params, n_steps=req.n_steps, n_paths=req.n_paths, seed=req.seed)
        res = _sw_lmm_mc(sim, curve, k_s, k_e, K, req.notional, req.is_payer)
        rebonato_vol = _rebonato(curve, params, k_s, k_e)

        return {
            "pv"               : round(res["pv"], 2),
            "std_err"          : round(res["std_err"], 2),
            "strike"           : round(K, 6),
            "is_payer"         : req.is_payer,
            "swap_rate_mean_pct": round(res["swap_rate_mean"] * 100, 5),
            "annuity_mean"     : round(res["annuity_mean"], 6),
            "rebonato_vol_pct" : round(rebonato_vol * 100, 4),
            "n_paths"          : req.n_paths,
            "notional"         : req.notional,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/lmm/calibrate-caplet-vols", tags=["LMM"])
def lmm_calibrate_vols_endpoint(req: LMMCalibrateRequest):
    """
    Bootstrap caplet vols from flat ATM cap implied vols.
    Returns calibrated per-period caplet vols and initial forward rates.
    """
    if len(req.cap_flat_vols) != req.n_periods:
        raise HTTPException(422,
            f"cap_flat_vols length {len(req.cap_flat_vols)} ≠ n_periods {req.n_periods}")
    try:
        tenors  = np.linspace(0.0, req.tenor_years, req.n_periods + 1)
        vols    = np.full(req.n_periods, req.cap_flat_vols[0])
        curve   = _flat_curve(req.sofr_on)
        p_tmpl  = _LMMParams(tenors=tenors, vols=vols, corr_decay=req.corr_decay)
        p_cal   = _cal_cap_vols(curve, p_tmpl, req.cap_flat_vols)
        F0      = _lmm_fwds(curve, tenors)

        return {
            "calibrated_vols_pct": [round(v * 100, 4) for v in p_cal.vols.tolist()],
            "input_cap_vols_pct" : [round(v * 100, 4) for v in req.cap_flat_vols],
            "initial_forwards_pct": [round(float(f) * 100, 5) for f in F0.tolist()],
            "tenors"             : [round(t, 4) for t in tenors.tolist()],
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/lmm/rebonato-surface", tags=["LMM"])
def lmm_rebonato_surface(
    sofr_on    : float = 0.0433,
    tenor_years: float = 5.0,
    n_periods  : int   = 10,
    flat_vol   : float = 0.25,
    corr_decay : float = 0.10,
):
    """
    Compute the Rebonato approximate swaption vol surface across
    all expiry × swap-length combinations.
    """
    try:
        curve, params = _build_lmm_params(sofr_on, tenor_years, n_periods, flat_vol, corr_decay)
        N = params.N
        surface = []
        for k_s in range(N - 1):
            for k_e in range(k_s + 2, N + 1):
                vol = _rebonato(curve, params, k_s, k_e)
                surface.append({
                    "expiry_years" : round(float(params.tenors[k_s]), 4),
                    "swap_tenor_yrs": round(float(params.tenors[k_e] - params.tenors[k_s]), 4),
                    "k_start"      : k_s,
                    "k_end"        : k_e,
                    "rebonato_vol_pct": round(vol * 100, 4),
                })
        return {"surface": surface, "n_entries": len(surface), "n_periods": N}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# XVA  (CVA / DVA / FVA)
# ══════════════════════════════════════════════════════════════════════════════

class XVAPriceRequest(BaseModel):
    sofr_on:        float = Field(0.0433, ge=0.0, le=0.15, description="SOFR overnight rate")
    fixed_rate:     float = Field(0.045,  ge=0.0, le=0.15, description="Swap fixed rate (decimal)")
    maturity:       float = Field(5.0,    ge=0.5, le=30.0, description="Maturity in years")
    notional:       float = Field(1e6,    ge=1.0,          description="Notional")
    is_payer:       bool  = Field(True,                    description="True = we pay fixed")
    recovery:       float = Field(0.40,   ge=0.0, le=1.0,  description="LGD recovery rate")
    hazard_rate:    float = Field(0.01,   ge=0.0, le=0.5,  description="Flat hazard rate")
    own_hazard:     float = Field(0.005,  ge=0.0, le=0.5,  description="Own hazard rate (for DVA)")
    funding_spread: float = Field(0.005,  ge=0.0, le=0.1,  description="Funding spread (for FVA)")
    n_steps:        int   = Field(20,     ge=5,  le=100,   description="Time steps for EPE")
    n_paths:        int   = Field(500,    ge=100, le=5000,  description="MC paths")
    hw_a:           float = Field(0.05,   ge=1e-4, le=1.0, description="Hull-White mean reversion")
    hw_sigma:       float = Field(0.015,  ge=1e-4, le=0.5, description="Hull-White vol")


def _build_xva_inputs(req: XVAPriceRequest):
    curve = flat_sofr_curve(date.today(), req.sofr_on)
    g2pp = G2ppParams(
        a=req.hw_a, b=req.hw_a * 1.5,
        sigma=req.hw_sigma, eta=req.hw_sigma * 0.7,
        rho=-0.3,
    )
    xva_params = _XVAParams(
        n_paths=req.n_paths,
        n_steps=req.n_steps,
        funding_spread=req.funding_spread,
    )
    mats = [req.maturity]
    c_haz = _HazardCurve.flat(req.hazard_rate, mats, recovery=req.recovery)
    o_haz = _HazardCurve.flat(req.own_hazard,   mats, recovery=req.recovery)
    return curve, g2pp, xva_params, c_haz, o_haz


@app.post("/xva/price", tags=["XVA"])
def xva_price(req: XVAPriceRequest):
    """Compute CVA, DVA, and FVA for an interest-rate swap using G2++ MC simulation."""
    try:
        curve, g2pp, xva_params, c_haz, o_haz = _build_xva_inputs(req)
        result = _full_xva(
            curve=curve,
            g2pp_params=g2pp,
            maturity=req.maturity,
            fixed_rate=req.fixed_rate,
            notional=req.notional,
            pay_fixed=req.is_payer,
            counterparty_hazard=c_haz,
            own_hazard=o_haz,
            xva_params=xva_params,
        )
        return {
            "cva_bps":       round(result.cva / req.notional * 1e4, 4),
            "dva_bps":       round(result.dva / req.notional * 1e4, 4),
            "fva_bps":       round(result.fva / req.notional * 1e4, 4),
            "total_xva_bps": round(result.total_xva / req.notional * 1e4, 4),
            "cva":           round(result.cva, 4),
            "dva":           round(result.dva, 4),
            "fva":           round(result.fva, 4),
            "total_xva":     round(result.total_xva, 4),
            "n_steps":       xva_params.n_steps,
            "n_paths":       xva_params.n_paths,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/xva/epe-profile", tags=["XVA"])
def xva_epe_profile(req: XVAPriceRequest):
    """Return the Expected Positive Exposure profile over time."""
    try:
        curve, g2pp, xva_params, c_haz, o_haz = _build_xva_inputs(req)
        epe = _compute_epe(
            curve=curve,
            g2pp_params=g2pp,
            maturity=req.maturity,
            fixed_rate=req.fixed_rate,
            notional=req.notional,
            pay_fixed=req.is_payer,
            xva_params=xva_params,
        )
        return {
            "times":   [round(float(t), 4) for t in epe.times],
            "epe":     [round(float(v), 4) for v in epe.epe],
            "ene":     [round(float(v), 4) for v in epe.ene],
            "n_paths": xva_params.n_paths,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Cross-Currency Basis Swaps
# ══════════════════════════════════════════════════════════════════════════════

class XCCYParBasisRequest(BaseModel):
    usd_sofr:    float = Field(0.0433, ge=0.0, le=0.15, description="USD SOFR overnight")
    eur_rate:    float = Field(0.038,  ge=0.0, le=0.15, description="EUR OIS overnight")
    spot_fx:     float = Field(1.09,   ge=0.5, le=2.0,  description="Spot FX (USD per EUR)")
    maturity:    float = Field(5.0,    ge=0.5, le=30.0, description="Swap maturity years")
    notional_eur:float = Field(1e6,   ge=1.0,           description="EUR notional")
    freq:        int   = Field(4,      ge=1,   le=12,    description="Payment frequency per year")


class XCCYPriceRequest(BaseModel):
    usd_sofr:    float = Field(0.0433, ge=0.0, le=0.15)
    eur_rate:    float = Field(0.038,  ge=0.0, le=0.15)
    spot_fx:     float = Field(1.09,   ge=0.5, le=2.0)
    maturity:    float = Field(5.0,    ge=0.5, le=30.0)
    notional_eur:float = Field(1e6,   ge=1.0)
    basis_bps:   float = Field(0.0,   ge=-200.0, le=200.0, description="Basis spread in bps")
    freq:        int   = Field(4,      ge=1,  le=12)


def _build_xccy_curves(usd_sofr, eur_rate, spot_fx):
    usd_curve = flat_sofr_curve(date.today(), usd_sofr)
    eur_curve = flat_sofr_curve(date.today(), eur_rate)
    fx_curve = _FXFwdCurve(
        spot_fx=spot_fx,
        usd_curve=usd_curve,
        eur_curve=eur_curve,
    )
    return usd_curve, eur_curve, fx_curve


@app.post("/xccy/par-basis", tags=["XCCY"])
def xccy_par_basis_endpoint(req: XCCYParBasisRequest):
    """Compute the CIP-implied par basis spread for a cross-currency swap."""
    try:
        usd_curve, eur_curve, fx_curve = _build_xccy_curves(req.usd_sofr, req.eur_rate, req.spot_fx)
        basis = _xccy_par(usd_curve, eur_curve, req.spot_fx, req.maturity, req.freq)
        fwd = _fx_fwd(usd_curve, eur_curve, req.spot_fx, req.maturity)
        return {
            "par_basis_bps":  round(float(basis * 1e4), 4),
            "spot_fx":        req.spot_fx,
            "forward_fx":     round(float(fwd), 6),
            "maturity_years": req.maturity,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/xccy/price", tags=["XCCY"])
def xccy_price_endpoint(req: XCCYPriceRequest):
    """Price a cross-currency basis swap (MTM to USD)."""
    try:
        usd_curve, eur_curve, fx_curve = _build_xccy_curves(req.usd_sofr, req.eur_rate, req.spot_fx)
        swap = _XCCYSwap(
            maturity_years=req.maturity,
            notional_usd=req.notional_eur * req.spot_fx,
            freq=req.freq,
            basis_spread=req.basis_bps / 1e4,
        )
        result = _xccy_pv(usd_curve, eur_curve, req.spot_fx, swap)
        return {
            "pv_usd":           round(float(result.pv_usd), 2),
            "usd_leg_pv":       round(float(result.usd_leg_pv), 2),
            "eur_leg_pv_usd":   round(float(result.eur_leg_pv_in_usd), 2),
            "par_basis_bps":    round(float(result.par_basis_bps), 4),
            "eur_annuity_usd":  round(float(result.eur_annuity_in_usd), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/xccy/basis-term-structure", tags=["XCCY"])
def xccy_basis_term_structure_endpoint(
    usd_sofr: float = 0.0433,
    eur_rate: float = 0.038,
    spot_fx:  float = 1.09,
    freq:     int   = 4,
):
    """Return the CIP-implied basis spread across maturities (1Y to 30Y)."""
    try:
        usd_curve, eur_curve, fx_curve = _build_xccy_curves(usd_sofr, eur_rate, spot_fx)
        tenors = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
        rows = []
        for t in tenors:
            b = _xccy_par(usd_curve, eur_curve, spot_fx, t, freq)
            fwd = _fx_fwd(usd_curve, eur_curve, spot_fx, t)
            rows.append({
                "maturity_years": t,
                "par_basis_bps":  round(float(b * 1e4), 4),
                "forward_fx":     round(float(fwd), 6),
            })
        return {"term_structure": rows, "spot_fx": spot_fx}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Inflation-Linked Products
# ══════════════════════════════════════════════════════════════════════════════

class ZCInflationRequest(BaseModel):
    maturity:      float = Field(10.0,  ge=0.5, le=30.0, description="Maturity in years")
    fixed_rate:    float = Field(0.025, ge=-0.05, le=0.15, description="Fixed CPI growth rate")
    notional:      float = Field(1e6,   ge=1.0,            description="Notional")
    infl_rate:     float = Field(0.025, ge=-0.05, le=0.15, description="Expected inflation rate")
    sofr_on:       float = Field(0.0433, ge=0.0, le=0.15, description="SOFR overnight")
    is_receiver:   bool  = Field(False, description="True = we receive CPI, pay fixed")


class YoYInflationRequest(BaseModel):
    maturity:    float = Field(5.0,  ge=0.5, le=30.0)
    fixed_rate:  float = Field(0.025, ge=-0.05, le=0.15)
    notional:    float = Field(1e6,  ge=1.0)
    infl_rate:   float = Field(0.025, ge=-0.05, le=0.15)
    sofr_on:     float = Field(0.0433, ge=0.0, le=0.15)
    freq:        int   = Field(1, ge=1, le=4)
    is_receiver: bool  = Field(False)


class InflationCapFloorRequest(BaseModel):
    maturity:    float = Field(5.0,  ge=0.5, le=30.0)
    strike:      float = Field(0.02, ge=-0.05, le=0.15)
    vol:         float = Field(0.015, ge=0.0, le=1.0, description="Implied vol of CPI ratio")
    notional:    float = Field(1e6,  ge=1.0)
    infl_rate:   float = Field(0.025, ge=-0.05, le=0.15)
    sofr_on:     float = Field(0.0433, ge=0.0, le=0.15)
    is_cap:      bool  = Field(True, description="True = cap, False = floor")
    freq:        int   = Field(1, ge=1, le=4)


def _build_infl(sofr_on: float, infl_rate: float, maturity: float):
    curve = flat_sofr_curve(date.today(), sofr_on)
    infl_curve = _InflCurve.flat(infl_rate, max_tenor=maturity + 5.0)
    return curve, infl_curve


@app.post("/inflation/zc-price", tags=["Inflation"])
def inflation_zc_price(req: ZCInflationRequest):
    """Price a Zero-Coupon Inflation-Linked Swap."""
    try:
        curve, infl_curve = _build_infl(req.sofr_on, req.infl_rate, req.maturity)
        swap = _ZCSwap(
            maturity=req.maturity,
            fixed_rate=req.fixed_rate,
            notional=req.notional,
            receive_inflation=not req.is_receiver,
        )
        result = _zc_pv(curve, infl_curve, swap)
        be_info = _breakeven(curve, infl_curve, req.maturity)
        return {
            "pv":                round(float(result.pv), 2),
            "inflation_leg_pv":  round(float(result.float_leg_pv), 2),
            "fixed_leg_pv":      round(float(result.fixed_leg_pv), 2),
            "breakeven_rate":    round(float(be_info["breakeven_inflation"]), 6),
            "breakeven_bps":     round(float(be_info["breakeven_inflation"]) * 1e4, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/inflation/yoy-price", tags=["Inflation"])
def inflation_yoy_price(req: YoYInflationRequest):
    """Price a Year-on-Year Inflation-Linked Swap."""
    try:
        curve, infl_curve = _build_infl(req.sofr_on, req.infl_rate, req.maturity)
        dt = 1.0 / req.freq
        payment_dates = list(np.arange(dt, req.maturity + 1e-10, dt))
        swap = _YoYSwap(
            payment_dates=payment_dates,
            fixed_rate=req.fixed_rate,
            notional=req.notional,
            receive_inflation=not req.is_receiver,
        )
        result = _yoy_pv(curve, infl_curve, swap)
        return {
            "pv":               round(float(result.pv), 2),
            "inflation_leg_pv": round(float(result.float_leg_pv), 2),
            "fixed_leg_pv":     round(float(result.fixed_leg_pv), 2),
            "n_payments":       result.n_periods,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/inflation/cap-floor-price", tags=["Inflation"])
def inflation_cap_floor_price(req: InflationCapFloorRequest):
    """Price an Inflation Cap or Floor strip using Black-76 on CPI ratios."""
    try:
        curve, infl_curve = _build_infl(req.sofr_on, req.infl_rate, req.maturity)
        dt = 1.0 / req.freq
        payment_dates = list(np.arange(dt, req.maturity + 1e-10, dt))
        cap_floor = _InflCapFloor(
            payment_dates=payment_dates,
            strike=req.strike,
            vol=req.vol,
            notional=req.notional,
            is_cap=req.is_cap,
        )
        result = _infl_cf_pv(curve, infl_curve, cap_floor)
        return {
            "pv":         round(float(result.pv), 2),
            "n_caplets":  len(result.caplet_pvs),
            "is_cap":     req.is_cap,
            "strike_pct": round(req.strike * 100, 4),
            "vol_pct":    round(req.vol * 100, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/inflation/breakeven", tags=["Inflation"])
def inflation_breakeven(
    sofr_on:    float = 0.0433,
    infl_rate:  float = 0.025,
    maturity:   float = 10.0,
):
    """Return the breakeven inflation rate implied by nominal vs. real yields."""
    try:
        curve, infl_curve = _build_infl(sofr_on, infl_rate, maturity)
        be_info = _breakeven(curve, infl_curve, maturity)
        return {
            "maturity_years":    maturity,
            "breakeven_rate":    round(float(be_info["breakeven_inflation"]), 6),
            "breakeven_bps":     round(float(be_info["breakeven_inflation"]) * 1e4, 2),
            "nominal_sofr":      sofr_on,
            "real_infl_rate":    infl_rate,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Callable Bond (Hull-White Trinomial Tree + OAS)
# ══════════════════════════════════════════════════════════════════════════════
from sofr_engine.callable_bond import (
    HWTreeParams as _HWTree,
    CallableBond as _CBond,
    price_callable_bond as _price_cb,
    straight_bond_price as _straight_cb,
    calibrate_oas as _cal_oas,
    effective_duration as _eff_dur,
    effective_convexity as _eff_cvx,
    price_callable_bond_full as _cb_full,
)


class CallableBondRequest(BaseModel):
    face:          float = Field(100.0,  ge=1.0,   description="Face value")
    coupon:        float = Field(0.06,   ge=0.0, le=0.30, description="Annual coupon rate (decimal)")
    maturity:      float = Field(5.0,    ge=0.5, le=30.0, description="Maturity in years")
    freq:          int   = Field(2,      ge=1,   le=4,    description="Coupon freq per year")
    sofr_on:       float = Field(0.0433, ge=0.0, le=0.15, description="SOFR overnight rate")
    hw_a:          float = Field(0.10,   ge=1e-4, le=2.0, description="HW mean reversion")
    hw_sigma:      float = Field(0.01,   ge=1e-4, le=0.20, description="HW short-rate vol")
    dt:            float = Field(0.25,   ge=0.01, le=1.0,  description="Tree time step (years)")
    call_schedule: list  = Field(default_factory=list, description="[{time, price}, …]")
    put_schedule:  list  = Field(default_factory=list, description="[{time, price}, …]")
    market_price:  float | None = Field(None, description="If set, calibrates OAS")


def _parse_schedule(sched: list) -> list:
    return [(float(s["time"]), float(s["price"])) for s in sched]


@app.post("/callable-bond/price", tags=["Callable Bond"])
def callable_bond_price(req: CallableBondRequest):
    """Price a callable/putable bond on a Hull-White trinomial tree, with optional OAS calibration."""
    try:
        curve = flat_sofr_curve(date.today(), req.sofr_on)
        hw    = _HWTree(a=req.hw_a, sigma=req.hw_sigma, dt=req.dt)
        bond  = _CBond(
            face=req.face, coupon=req.coupon, maturity=req.maturity, freq=req.freq,
            call_schedule=_parse_schedule(req.call_schedule),
            put_schedule=_parse_schedule(req.put_schedule),
        )
        res = _cb_full(curve, hw, bond, market_price=req.market_price)
        return {
            "price":               round(res.price, 4),
            "straight_price":      round(res.straight_price, 4),
            "option_value":        round(res.option_value, 4),
            "oas_bps":             round(res.oas * 1e4, 2),
            "effective_duration":  round(res.effective_duration, 4),
            "effective_convexity": round(res.effective_convexity, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/callable-bond/straight-price", tags=["Callable Bond"])
def callable_bond_straight(
    sofr_on: float = 0.0433,
    coupon:  float = 0.06,
    maturity: float = 5.0,
    freq:    int   = 2,
    face:    float = 100.0,
):
    """Analytical straight bond price (no optionality)."""
    try:
        curve = flat_sofr_curve(date.today(), sofr_on)
        bond  = _CBond(face=face, coupon=coupon, maturity=maturity, freq=freq)
        p = _straight_cb(curve, bond)
        return {"straight_price": round(p, 4), "face": face,
                "coupon_pct": round(coupon * 100, 4), "maturity": maturity}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# FX Options (Garman-Kohlhagen + Vanna-Volga Surface)
# ══════════════════════════════════════════════════════════════════════════════
from sofr_engine.fx_options import (
    FXOptionParams as _FXParams,
    gk_price as _gk_price,
    gk_greeks as _gk_greeks,
    gk_implied_vol as _gk_iv,
    FXVolSurface as _FXSurface,
    vol_for_strike as _vfs,
    fx_smile as _fx_smile,
)


class FXOptionRequest(BaseModel):
    spot:          float = Field(1.09,  ge=0.01, le=1000.0, description="Spot FX rate")
    strike:        float = Field(1.09,  ge=0.01, le=1000.0, description="Strike")
    vol:           float = Field(0.08,  ge=0.001, le=5.0,   description="Implied vol (decimal)")
    domestic_rate: float = Field(0.04,  ge=-0.1, le=0.2,   description="Domestic rate (USD)")
    foreign_rate:  float = Field(0.03,  ge=-0.1, le=0.2,   description="Foreign rate (EUR)")
    maturity:      float = Field(1.0,   ge=0.01, le=5.0,   description="Maturity (years)")
    is_call:       bool  = Field(True,                      description="Call or put")


class FXSmileRequest(BaseModel):
    maturities:    list  = Field([0.25, 0.5, 1.0, 2.0], description="Tenors in years")
    atm_vols:      list  = Field([0.07, 0.08, 0.09, 0.10])
    rr25:          list  = Field([0.002, 0.003, 0.004, 0.005], description="25Δ risk reversals")
    bf25:          list  = Field([0.001, 0.001, 0.002, 0.002], description="25Δ butterflies")
    spot:          float = Field(1.09,  ge=0.01, le=1000.0)
    domestic_rate: float = Field(0.04,  ge=-0.1, le=0.2)
    foreign_rate:  float = Field(0.03,  ge=-0.1, le=0.2)
    target_mat:    float = Field(1.0,   ge=0.01, le=10.0, description="Target maturity for smile")
    n_strikes:     int   = Field(21,    ge=5,   le=101)


@app.post("/fx/price", tags=["FX Options"])
def fx_option_price(req: FXOptionRequest):
    """Price a vanilla FX option and return full Garman-Kohlhagen Greeks."""
    try:
        p = _FXParams(
            spot=req.spot, strike=req.strike, vol=req.vol,
            domestic_rate=req.domestic_rate, foreign_rate=req.foreign_rate,
            maturity=req.maturity, is_call=req.is_call,
        )
        res = _gk_greeks(p)
        return {
            "pv":     round(res.pv, 6),
            "delta":  round(res.delta, 6),
            "gamma":  round(res.gamma, 6),
            "vega":   round(res.vega, 6),
            "theta":  round(res.theta, 6),
            "rho_d":  round(res.rho_d, 6),
            "rho_f":  round(res.rho_f, 6),
            "vanna":  round(res.vanna, 6),
            "volga":  round(res.volga, 6),
            "is_call": req.is_call,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/fx/implied-vol", tags=["FX Options"])
def fx_implied_vol(req: FXOptionRequest):
    """Back out implied vol from a market price using Newton-Raphson."""
    try:
        p = _FXParams(
            spot=req.spot, strike=req.strike, vol=req.vol,
            domestic_rate=req.domestic_rate, foreign_rate=req.foreign_rate,
            maturity=req.maturity, is_call=req.is_call,
        )
        market_price = _gk_price(p)  # use vol as market price for demo
        iv = _gk_iv(
            market_price, req.spot, req.strike,
            req.domestic_rate, req.foreign_rate,
            req.maturity, req.is_call,
        )
        return {
            "implied_vol":     round(iv, 6),
            "implied_vol_pct": round(iv * 100, 4),
            "market_price":    round(market_price, 6),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/fx/smile", tags=["FX Options"])
def fx_vol_smile(req: FXSmileRequest):
    """Return the Vanna-Volga smile (strikes, vols) at a target maturity."""
    try:
        surface = _FXSurface(
            maturities=req.maturities,
            atm_vols=req.atm_vols,
            rr25=req.rr25,
            bf25=req.bf25,
            spot=req.spot,
            domestic_rate=req.domestic_rate,
            foreign_rate=req.foreign_rate,
        )
        strikes, vols = _fx_smile(surface, req.target_mat, req.n_strikes)
        return {
            "strikes":    [round(float(k), 6) for k in strikes],
            "vols_pct":   [round(float(v) * 100, 4) for v in vols],
            "target_mat": req.target_mat,
            "n_strikes":  req.n_strikes,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

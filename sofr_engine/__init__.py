"""
SOFR Pricing Engine
===================
A from-scratch implementation of SOFR curve construction and instrument pricing.

Modules
-------
day_count      : Day count conventions (ACT/360, ACT/ACT)
curve          : DiscountCurve — discount factors, zero rates, forward rates
bootstrap      : Bootstrap SOFR forward curve from CME futures + OIS swaps
convexity      : Hull-White convexity adjustment for futures vs. OIS
instruments    : SOFR swap, FRA, and futures pricers
cap_floor      : Cap/Floor strip pricing — Black-76, Bachelier, vol bootstrap
monte_carlo    : Hull-White 1F exact MC — ZCB/caplet pricing, portfolio VaR
sabr           : SABR stochastic-vol model — smile, skew, calibration
"""
from .curve import DiscountCurve
from .bootstrap import SOFRCurveBootstrapper
from .instruments import SOFRSwap, ForwardRateAgreement, SOFRFutures
from .convexity import hull_white_convexity_adjustment
from .risk import ScenarioEngine, RiskReport
from .swaption import Swaption, SwaptionVolSurface, price_swaption
from .sabr import SABRParams, SABRSurface, sabr_implied_vol, sabr_vol_smile, calibrate_sabr
from .monte_carlo import (
    HullWhiteParams, SimulationResult,
    simulate_hw, zcb_price_hw,
    price_zcb_mc, price_caplet_mc,
    portfolio_var_hw, parametric_var,
    convergence_diagnostics,
)
from .cap_floor import (
    Caplet, Cap, Floor,
    caplet_black_pv, caplet_bachelier_pv,
    cap_floor_parity_pv,
    CapFloorVolSurface, strip_caplet_vols, price_cap_floor,
)

__all__ = [
    "DiscountCurve",
    "SOFRCurveBootstrapper",
    "SOFRSwap",
    "ForwardRateAgreement",
    "SOFRFutures",
    "hull_white_convexity_adjustment",
    "ScenarioEngine",
    "RiskReport",
    "Swaption",
    "SwaptionVolSurface",
    "price_swaption",
    "SABRParams",
    "SABRSurface",
    "sabr_implied_vol",
    "sabr_vol_smile",
    "calibrate_sabr",
    "HullWhiteParams",
    "SimulationResult",
    "simulate_hw",
    "zcb_price_hw",
    "price_zcb_mc",
    "price_caplet_mc",
    "portfolio_var_hw",
    "parametric_var",
    "convergence_diagnostics",
    "Caplet",
    "Cap",
    "Floor",
    "caplet_black_pv",
    "caplet_bachelier_pv",
    "cap_floor_parity_pv",
    "CapFloorVolSurface",
    "strip_caplet_vols",
    "price_cap_floor",
]

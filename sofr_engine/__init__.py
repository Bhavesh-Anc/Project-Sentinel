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
bermudan       : Bermudan swaption — Longstaff-Schwartz LSM + European HW MC
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
from .bermudan import (
    BermudanSwaptionResult, price_bermudan_swaption,
    price_european_swaption_hw,
)
from .cap_floor import (
    Caplet, Cap, Floor,
    caplet_black_pv, caplet_bachelier_pv,
    cap_floor_parity_pv,
    CapFloorVolSurface, strip_caplet_vols, price_cap_floor,
)
from .credit import (
    HazardRateCurve,
    CDSContract,
    CDSResult,
    bootstrap_hazard_curve,
    cds_pv,
    cds_par_spread,
    cds_cs01,
    cds_dv01,
    risky_annuity,
)
from .g2pp import (
    G2ppParams,
    G2ppSimResult,
    g2pp_zcb,
    g2pp_inst_forward,
    simulate_g2pp,
    g2pp_swaption,
    g2pp_swaption_mc,
    g2pp_portfolio_var,
    calibrate_g2pp,
)
from .cms import (
    CMSConvexityResult,
    CMSCaplet,
    CMSSpreadOption,
    CMSSwap,
    cms_convexity_adj,
    cms_caplet_pv,
    cms_floorlet_pv,
    cms_caplet_floorlet_parity,
    cms_spread_option_pv,
)
from .xva import (
    XVAParams,
    EPEProfile,
    XVAResult,
    compute_epe_profile,
    cva,
    dva,
    fva,
    full_xva,
    cva_sensitivity,
)
from .xccy import (
    FXForwardCurve,
    CrossCurrencySwap,
    XCCYResult,
    fx_forward,
    xccy_par_basis,
    xccy_swap_pv,
    xccy_dv01,
    xccy_cs01,
    build_eur_curve_from_xccy,
    xccy_basis_term_structure,
)
from .inflation import (
    InflationCurve,
    ZCInflationSwap,
    YoYInflationSwap,
    InflationCapFloor,
    ZCInflationResult,
    YoYResult,
    InflationCapResult,
    zc_inflation_pv,
    yoy_inflation_pv,
    inflation_caplet_pv,
    inflation_cap_floor_pv,
    inflation_cap_floor_parity,
    breakeven_inflation,
    calibrate_inflation_curve,
)
from .lmm import (
    LMMParams,
    LMMSimResult,
    initial_forwards,
    exponential_correlation,
    simulate_lmm,
    caplet_black76,
    cap_black76,
    cap_implied_vol,
    caplet_lmm_mc,
    swaption_lmm_mc,
    rebonato_swaption_vol,
    calibrate_caplet_vols,
    calibrate_corr_decay,
    swaption_implied_vol,
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
    "BermudanSwaptionResult",
    "price_bermudan_swaption",
    "price_european_swaption_hw",
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
    "HazardRateCurve",
    "CDSContract",
    "CDSResult",
    "bootstrap_hazard_curve",
    "cds_pv",
    "cds_par_spread",
    "cds_cs01",
    "cds_dv01",
    "risky_annuity",
    "G2ppParams",
    "G2ppSimResult",
    "g2pp_zcb",
    "g2pp_inst_forward",
    "simulate_g2pp",
    "g2pp_swaption",
    "g2pp_swaption_mc",
    "g2pp_portfolio_var",
    "calibrate_g2pp",
    "CMSConvexityResult",
    "CMSCaplet",
    "CMSSpreadOption",
    "CMSSwap",
    "cms_convexity_adj",
    "cms_caplet_pv",
    "cms_floorlet_pv",
    "cms_caplet_floorlet_parity",
    "cms_spread_option_pv",
    "LMMParams",
    "LMMSimResult",
    "initial_forwards",
    "exponential_correlation",
    "simulate_lmm",
    "caplet_black76",
    "cap_black76",
    "cap_implied_vol",
    "caplet_lmm_mc",
    "swaption_lmm_mc",
    "rebonato_swaption_vol",
    "calibrate_caplet_vols",
    "calibrate_corr_decay",
    "swaption_implied_vol",
    # XVA
    "XVAParams",
    "EPEProfile",
    "XVAResult",
    "compute_epe_profile",
    "cva",
    "dva",
    "fva",
    "full_xva",
    "cva_sensitivity",
    # XCCY
    "FXForwardCurve",
    "CrossCurrencySwap",
    "XCCYResult",
    "fx_forward",
    "xccy_par_basis",
    "xccy_swap_pv",
    "xccy_dv01",
    "xccy_cs01",
    "build_eur_curve_from_xccy",
    "xccy_basis_term_structure",
    # Inflation
    "InflationCurve",
    "ZCInflationSwap",
    "YoYInflationSwap",
    "InflationCapFloor",
    "ZCInflationResult",
    "YoYResult",
    "InflationCapResult",
    "zc_inflation_pv",
    "yoy_inflation_pv",
    "inflation_caplet_pv",
    "inflation_cap_floor_pv",
    "inflation_cap_floor_parity",
    "breakeven_inflation",
    "calibrate_inflation_curve",
]

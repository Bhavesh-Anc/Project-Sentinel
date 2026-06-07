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
sabr           : SABR stochastic-vol model — smile, skew, calibration
"""
from .curve import DiscountCurve
from .bootstrap import SOFRCurveBootstrapper
from .instruments import SOFRSwap, ForwardRateAgreement, SOFRFutures
from .convexity import hull_white_convexity_adjustment
from .risk import ScenarioEngine, RiskReport
from .swaption import Swaption, SwaptionVolSurface, price_swaption
from .sabr import SABRParams, SABRSurface, sabr_implied_vol, sabr_vol_smile, calibrate_sabr

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
]

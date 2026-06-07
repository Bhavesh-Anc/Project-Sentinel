"""
SABR Stochastic Volatility Model — Hagan, Kumar, Lesniewski & Woodward (2002).

The SABR model describes the dynamics of a forward rate F under risk-neutral measure:

    dF = σ · F^β · dW
    dσ = ν · σ · dZ
    dW · dZ = ρ · dt

where σ (= alpha at t=0) is the initial vol level, β is the backbone exponent,
ρ is the Brownian correlation, and ν is the vol-of-vol.

Black-76 implied vol approximation (Hagan et al. 2002, Eq. 2.17b):
For F ≠ K:

    z    = (ν/α) · (F·K)^((1−β)/2) · ln(F/K)
    χ    = ln[(√(1 − 2ρz + z²) + z − ρ) / (1 − ρ)]

    A(F,K,β) = (F·K)^((1−β)/2) · [1 + ((1−β)²/24)·ln²(F/K) + ((1−β)⁴/1920)·ln⁴(F/K)]

    σ_B = α / A(F,K,β) · (z/χ) · num_corr(T)

where the time-dependent numerator correction is:
    num_corr = 1 + [(1−β)²α² / (24(FK)^(1−β))
                   + ρβνα / (4(FK)^((1−β)/2))
                   + (2−3ρ²)ν²/24] · T

For F = K (ATM):
    σ_B_ATM = α / F^(1−β) · num_corr_ATM(T)

where the ATM numerator correction substitutes F for (FK)^(1/2):
    num_corr_ATM = 1 + [(1−β)²α² / (24F^(2(1−β)))
                        + ρβνα / (4F^(1−β))
                        + (2−3ρ²)ν²/24] · T

Parameters
----------
alpha : α > 0    — initial vol level (roughly ATM vol for β=1)
beta  : 0 ≤ β ≤ 1 — backbone: 0=normal, 0.5=CIR-like, 1=log-normal
rho   : −1 < ρ < 1 — correlation; negative ρ creates a put skew (payer OTM < ATM)
nu    : ν ≥ 0    — vol-of-vol; controls smile curvature

USD Swaption convention
-----------------------
β = 0.5 (fixed), ρ ≈ −0.25 (put skew), ν ≈ 0.40 (moderate curvature).

Shifted SABR
------------
Standard SABR is undefined for F ≤ 0 or K ≤ 0.  For ZLB environments use a
rate shift (not implemented here) — a ValueError is raised for non-positive inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence

from scipy.optimize import minimize, brentq


# ── Parameter container ───────────────────────────────────────────────────────

@dataclass
class SABRParams:
    """
    SABR model parameters.

    Attributes
    ----------
    alpha : Initial vol level (α > 0).
    beta  : Backbone exponent (0 ≤ β ≤ 1).  Fix at 0.5 for USD swaptions.
    rho   : Forward–vol correlation (−1 < ρ < 1).  Negative → put skew.
    nu    : Vol-of-vol (ν ≥ 0).  Controls smile curvature.
    """
    alpha: float
    beta:  float
    rho:   float
    nu:    float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.nu < 0:
            raise ValueError(f"nu must be >= 0, got {self.nu}")


# ── Core pricing functions ────────────────────────────────────────────────────

def sabr_implied_vol(F: float, K: float, T: float, params: SABRParams) -> float:
    """
    Hagan et al. (2002) Black-76 implied vol approximation.

    Parameters
    ----------
    F      : Forward rate (positive; e.g. 0.0453 for 4.53%).
    K      : Strike (positive).
    T      : Time to expiry in years (must be > 0 for a meaningful vol).
    params : SABRParams with (alpha, beta, rho, nu).

    Returns
    -------
    Black-76 implied vol as a decimal (e.g. 0.20 for 20%).

    Raises
    ------
    ValueError
        If F ≤ 0 or K ≤ 0 (SABR is undefined; use shifted-SABR for ZLB).
    """
    if F <= 0.0:
        raise ValueError(f"Forward rate F must be > 0, got {F}")
    if K <= 0.0:
        raise ValueError(f"Strike K must be > 0, got {K}")
    if T <= 0.0:
        return 0.0

    alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu
    one_minus_beta = 1.0 - beta

    # ── Near-ATM branch (avoids 0/0 in z/chi) ───────────────────────────────
    # Use ATM formula when |ln(F/K)| < 1e-7 (≈ 0.001 bp in rate space)
    if abs(np.log(F / K)) < 1e-7:
        return _sabr_atm_vol(F, T, params)

    log_fk = np.log(F / K)
    fk_mid = F * K

    # Denominator correction: accounts for higher-order ln(F/K)^2 terms
    log_fk2 = log_fk ** 2
    denom_corr = (
        1.0
        + (one_minus_beta ** 2 / 24.0) * log_fk2
        + (one_minus_beta ** 4 / 1920.0) * log_fk2 ** 2
    )

    # Backbone: (FK)^((1−β)/2)
    fk_backbone = fk_mid ** (one_minus_beta / 2.0)

    # z and chi mapping
    z = (nu / alpha) * fk_backbone * log_fk
    sqrt_discriminant = np.sqrt(1.0 - 2.0 * rho * z + z ** 2)
    chi = np.log((sqrt_discriminant + z - rho) / (1.0 - rho))

    # Taylor expansion of z/chi near z=0 (near-ATM) to avoid numerical cancellation
    if abs(z) < 1e-6:
        z_over_chi = 1.0 + (rho / 2.0) * z + ((2.0 - 3.0 * rho ** 2) / 12.0) * z ** 2
    else:
        z_over_chi = z / chi

    # Time-dependent numerator correction (Eq. 2.17b)
    fk_one_minus_beta = fk_mid ** one_minus_beta
    num_corr = 1.0 + (
        (one_minus_beta ** 2 * alpha ** 2) / (24.0 * fk_one_minus_beta)
        + (rho * beta * nu * alpha) / (4.0 * fk_backbone)
        + ((2.0 - 3.0 * rho ** 2) / 24.0) * nu ** 2
    ) * T

    return alpha / (fk_backbone * denom_corr) * z_over_chi * num_corr


def _sabr_atm_vol(F: float, T: float, params: SABRParams) -> float:
    """
    ATM (F = K) Black-76 SABR vol using Eq. 2.17a of Hagan et al. (2002).

    L'Hôpital limit gives z/chi → 1 as K → F; the backbone simplifies to F^(1−β).
    """
    alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu
    one_minus_beta = 1.0 - beta

    f_backbone = F ** one_minus_beta

    num_corr_atm = 1.0 + (
        (one_minus_beta ** 2 * alpha ** 2) / (24.0 * F ** (2.0 * one_minus_beta))
        + (rho * beta * nu * alpha) / (4.0 * f_backbone)
        + ((2.0 - 3.0 * rho ** 2) / 24.0) * nu ** 2
    ) * T

    return (alpha / f_backbone) * num_corr_atm


def sabr_normal_vol(F: float, K: float, T: float, params: SABRParams) -> float:
    """
    Bachelier (normal) implied vol from SABR.

    Uses the at-the-money approximation:
        σ_normal ≈ σ_black · √(F · K)

    This is exact at K = F and a good approximation for near-ATM strikes with β ≠ 1.
    For β = 1 (log-normal backbone) σ_normal ≈ σ_black · F exactly at ATM.

    Parameters
    ----------
    F      : Forward rate (positive).
    K      : Strike (positive).
    T      : Time to expiry in years.
    params : SABRParams.

    Returns
    -------
    Normal (Bachelier) implied vol as a decimal.
    """
    sigma_black = sabr_implied_vol(F, K, T, params)
    return sigma_black * np.sqrt(F * K)


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_sabr(
    F: float,
    T: float,
    strikes: list[float],
    market_vols: list[float],
    beta: float = 0.5,
    initial_guess: tuple = (0.05, -0.25, 0.40),
) -> SABRParams:
    """
    Fit SABR parameters {alpha, rho, nu} to market Black-76 vol quotes.

    beta is held fixed (set to 0.5 for USD swaptions by convention).
    Minimises the sum of squared vol errors (RMSE in vol space) via L-BFGS-B.

    Parameters
    ----------
    F             : Forward rate (positive decimal).
    T             : Option expiry in years.
    strikes       : List of strike rates (positive decimals).
    market_vols   : Corresponding Black-76 implied vols (decimal).
    beta          : Fixed backbone exponent (default 0.5).
    initial_guess : (alpha0, rho0, nu0) starting values.

    Returns
    -------
    SABRParams
        Calibrated parameters with fixed beta.

    Raises
    ------
    ValueError
        If the optimiser fails to converge or the recovered vol surface has
        NaN values (e.g. degenerate parameter combination).
    """
    strikes_arr = np.asarray(strikes, dtype=float)
    mkt_arr     = np.asarray(market_vols, dtype=float)

    if len(strikes_arr) != len(mkt_arr):
        raise ValueError("strikes and market_vols must have the same length.")
    if len(strikes_arr) == 0:
        raise ValueError("At least one strike/vol pair is required.")

    alpha0, rho0, nu0 = initial_guess

    # When only a single ATM quote is available, anchor alpha to match that vol
    # and keep rho/nu at their initial values (the system is under-determined).
    if len(strikes_arr) == 1 and abs(strikes_arr[0] - F) / max(abs(F), 1e-8) < 1e-4:
        target_atm = mkt_arr[0]

        # Bracket: alpha ∈ (1e-6, 10) should always contain the root
        lo, hi = 1e-6, 10.0

        def _atm_error(a: float) -> float:
            try:
                p = SABRParams(alpha=a, beta=beta, rho=rho0, nu=nu0)
                return _sabr_atm_vol(F, T, p) - target_atm
            except ValueError:
                return 1e10

        try:
            alpha_cal = brentq(_atm_error, lo, hi, xtol=1e-10, rtol=1e-10)
        except ValueError as exc:
            raise ValueError(
                f"Single-strike ATM alpha calibration failed to bracket root: {exc}"
            ) from exc

        return SABRParams(alpha=alpha_cal, beta=beta, rho=rho0, nu=nu0)

    def _objective(x: np.ndarray) -> float:
        alpha, rho, nu = x
        try:
            p = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
        except ValueError:
            return 1e10
        sse = 0.0
        for k, mv in zip(strikes_arr, mkt_arr):
            try:
                model_vol = sabr_implied_vol(F, k, T, p)
                sse += (model_vol - mv) ** 2
            except (ValueError, FloatingPointError):
                sse += 1e4
        return sse

    bounds = [
        (1e-6,  None),    # alpha > 0
        (-0.999, 0.999),  # rho ∈ (-1, 1)
        (0.0,   None),    # nu ≥ 0
    ]

    result = minimize(
        _objective,
        x0=np.array([alpha0, rho0, nu0]),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000},
    )

    if not result.success and result.fun > 1e-8:
        raise ValueError(
            f"SABR calibration did not converge: {result.message} "
            f"(residual RMSE = {np.sqrt(result.fun / len(strikes_arr)) * 1e4:.4f} bp)"
        )

    alpha_cal, rho_cal, nu_cal = result.x

    # Clamp to valid domain (L-BFGS-B bounds are not hard constraints)
    alpha_cal = max(alpha_cal, 1e-8)
    rho_cal   = float(np.clip(rho_cal, -0.9999, 0.9999))
    nu_cal    = max(nu_cal, 0.0)

    return SABRParams(alpha=alpha_cal, beta=beta, rho=rho_cal, nu=nu_cal)


# ── Vol smile DataFrame ───────────────────────────────────────────────────────

def sabr_vol_smile(
    F: float,
    T: float,
    params: SABRParams,
    n_strikes: int = 21,
    strike_range_bps: float = 200.0,
) -> pd.DataFrame:
    """
    Compute the SABR vol smile across a symmetric strike grid around ATM.

    Parameters
    ----------
    F                : Forward rate (decimal, e.g. 0.0453).
    T                : Expiry in years.
    params           : Calibrated SABRParams.
    n_strikes        : Number of strike grid points (odd → symmetric around ATM).
    strike_range_bps : Half-width of the strike grid in basis points.

    Returns
    -------
    pd.DataFrame with columns:
        strike_pct     — strike as a percentage (e.g. 4.53 for K=0.0453)
        moneyness_bps  — K − F in basis points (negative = OTM receiver)
        sabr_vol_pct   — Black-76 SABR vol in percent
        normal_vol_bps — Bachelier normal vol in basis points
    """
    delta_k = strike_range_bps / 10_000.0
    strikes = np.linspace(F - delta_k, F + delta_k, n_strikes)
    # Ensure all strikes are positive (floor at 1 bp so SABR is defined)
    strikes = np.maximum(strikes, 1e-4)

    rows = []
    for K in strikes:
        bvol = sabr_implied_vol(F, K, T, params)
        nvol = sabr_normal_vol(F, K, T, params)
        rows.append({
            "strike_pct":     K * 100.0,
            "moneyness_bps":  (K - F) * 10_000.0,
            "sabr_vol_pct":   bvol * 100.0,
            "normal_vol_bps": nvol * 10_000.0,
        })

    return pd.DataFrame(rows)


# ── SABR vol surface ──────────────────────────────────────────────────────────

class SABRSurface:
    """
    SABR vol surface: calibrated SABRParams for each (expiry, tenor) node.

    The surface holds one SABRParams per (expiry, tenor) grid point.  The
    ``vol`` method performs bilinear interpolation of the four SABR parameters
    individually, then evaluates sabr_implied_vol at the interpolated params.
    This is an approximation — exact interpolation would require interpolating
    option prices — but it is standard practice for swaption vol surfaces.

    Parameters
    ----------
    expiries : 1-D sequence of option expiries in years.
    tenors   : 1-D sequence of swap tenors in years.
    params   : 2-D list of SABRParams, shape (n_expiries, n_tenors).
    """

    def __init__(
        self,
        expiries: Sequence[float],
        tenors:   Sequence[float],
        params:   list[list[SABRParams]],
    ) -> None:
        self._expiries = np.asarray(expiries, dtype=float)
        self._tenors   = np.asarray(tenors,   dtype=float)
        self._params   = params  # row = expiry, col = tenor

        if len(self._params) != len(self._expiries):
            raise ValueError("params row count must equal len(expiries)")
        for row in self._params:
            if len(row) != len(self._tenors):
                raise ValueError("params column count must equal len(tenors)")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _bracket(self, arr: np.ndarray, x: float) -> tuple[int, int, float]:
        """Return (i1, i2, weight) for linear interpolation of x in arr."""
        x_clamped = float(np.clip(x, arr[0], arr[-1]))
        i1 = int(np.searchsorted(arr, x_clamped, side="right")) - 1
        i1 = int(np.clip(i1, 0, len(arr) - 2))
        i2 = i1 + 1
        span = arr[i2] - arr[i1]
        w = (x_clamped - arr[i1]) / span if span > 1e-14 else 0.0
        return i1, i2, float(w)

    def _interp_params(self, expiry: float, tenor: float) -> SABRParams:
        """Bilinear interpolation of the four SABR scalars across the grid."""
        i1, i2, we = self._bracket(self._expiries, expiry)
        j1, j2, wt = self._bracket(self._tenors,   tenor)

        p11 = self._params[i1][j1]
        p12 = self._params[i1][j2]
        p21 = self._params[i2][j1]
        p22 = self._params[i2][j2]

        def _bilin(v11: float, v12: float, v21: float, v22: float) -> float:
            return (
                v11 * (1 - we) * (1 - wt)
                + v12 * (1 - we) * wt
                + v21 * we * (1 - wt)
                + v22 * we * wt
            )

        alpha = _bilin(p11.alpha, p12.alpha, p21.alpha, p22.alpha)
        beta  = _bilin(p11.beta,  p12.beta,  p21.beta,  p22.beta)
        rho   = _bilin(p11.rho,   p12.rho,   p21.rho,   p22.rho)
        nu    = _bilin(p11.nu,    p12.nu,    p21.nu,    p22.nu)

        # Clamp to valid domain after interpolation
        alpha = max(alpha, 1e-8)
        beta  = float(np.clip(beta, 0.0, 1.0))
        rho   = float(np.clip(rho, -0.9999, 0.9999))
        nu    = max(nu, 0.0)

        return SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)

    # ── Public interface ──────────────────────────────────────────────────────

    def vol(self, expiry: float, tenor: float, K: float, F: float) -> float:
        """
        SABR Black-76 vol at a given (expiry, tenor, strike, forward) point.

        Bilinearly interpolates SABR parameters across the surface grid, then
        evaluates sabr_implied_vol with the interpolated params.

        Parameters
        ----------
        expiry : Option expiry in years.
        tenor  : Swap tenor in years.
        K      : Strike (positive decimal).
        F      : Forward rate (positive decimal).

        Returns
        -------
        Black-76 implied vol (decimal).
        """
        p = self._interp_params(expiry, tenor)
        return sabr_implied_vol(F, K, expiry, p)

    def smile_df(self, expiry: float, tenor: float, F: float) -> pd.DataFrame:
        """
        Full SABR vol smile at a given surface node.

        Parameters
        ----------
        expiry : Option expiry in years.
        tenor  : Swap tenor in years.
        F      : Forward rate at that node (positive decimal).

        Returns
        -------
        pd.DataFrame — same schema as :func:`sabr_vol_smile`.
        """
        p = self._interp_params(expiry, tenor)
        return sabr_vol_smile(F, expiry, p)

    @classmethod
    def calibrate_from_atm_surface(
        cls,
        vol_surface: "SwaptionVolSurface",  # type: ignore[name-defined]
        curve: "DiscountCurve",             # type: ignore[name-defined]
        beta: float = 0.5,
        rho:  float = -0.25,
        nu:   float = 0.40,
    ) -> "SABRSurface":
        """
        Construct a SABR surface calibrated to ATM vols only.

        For each (expiry, tenor) node, fixes β, ρ, ν at USD swaption
        convention defaults and solves for α so that the SABR ATM formula
        exactly reproduces the market ATM vol.  This yields a surface with
        a realistic put skew and smile curvature while anchored to observable ATM vols.

        Parameters
        ----------
        vol_surface : SwaptionVolSurface with ATM Black-76 vols.
        curve       : DiscountCurve for computing forward swap rates.
        beta        : Fixed backbone exponent (default 0.5).
        rho         : Fixed correlation (default −0.25, typical USD payer skew).
        nu          : Fixed vol-of-vol (default 0.40).

        Returns
        -------
        SABRSurface with one SABRParams per grid node.
        """
        from .swaption import Swaption

        expiries = vol_surface._expiries.tolist()
        tenors   = vol_surface._tenors.tolist()

        params_grid: list[list[SABRParams]] = []

        for i, T in enumerate(expiries):
            row: list[SABRParams] = []
            for j, tenor in enumerate(tenors):
                atm_vol = float(vol_surface._vols[i, j])

                sw = Swaption(
                    expiry_years=T,
                    swap_tenor_years=tenor,
                    strike=0.0,
                )
                try:
                    F = sw.forward_swap_rate(curve)
                except Exception:
                    F = float("nan")

                if not np.isfinite(F) or F <= 0.0:
                    F = curve.par_ois_rate(T + tenor / 2.0)

                if not np.isfinite(F) or F <= 0.0:
                    F = 0.05  # hard fallback: 5%

                p = calibrate_sabr(
                    F=F,
                    T=T,
                    strikes=[F],
                    market_vols=[atm_vol],
                    beta=beta,
                    initial_guess=(atm_vol * F ** (1.0 - beta), rho, nu),
                )
                row.append(p)
            params_grid.append(row)

        return cls(expiries=expiries, tenors=tenors, params=params_grid)

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def expiries(self) -> np.ndarray:
        """Option expiry grid (years)."""
        return self._expiries.copy()

    @property
    def tenors(self) -> np.ndarray:
        """Swap tenor grid (years)."""
        return self._tenors.copy()

    def __repr__(self) -> str:
        return (
            f"SABRSurface(expiries={list(self._expiries)}, "
            f"tenors={list(self._tenors)}, "
            f"shape=({len(self._expiries)}, {len(self._tenors)}))"
        )

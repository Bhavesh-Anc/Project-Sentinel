"""
Nelson-Siegel and Svensson yield curve models.

Nelson-Siegel (1987):
    y(τ) = β₀ + β₁·L(τ) + β₂·C(τ)
where:
    L(τ) = (1 - e^{-λτ}) / (λτ)           [level loading]
    C(τ) = L(τ) - e^{-λτ}                  [curvature loading]

Factor interpretation:
    β₀ : Level  — long-run rate (parallel shift across all maturities)
    β₁ : Slope  — short minus long rate; negative in normal curve
    β₂ : Curvature — belly richness/cheapness; peaks at τ* = 1/λ
    λ  : Decay parameter; controls where curvature peaks

Svensson (1994) extension (2nd hump):
    y(τ) = β₀ + β₁·L(τ,λ₁) + β₂·C(τ,λ₁) + β₃·C(τ,λ₂)
Better captures kinked curves (e.g., post-hike flattening with twin humps).

Typical US Treasury values (2024):
    λ   ≈ 0.50–0.70  (curvature peak at ~1.5–2Y)
    β₀  ≈ 4.0–5.0%   (level — long-term rate)
    β₁  ≈ -1.0 to +2.0% (slope; positive = inverted curve)
    β₂  ≈ ±2.0%       (curvature)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
from dataclasses import dataclass
from typing import Sequence


@dataclass
class NSParams:
    """Nelson-Siegel parameter set."""
    beta0: float   # level
    beta1: float   # slope
    beta2: float   # curvature
    lam:   float   # decay (λ)

    def to_array(self) -> np.ndarray:
        return np.array([self.beta0, self.beta1, self.beta2, self.lam])

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> "NSParams":
        return cls(*arr)

    def __repr__(self) -> str:
        return (f"NSParams(β₀={self.beta0:.4f}, β₁={self.beta1:.4f}, "
                f"β₂={self.beta2:.4f}, λ={self.lam:.4f})")


@dataclass
class SvenssonParams:
    """Svensson (extended Nelson-Siegel) parameter set."""
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    lam1:  float
    lam2:  float

    def to_array(self) -> np.ndarray:
        return np.array([self.beta0, self.beta1, self.beta2, self.beta3, self.lam1, self.lam2])

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> "SvenssonParams":
        return cls(*arr)


# ── Nelson-Siegel loading functions ──────────────────────────────────────────

def ns_loadings(tau: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (L, slope_loading, curvature_loading) for given maturities and λ.
    L is all-ones (level loads uniformly).
    """
    lam_tau = lam * tau
    # Handle τ → 0 limit (L'Hôpital: limit = 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        l_tau = np.where(lam_tau < 1e-8, 1.0, (1 - np.exp(-lam_tau)) / lam_tau)
    c_tau = l_tau - np.exp(-lam_tau)
    level = np.ones_like(tau)
    return level, l_tau, c_tau


def ns_yield(tau: np.ndarray, params: NSParams) -> np.ndarray:
    """Compute NS yields for an array of maturities τ (in years)."""
    level, slope, curv = ns_loadings(tau, params.lam)
    return (params.beta0 * level
            + params.beta1 * slope
            + params.beta2 * curv)


def svensson_loadings(tau: np.ndarray, lam1: float, lam2: float):
    """Return Svensson loadings (level, slope, C1, C2)."""
    l1_tau = lam1 * tau
    l2_tau = lam2 * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = np.where(l1_tau < 1e-8, 1.0, (1 - np.exp(-l1_tau)) / l1_tau)
        s2 = np.where(l2_tau < 1e-8, 1.0, (1 - np.exp(-l2_tau)) / l2_tau)
    c1 = s1 - np.exp(-l1_tau)
    c2 = s2 - np.exp(-l2_tau)
    return np.ones_like(tau), s1, c1, c2


def svensson_yield(tau: np.ndarray, params: SvenssonParams) -> np.ndarray:
    level, s1, c1, c2 = svensson_loadings(tau, params.lam1, params.lam2)
    return (params.beta0 * level
            + params.beta1 * s1
            + params.beta2 * c1
            + params.beta3 * c2)


# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_nelson_siegel(
    maturities: np.ndarray,
    yields: np.ndarray,
    lam_grid: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> NSParams:
    """
    Fit Nelson-Siegel parameters to observed (maturity, yield) pairs.

    Strategy:
    1. Grid search over λ values (λ is non-linear; fix it and do linear regression)
    2. For each λ: OLS on [level, slope_loading, curv_loading] → (β₀, β₁, β₂)
    3. Select λ minimizing weighted SSE

    Parameters
    ----------
    maturities : array of tenors in years (e.g. [0.25, 0.5, 1, 2, 5, 10, 30])
    yields     : observed yields in % (same units as returned)
    lam_grid   : λ values to search; default covers τ* from 0.5Y to 5Y
    weights    : per-maturity fitting weights (default: equal)

    Returns
    -------
    NSParams with best-fit parameters
    """
    tau = np.asarray(maturities, dtype=float)
    y   = np.asarray(yields, dtype=float)

    if lam_grid is None:
        # τ* = 1/λ in [0.5Y, 5Y] → λ in [0.2, 2.0]
        lam_grid = np.linspace(0.20, 2.0, 100)

    w = weights if weights is not None else np.ones(len(y))

    best_sse  = np.inf
    best_params = None

    for lam in lam_grid:
        level, slope, curv = ns_loadings(tau, lam)
        X = np.column_stack([level, slope, curv])

        # Weighted OLS: (XᵀWX)β = XᵀWy
        W  = np.diag(w)
        XtW = X.T @ W
        try:
            betas = np.linalg.solve(XtW @ X, XtW @ y)
        except np.linalg.LinAlgError:
            continue

        y_hat = X @ betas
        sse   = float(w @ (y - y_hat) ** 2)

        if sse < best_sse:
            best_sse    = sse
            best_params = NSParams(betas[0], betas[1], betas[2], lam)

    if best_params is None:
        raise RuntimeError("Nelson-Siegel fitting failed — check input data")

    return best_params


def fit_svensson(
    maturities: np.ndarray,
    yields: np.ndarray,
    initial: SvenssonParams | None = None,
) -> SvenssonParams:
    """
    Fit Svensson parameters using nonlinear least squares (scipy).
    Svensson has 6 parameters; cannot be linearized like NS → full nonlinear opt.
    """
    tau = np.asarray(maturities, dtype=float)
    y   = np.asarray(yields, dtype=float)

    if initial is None:
        # Sensible initial guess for US Treasury curve
        initial = SvenssonParams(4.0, -1.0, 1.0, 1.0, 0.5, 1.5)

    def residuals(params):
        p = SvenssonParams.from_array(params)
        if p.lam1 <= 0.01 or p.lam2 <= 0.01:
            return np.ones(len(y)) * 1e6
        return svensson_yield(tau, p) - y

    result = least_squares(
        residuals,
        initial.to_array(),
        bounds=([-np.inf, -np.inf, -np.inf, -np.inf, 0.01, 0.01], np.inf),
        method="trf",
    )
    return SvenssonParams.from_array(result.x)


# ── Rolling curve factor extraction ──────────────────────────────────────────

def rolling_ns_factors(
    yield_df: pd.DataFrame,
    maturity_cols: list[str],
    maturities: list[float],
    lam_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Fit Nelson-Siegel to each row of a yield DataFrame.
    Returns a DataFrame with columns: [beta0, beta1, beta2, lam, fit_rmse, date].

    Parameters
    ----------
    yield_df     : DataFrame where each row is a date, columns are yield tenors
    maturity_cols: column names corresponding to each tenor (e.g., ['tsy_2y', 'tsy_10y'])
    maturities   : matching list of tenors in years (e.g., [2.0, 10.0])
    """
    records = []
    for date, row in yield_df.iterrows():
        y_obs = row[maturity_cols].values.astype(float)
        valid = ~np.isnan(y_obs)

        if valid.sum() < 4:  # Need at least 4 points to fit 4 NS parameters
            continue

        tau_valid = np.array(maturities)[valid]
        y_valid   = y_obs[valid]

        try:
            params = fit_nelson_siegel(tau_valid, y_valid, lam_grid=lam_grid)
            y_hat  = ns_yield(tau_valid, params)
            rmse   = float(np.sqrt(np.mean((y_valid - y_hat) ** 2)))
            records.append({
                "date":  date,
                "beta0": params.beta0,
                "beta1": params.beta1,
                "beta2": params.beta2,
                "lam":   params.lam,
                "fit_rmse": rmse,
            })
        except Exception:
            continue

    df = pd.DataFrame(records).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def classify_curve_regime(ns_df: pd.DataFrame) -> pd.Series:
    """
    Classify each date's curve regime using NS slope factor β₁.

    β₁ < -100bps  → Normal (steep)
    -100 to 0     → Flat/slightly normal
    0 to +100bps  → Flat/slightly inverted
    > +100bps     → Inverted
    """
    b1 = ns_df["beta1"]
    conditions = [
        b1 < -1.0,
        (b1 >= -1.0) & (b1 < 0.0),
        (b1 >= 0.0)  & (b1 < 1.0),
        b1 >= 1.0,
    ]
    labels = ["normal_steep", "normal_flat", "inverted_slight", "inverted_deep"]
    return pd.Series(
        np.select(conditions, labels, default="unknown"),
        index=ns_df.index,
        name="regime",
    )


if __name__ == "__main__":
    # Fit NS to a typical US Treasury curve (2024-style inverted)
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    yields_pct  = np.array([5.25, 5.20, 5.10, 4.80, 4.60, 4.40, 4.35, 4.30, 4.50, 4.45])

    params = fit_nelson_siegel(maturities, yields_pct)
    print(f"NS fit: {params}")

    y_fit = ns_yield(maturities, params)
    rmse  = np.sqrt(np.mean((yields_pct - y_fit) ** 2))
    print(f"RMSE: {rmse:.4f}%")

    print("\nTenor | Observed | Fitted | Error")
    for tau, obs, fit in zip(maturities, yields_pct, y_fit):
        print(f"{tau:5.2f}Y | {obs:7.4f}% | {fit:7.4f}% | {obs-fit:+.4f}%")

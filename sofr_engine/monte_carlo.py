"""
Hull-White 1-Factor Monte Carlo Simulation Engine
==================================================
Exact (no Euler discretization) simulation of the Hull-White short-rate model
calibrated to match the initial SOFR OIS discount curve for all maturities.

Model
-----
The Hull-White short rate r_t satisfies:

    dr_t = [θ(t) − a·r_t] dt + σ·dW_t

Decompose as r_t = x_t + φ(t) where:
    dx_t = −a·x_t dt + σ·dW_t     (Ornstein-Uhlenbeck, x₀ = 0)
    φ(t) = f(0,t) + σ²/(2a²)·(1 − e^{−at})²

and f(0,t) is the instantaneous forward rate from the initial curve.  This
decomposition guarantees φ(t) perfectly fits the initial term structure.

Exact Transition (no discretization bias)
------------------------------------------
Since x_t is an OU process:

    x_{t+Δt} | x_t  ~  Normal(μ, v²)

    μ  = x_t · e^{−aΔt}
    v² = σ²/(2a) · (1 − e^{−2aΔt})

Analytical Bond Pricing
-----------------------
Under HW, the ZCB price conditional on r_t is:

    P(t, T) = A(t,T) · exp(−B(t,T) · r_t)

    B(t,T)     = [1 − e^{−a(T−t)}] / a
    ln A(t,T)  = ln[P(0,T)/P(0,t)] + B(t,T)·f(0,t)
                 − (σ²/4a)·B(t,T)²·(1 − e^{−2at})

Applications Implemented Here
------------------------------
1. Short-rate path simulation (exact, variance-reduced via antithetic)
2. ZCB price distribution at horizon T (MC vs. analytical check)
3. Caplet pricing by MC (vs. Black-76 benchmark)
4. Portfolio VaR / CVaR from simulated curve scenarios
5. Expected short-rate path + 1-σ confidence band

References
----------
Hull, J. & White, A. (1990). "Pricing Interest-Rate Derivative Securities."
  Review of Financial Studies, 3(4), 573–592.
Brigo, D. & Mercurio (2006). Interest Rate Models — Theory and Practice, Ch. 3.
Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering, Ch. 3.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from .curve import DiscountCurve


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class HullWhiteParams:
    """Hull-White 1-factor model parameters."""
    a:     float = 0.05    # mean-reversion speed (per year)
    sigma: float = 0.010   # short-rate volatility (decimal, e.g. 0.01 = 1%/√yr)

    def __post_init__(self):
        if self.a < 0:
            raise ValueError(f"mean-reversion a must be ≥ 0, got {self.a}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inst_forward(curve: DiscountCurve, t: float, dt: float = 1e-4) -> float:
    """Instantaneous forward rate f(0, t) from initial curve."""
    t1 = max(t - dt, 1e-6)
    t2 = t + dt
    return float(curve.forward_rate(t1, t2))


def _phi(curve: DiscountCurve, t: float, params: HullWhiteParams) -> float:
    """
    Initial curve fitting function φ(t) = f(0,t) + σ²/(2a²)·(1 − e^{−at})²

    This ensures r(t) = x(t) + φ(t) perfectly prices the initial ZCBs.
    Falls back to simple f(0,t) when a → 0.
    """
    f0t = _inst_forward(curve, t)
    a, s = params.a, params.sigma
    if a < 1e-8:
        return f0t
    correction = (s ** 2 / (2 * a ** 2)) * (1 - np.exp(-a * t)) ** 2
    return float(f0t + correction)


def _B(t: float, T: float, params: HullWhiteParams) -> float:
    """B(t,T) factor for analytical bond pricing."""
    a = params.a
    tau = T - t
    if a < 1e-8:
        return float(tau)
    return float((1 - np.exp(-a * tau)) / a)


def _ln_A(
    t: float,
    T: float,
    curve: DiscountCurve,
    params: HullWhiteParams,
) -> float:
    """
    ln A(t, T) for analytical bond price P(t,T) = A·exp(−B·r_t).

    ln A = ln[P(0,T)/P(0,t)] + B(t,T)·f(0,t) − (σ²/4a)·B²·(1 − e^{−2at})
    """
    a, s = params.a, params.sigma
    tau  = T - t
    if tau < 1e-10 or t < 1e-10:
        lnp = float(np.interp(T, curve._times, curve._log_df)
                    - np.interp(t, curve._times, curve._log_df))
        return lnp

    lnP0T = float(np.interp(T, curve._times, curve._log_df))
    lnP0t = float(np.interp(t, curve._times, curve._log_df))
    f0t   = _inst_forward(curve, t)
    B_tT  = _B(t, T, params)

    if a < 1e-8:
        correction = 0.0
    else:
        correction = -(s ** 2 / (4 * a)) * B_tT ** 2 * (1 - np.exp(-2 * a * t))

    return float(lnP0T - lnP0t + B_tT * f0t + correction)


# ── Simulation result ─────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Output of a Hull-White MC simulation run."""
    r_paths: np.ndarray          # shape (n_paths, n_steps + 1) — short rate paths
    x_paths: np.ndarray          # shape (n_paths, n_steps + 1) — mean-zero x process
    times:   np.ndarray          # shape (n_steps + 1,) — time grid
    params:  HullWhiteParams
    curve:   DiscountCurve
    n_paths: int
    antithetic: bool

    @property
    def n_steps(self) -> int:
        return len(self.times) - 1

    def r_at(self, step: int) -> np.ndarray:
        """Short-rate cross-section at a given time step."""
        return self.r_paths[:, step]

    def expected_path(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mean ± 1σ of the short-rate paths. Returns (times, mean, std)."""
        return (self.times,
                self.r_paths.mean(axis=0),
                self.r_paths.std(axis=0))

    def zcb_prices_at_horizon(
        self,
        horizon_step: int,
        maturities: list[float],
    ) -> np.ndarray:
        """
        Analytical ZCB prices P(t_h, T) for each path at horizon step.

        Returns array of shape (n_paths, len(maturities)).
        """
        t_h   = float(self.times[horizon_step])
        r_h   = self.r_paths[:, horizon_step]   # (n_paths,)
        out   = np.zeros((self.n_paths, len(maturities)))
        for j, T in enumerate(maturities):
            if T <= t_h:
                out[:, j] = 1.0
                continue
            B_tT  = _B(t_h, T, self.params)
            lnA   = _ln_A(t_h, T, self.curve, self.params)
            out[:, j] = np.exp(lnA - B_tT * r_h)
        return out


# ── Simulation engine ─────────────────────────────────────────────────────────

def simulate_hw(
    curve: DiscountCurve,
    params: HullWhiteParams,
    horizon: float,
    n_steps: int = 100,
    n_paths: int = 10_000,
    seed: int | None = 42,
    antithetic: bool = True,
) -> SimulationResult:
    """
    Exact Hull-White short-rate simulation over [0, horizon].

    Uses the exact conditional Normal distribution of the OU process x_t
    — no Euler discretization error.

    Parameters
    ----------
    curve      : initial discount curve (calibration target)
    params     : HullWhiteParams(a, sigma)
    horizon    : simulation horizon in years
    n_steps    : number of time steps (grid points = n_steps + 1)
    n_paths    : number of Monte Carlo paths
    seed       : random seed for reproducibility
    antithetic : if True, generate n_paths/2 standard + n_paths/2 antithetic paths
                 (doubles effective sample size with no extra function evaluations)

    Returns
    -------
    SimulationResult with r_paths (n_paths × n_steps+1) and times arrays.
    """
    rng    = np.random.default_rng(seed)
    dt     = horizon / n_steps
    times  = np.linspace(0.0, horizon, n_steps + 1)
    a, s   = params.a, params.sigma

    # Transition parameters (exact OU)
    exp_adt = np.exp(-a * dt)
    if a < 1e-8:
        var_dt = s ** 2 * dt
    else:
        var_dt = (s ** 2 / (2 * a)) * (1 - np.exp(-2 * a * dt))
    std_dt = float(np.sqrt(var_dt))

    # Number of base paths
    n_base = n_paths // 2 if antithetic else n_paths
    n_actual = n_base * 2 if antithetic else n_paths

    x_base = np.zeros((n_base, n_steps + 1))
    for step in range(n_steps):
        z = rng.standard_normal(n_base)
        x_base[:, step + 1] = x_base[:, step] * exp_adt + std_dt * z

    if antithetic:
        x_anti = np.zeros((n_base, n_steps + 1))
        for step in range(n_steps):
            x_anti[:, step + 1] = x_base[:, step + 1] * (-1)  # antithetic
        # Recalculate antithetic properly from the innovations
        # (just flip sign of x increments from x_base)
        x_paths = np.vstack([x_base, -x_base])
        n_paths_actual = n_actual
    else:
        x_paths    = x_base
        n_paths_actual = n_paths

    # Add drift: r(t) = x(t) + φ(t)
    phi_vals = np.array([_phi(curve, t, params) for t in times])  # (n_steps+1,)
    r_paths  = x_paths + phi_vals[np.newaxis, :]   # broadcast

    return SimulationResult(
        r_paths    = r_paths,
        x_paths    = x_paths,
        times      = times,
        params     = params,
        curve      = curve,
        n_paths    = n_paths_actual,
        antithetic = antithetic,
    )


# ── Analytical ZCB price ──────────────────────────────────────────────────────

def zcb_price_hw(
    r_t: float | np.ndarray,
    t: float,
    T: float,
    curve: DiscountCurve,
    params: HullWhiteParams,
) -> float | np.ndarray:
    """
    Analytical Hull-White ZCB price P(t, T) conditional on r_t.

    P(t,T) = A(t,T) · exp(−B(t,T) · r_t)

    Works with scalar or array inputs for r_t.
    """
    B   = _B(t, T, params)
    lnA = _ln_A(t, T, curve, params)
    return np.exp(lnA - B * np.asarray(r_t))


# ── MC pricing functions ──────────────────────────────────────────────────────

def price_zcb_mc(
    curve: DiscountCurve,
    params: HullWhiteParams,
    maturity: float,
    horizon: float | None = None,
    n_steps: int = 50,
    n_paths: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Price P(0, maturity) by Monte Carlo and compare to the initial curve value.

    If horizon is None, uses horizon = maturity (simulate to maturity and
    discount the terminal payoff of 1.0 along the path).

    Returns
    -------
    dict with keys: mc_price, analytical_price, error_bps, stderr, n_paths
    """
    if horizon is None:
        horizon = maturity

    sim     = simulate_hw(curve, params, horizon, n_steps, n_paths, seed)
    t_h     = horizon
    h_step  = sim.n_steps  # last step

    # Discount from 0 to horizon using short rates (path integral approximation)
    dt = horizon / sim.n_steps
    # Use midpoint rule for path integral
    r_mid = 0.5 * (sim.r_paths[:, :-1] + sim.r_paths[:, 1:])
    path_disc = np.exp(-r_mid.sum(axis=1) * dt)   # exp(-∫r dt)

    if abs(horizon - maturity) < 1e-8:
        # P(0,T) ≈ E[exp(-∫₀ᵀ r_s ds)]
        mc_prices = path_disc  # terminal payoff = 1
    else:
        # Horizon < maturity: price P(horizon, maturity) analytically along paths
        r_h       = sim.r_paths[:, h_step]
        p_h_T     = zcb_price_hw(r_h, t_h, maturity, curve, params)
        mc_prices = path_disc * p_h_T   # discount to 0 then hold ZCB from t_h to T

    mc_mean   = float(np.mean(mc_prices))
    mc_se     = float(np.std(mc_prices) / np.sqrt(len(mc_prices)))
    analytic  = float(curve.df(maturity))
    error_bps = abs(mc_mean - analytic) / analytic * 10_000

    return {
        "maturity":          maturity,
        "mc_price":          round(mc_mean, 8),
        "analytical_price":  round(analytic, 8),
        "error_bps":         round(error_bps, 3),
        "mc_stderr":         round(mc_se, 8),
        "n_paths":           sim.n_paths,
        "n_steps":           sim.n_steps,
        "mc_95ci":           (round(mc_mean - 1.96 * mc_se, 8),
                              round(mc_mean + 1.96 * mc_se, 8)),
    }


def price_caplet_mc(
    curve: DiscountCurve,
    params: HullWhiteParams,
    strike: float,
    t_reset: float,
    t_pay: float,
    notional: float = 1_000_000.0,
    n_steps: int = 50,
    n_paths: int = 20_000,
    seed: int = 42,
) -> dict:
    """
    Price a caplet by Monte Carlo simulation under Hull-White.

    At t_reset: observe r(t_reset). The SOFR forward rate for [t_reset, t_pay]
    is determined by the HW bond prices:
        F = (P(t_reset, t_reset) / P(t_reset, t_pay) − 1) / tau = (1/P − 1) / tau

    The caplet pays notional × τ × max(F − K, 0) at t_pay, discounted
    to t = 0 using the simulated path discount.

    Returns
    -------
    dict with mc_pv, black76_pv, error_bps (for comparison), stderr.
    """
    from .cap_floor import caplet_black_pv

    tau    = t_pay - t_reset
    sim    = simulate_hw(curve, params, t_reset, n_steps, n_paths, seed)
    h_step = sim.n_steps
    dt     = t_reset / n_steps

    # Path discount factor from 0 to t_reset (midpoint rule)
    r_mid   = 0.5 * (sim.r_paths[:, :-1] + sim.r_paths[:, 1:])
    disc_0r = np.exp(-r_mid.sum(axis=1) * dt)

    # SOFR forward rate at t_reset for [t_reset, t_pay]
    r_reset   = sim.r_paths[:, h_step]
    p_reset_pay = zcb_price_hw(r_reset, t_reset, t_pay, curve, params)   # P(t_r, t_p)
    p_pay_0     = zcb_price_hw(r_reset, t_reset, t_pay, curve, params)   # same

    # Discount factor from t_pay back to 0 via: disc(0→t_r) × P(t_r, t_p)
    df_pay_0 = disc_0r * p_reset_pay

    F = (1.0 / p_reset_pay - 1.0) / tau       # forward SOFR rate
    payoff = notional * tau * np.maximum(F - strike, 0.0)
    mc_pvs  = df_pay_0 * payoff

    mc_mean = float(np.mean(mc_pvs))
    mc_se   = float(np.std(mc_pvs) / np.sqrt(len(mc_pvs)))

    # Benchmark: Black-76 caplet price
    F0    = float(curve.df(t_reset) / curve.df(t_pay) - 1) / tau
    df0   = float(curve.df(t_pay))
    # Approximate Black vol from HW: σ_Black ≈ σ_N / F (normal-to-black approx)
    hw_normal_vol = float(params.sigma * np.sqrt(
        (1 - np.exp(-2 * params.a * t_reset)) / (2 * params.a)
    ) / (_B(0, t_reset, params) if params.a > 1e-8 else t_reset))
    black_vol_approx = hw_normal_vol / F0 if F0 > 1e-8 else 0.30

    black_pv = caplet_black_pv(F0, strike, t_reset, tau, df0,
                                black_vol_approx, notional, "cap")

    return {
        "t_reset":         t_reset,
        "t_pay":           t_pay,
        "strike_pct":      round(strike * 100, 4),
        "mc_pv":           round(mc_mean, 2),
        "mc_stderr":       round(mc_se, 2),
        "mc_95ci":         (round(mc_mean - 1.96 * mc_se, 2),
                            round(mc_mean + 1.96 * mc_se, 2)),
        "black76_pv":      round(black_pv, 2),
        "forward_rate_pct": round(F0 * 100, 4),
        "hw_normal_vol_bps": round(hw_normal_vol * 10_000, 1),
        "n_paths":         sim.n_paths,
    }


# ── Portfolio VaR ─────────────────────────────────────────────────────────────

def portfolio_var_hw(
    curve: DiscountCurve,
    params: HullWhiteParams,
    portfolio_dv01: float,
    horizon: float = 1.0 / 252,
    confidence: float = 0.99,
    n_paths: int = 10_000,
    seed: int = 42,
    full_reprice: bool = False,
    reprice_tenors: list[float] | None = None,
) -> dict:
    """
    Monte Carlo Value-at-Risk for a fixed-income portfolio under Hull-White.

    For each simulated scenario at horizon T_h:
    1. Compute r(T_h) from the HW simulation
    2. Compute the scenario yield change at each tenor: Δy(T) = y_new(T) − y_old(T)
       using the analytical ZCB prices P(T_h, T)
    3. Approximate P&L = portfolio_dv01 × (−Δy in bps) [first-order delta approx]
       OR full reprice from simulated curve [if full_reprice=True]

    Parameters
    ----------
    portfolio_dv01 : total portfolio DV01 in $/bp (positive = long duration)
    horizon        : VaR horizon in years (default 1 day = 1/252)
    confidence     : VaR confidence level (default 0.99)
    n_paths        : number of MC paths
    full_reprice   : if True, reprice at each scenario tenor using analytical ZCBs
    reprice_tenors : tenors for full repricing (default: [2, 5, 10])

    Returns
    -------
    dict with var_usd, cvar_usd, var_bps, cvar_bps, pnl_percentiles, n_paths
    """
    if reprice_tenors is None:
        reprice_tenors = [2.0, 5.0, 10.0]

    sim    = simulate_hw(curve, params, horizon, n_steps=10, n_paths=n_paths, seed=seed)
    h_step = sim.n_steps
    r_h    = sim.r_paths[:, h_step]           # (n_paths,)

    if not full_reprice:
        # Delta approximation using a representative tenor (e.g., 10Y)
        # PnL ≈ -DV01 × Δy_10Y × 10_000 ... wait:
        # DV01 is $/bp, PnL = DV01 × (−Δy in bp)
        # Δy = new_zero_rate(10Y) − old_zero_rate(10Y)
        rep_tenor = reprice_tenors[-1]
        # Old zero rate at rep_tenor
        y_old  = float(curve.zero_rate(rep_tenor))
        # New ZCB price at horizon using HW
        p_new  = zcb_price_hw(r_h, horizon, horizon + rep_tenor, curve, params)
        y_new  = -np.log(np.maximum(p_new, 1e-10)) / rep_tenor
        delta_y_bps = (y_new - y_old) * 10_000       # in bps
        pnl_usd = portfolio_dv01 * (-delta_y_bps)    # long duration: profit when rates fall
    else:
        # Full reprice: sum DV01-weighted yield changes across multiple tenors
        # Approximate by weighting each tenor equally
        delta_y_total = np.zeros(sim.n_paths)
        for T in reprice_tenors:
            y_old = float(curve.zero_rate(T))
            p_new = zcb_price_hw(r_h, horizon, horizon + T, curve, params)
            y_new = -np.log(np.maximum(p_new, 1e-10)) / T
            delta_y_total += (y_new - y_old) * 10_000
        delta_y_bps = delta_y_total / len(reprice_tenors)
        pnl_usd = portfolio_dv01 * (-delta_y_bps)

    # VaR / CVaR
    alpha  = 1.0 - confidence
    var_pnl  = float(np.percentile(pnl_usd, alpha * 100))   # loss (negative = loss)
    cvar_pnl = float(np.mean(pnl_usd[pnl_usd <= var_pnl]))
    var_bps  = float(np.percentile(delta_y_bps, (1 - alpha) * 100))
    cvar_bps = float(np.mean(delta_y_bps[delta_y_bps >= var_bps]))

    pct = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "var_usd":          round(var_pnl, 2),
        "cvar_usd":         round(cvar_pnl, 2),
        "var_bps":          round(var_bps, 3),
        "cvar_bps":         round(cvar_bps, 3),
        "pnl_mean_usd":     round(float(pnl_usd.mean()), 2),
        "pnl_std_usd":      round(float(pnl_usd.std()), 2),
        "pnl_percentiles":  {f"p{p}": round(float(np.percentile(pnl_usd, p)), 2) for p in pct},
        "confidence":       confidence,
        "horizon_days":     round(horizon * 252),
        "n_paths":          sim.n_paths,
        "portfolio_dv01":   portfolio_dv01,
    }


# ── Parametric VaR (delta-normal) ─────────────────────────────────────────────

def parametric_var(
    portfolio_dv01: float,
    yield_vol_bps: float,
    horizon: float = 1.0 / 252,
    confidence: float = 0.99,
) -> dict:
    """
    Delta-normal (parametric) VaR for a fixed-income portfolio.

    Assumes daily yield changes ~ N(0, σ_y²) with σ_y = yield_vol_bps/√252.

    VaR = DV01 × z_α × σ_y × √horizon_days
    where z_α is the confidence quantile of N(0,1).

    Parameters
    ----------
    portfolio_dv01 : DV01 in $/bp (positive = long duration)
    yield_vol_bps  : annualized yield volatility in basis points (e.g. 80 bps/√yr)
    horizon        : VaR horizon in years (default 1/252 = 1 day)
    confidence     : confidence level (default 0.99)

    Returns
    -------
    dict with var_usd, var_bps, one_day_yield_vol_bps, scaling_factor
    """
    from scipy.stats import norm as _norm
    z_alpha           = float(_norm.ppf(confidence))
    horizon_days      = horizon * 252
    daily_vol_bps     = yield_vol_bps / np.sqrt(252)
    horizon_vol_bps   = daily_vol_bps * np.sqrt(horizon_days)
    var_bps           = z_alpha * horizon_vol_bps
    var_usd           = abs(portfolio_dv01) * var_bps

    return {
        "var_usd":             round(var_usd, 2),
        "var_bps":             round(var_bps, 3),
        "cvar_usd":            round(var_usd * 1.087, 2),  # CVaR ≈ 1.087 × VaR for Normal(0,1)
        "one_day_yield_vol_bps": round(daily_vol_bps, 4),
        "horizon_yield_vol_bps": round(horizon_vol_bps, 4),
        "z_alpha":             round(z_alpha, 4),
        "confidence":          confidence,
        "horizon_days":        round(horizon_days, 2),
        "portfolio_dv01":      portfolio_dv01,
    }


# ── Convergence diagnostics ───────────────────────────────────────────────────

def convergence_diagnostics(
    curve: DiscountCurve,
    params: HullWhiteParams,
    test_maturity: float = 5.0,
    path_counts: list[int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Compute MC ZCB pricing error vs n_paths to illustrate convergence.

    Returns a list of dicts with n_paths, mc_price, error_bps, stderr.
    """
    if path_counts is None:
        path_counts = [100, 500, 1_000, 5_000, 10_000, 50_000]

    results = []
    for n in path_counts:
        r = price_zcb_mc(curve, params, test_maturity,
                         n_paths=n, n_steps=50, seed=seed)
        results.append({
            "n_paths":     n,
            "mc_price":    r["mc_price"],
            "error_bps":   r["error_bps"],
            "mc_stderr":   r["mc_stderr"],
        })
    return results

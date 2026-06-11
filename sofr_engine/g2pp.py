"""
G2++ Two-Factor Gaussian Interest Rate Model
=============================================
The G2++ model (Brigo & Mercurio, 2006) extends Hull-White 1F with a
second mean-reverting Gaussian factor:

    r(t) = x(t) + y(t) + φ(t)

    dx(t) = −a·x(t)·dt + σ·dW₁(t)
    dy(t) = −b·y(t)·dt + η·dW₂(t)
    dW₁·dW₂ = ρ·dt

where φ(t) is chosen to exactly fit the initial term structure.

Features
--------
- Analytical ZCB price P(t,T) and instantaneous forward rate f(t,T)
- Analytical European swaption pricing via Brigo-Mercurio numerical integral
- Exact simulation: (x,y) are jointly Gaussian with known conditional mean & cov
- Calibration to ATM swaption vol matrix (scipy.optimize.minimize)
- Portfolio VaR via scenario simulation

References
----------
Brigo, D. & Mercurio, F. (2006) "Interest Rate Models — Theory and Practice",
2nd ed., Springer. Chapters 4 (G2++) and 7 (swaption pricing).

Hull, J. & White, A. (1994) "Numerical Procedures for Implementing Term
Structure Models II: Two-Factor Models", JFQA 29(4), 613-656.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm

from .curve import DiscountCurve

Φ = norm.cdf
φ = norm.pdf

__all__ = [
    "G2ppParams",
    "G2ppSimResult",
    "g2pp_zcb",
    "g2pp_inst_forward",
    "simulate_g2pp",
    "g2pp_swaption",
    "g2pp_portfolio_var",
    "calibrate_g2pp",
]


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class G2ppParams:
    """
    G2++ model parameters.

    Attributes
    ----------
    a   : mean-reversion speed of x-factor (a > 0)
    b   : mean-reversion speed of y-factor (b > 0, b ≠ a for numerical stability)
    sigma : vol of x-factor (σ > 0)
    eta   : vol of y-factor (η > 0)
    rho   : correlation between W₁ and W₂ (-1 < ρ < 1)
    """
    a:     float
    b:     float
    sigma: float
    eta:   float
    rho:   float

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError("a must be positive")
        if self.b <= 0:
            raise ValueError("b must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must be in (-1, 1)")


@dataclass
class G2ppSimResult:
    """Results from G2++ Monte Carlo simulation."""
    t_grid:   np.ndarray          # shape (n_steps+1,)
    x_paths:  np.ndarray          # shape (n_paths, n_steps+1)
    y_paths:  np.ndarray          # shape (n_paths, n_steps+1)
    r_paths:  np.ndarray          # shape (n_paths, n_steps+1)  r = x + y + phi
    curve:    DiscountCurve
    params:   G2ppParams
    n_paths:  int
    horizon:  float

    def zcb_prices_at(
        self, step: int, maturities: Sequence[float]
    ) -> np.ndarray:
        """
        Analytical G2++ ZCB prices at simulation step `step` for given maturities.

        Returns array of shape (n_paths, len(maturities)).
        """
        t = float(self.t_grid[step])
        x = self.x_paths[:, step]    # shape (n_paths,)
        y = self.y_paths[:, step]
        out = np.zeros((self.n_paths, len(maturities)))
        for j, T in enumerate(maturities):
            out[:, j] = np.array([
                g2pp_zcb(self.curve, self.params, t, T, float(x[i]), float(y[i]))
                for i in range(self.n_paths)
            ])
        return out


# ── Analytical building blocks ────────────────────────────────────────────────

def _B(kappa: float, t: float, T: float) -> float:
    """B_κ(t,T) = (1 - e^{-κ(T-t)}) / κ."""
    tau = T - t
    if tau < 1e-12:
        return 0.0
    return (1.0 - math.exp(-kappa * tau)) / kappa


def _V_single(kappa: float, vol: float, tau: float) -> float:
    """
    V_κ(τ) = (vol/κ)² × [τ - (2/κ)*(1-e^{-κτ}) + (1/(2κ))*(1-e^{-2κτ})]

    This is Var[∫_0^τ x(s) ds] / 1 for an OU process starting at 0.
    Equivalently, the "variance integral" of B_κ over [0,τ].
    """
    return (vol / kappa)**2 * (
        tau
        - (2.0 / kappa) * (1.0 - math.exp(-kappa * tau))
        + (1.0 / (2.0 * kappa)) * (1.0 - math.exp(-2.0 * kappa * tau))
    )


def _ln_A_g2pp(
    curve:  DiscountCurve,
    params: G2ppParams,
    t:      float,
    T:      float,
) -> float:
    """
    ln A(t,T) in the G2++ ZCB formula P(t,T;x,y) = A(t,T)·exp(−Bₐ·x − Bᵦ·y).

    Derived from first principles using the affine conditional MGF:

        ln A(t,T) = ln[P(0,T)/P(0,t)]
                  − σ²/(2a²)·Bₐ·(1−e^{−at})
                  − σ²/(4a³)·(1−e^{−2a(T−t)})·(1−e^{−2at})
                  − η²/(2b²)·Bᵦ·(1−e^{−bt})
                  − η²/(4b³)·(1−e^{−2b(T−t)})·(1−e^{−2bt})
                  − ρση/(ab)·[Bₐ·(1−e^{−at}) + Bᵦ·(1−e^{−bt}) − B_{a+b}·(1−e^{−(a+b)t})]

    Satisfies A(0,T)=P(0,T) and A(t,T)→1 as T→t.
    """
    a, b, s, e, rho = params.a, params.b, params.sigma, params.eta, params.rho

    lnP0T = float(np.log(curve.df(T)))
    lnP0t = float(np.log(curve.df(t))) if t > 1e-10 else 0.0

    if t < 1e-10:
        return lnP0T   # A(0,T) = P(0,T)

    tau  = T - t
    Ba   = _B(a, t, T)
    Bb   = _B(b, t, T)
    Bab  = _B(a + b, t, T)

    ea_t   = math.exp(-a * t)
    eb_t   = math.exp(-b * t)
    eab_t  = math.exp(-(a + b) * t)
    e2a_t  = math.exp(-2.0 * a * t)
    e2b_t  = math.exp(-2.0 * b * t)
    e2a_T  = math.exp(-2.0 * a * tau)
    e2b_T  = math.exp(-2.0 * b * tau)

    correction = (
        - (s**2 / (2.0 * a**2)) * Ba * (1.0 - ea_t)
        - (s**2 / (4.0 * a**3)) * (1.0 - e2a_T) * (1.0 - e2a_t)
        - (e**2 / (2.0 * b**2)) * Bb * (1.0 - eb_t)
        - (e**2 / (4.0 * b**3)) * (1.0 - e2b_T) * (1.0 - e2b_t)
        - (rho * s * e / (a * b)) * (
              Ba * (1.0 - ea_t)
              + Bb * (1.0 - eb_t)
              - Bab * (1.0 - eab_t)
          )
    )
    return lnP0T - lnP0t + correction


def g2pp_inst_forward(curve: DiscountCurve, t: float, dt: float = 1e-5) -> float:
    """Instantaneous forward rate f(0,t) = -d/dt ln P(0,t)."""
    if t < dt:
        t = dt
    lnP_m = float(np.log(curve.df(max(t - dt, 1e-8))))
    lnP_p = float(np.log(curve.df(t + dt)))
    return -(lnP_p - lnP_m) / (2.0 * dt)


def g2pp_zcb(
    curve:  DiscountCurve,
    params: G2ppParams,
    t:      float,
    T:      float,
    x_t:    float,
    y_t:    float,
) -> float:
    """
    Analytical G2++ zero-coupon bond price.

        P(t,T; x,y) = A(t,T) × exp(−Bₐ(t,T)·x − Bᵦ(t,T)·y)

    Parameters
    ----------
    t    : current time (years from today)
    T    : maturity (years from today), T > t
    x_t  : current x-factor state
    y_t  : current y-factor state
    """
    if T <= t:
        return 1.0 if abs(T - t) < 1e-10 else 0.0
    Ba   = _B(params.a, t, T)
    Bb   = _B(params.b, t, T)
    lnA  = _ln_A_g2pp(curve, params, t, T)
    return math.exp(lnA - Ba * x_t - Bb * y_t)


# ── Phi: the deterministic shift ──────────────────────────────────────────────

def _g2pp_phi(curve: DiscountCurve, params: G2ppParams, t: float) -> float:
    """
    Deterministic shift φ(t) = f(0,t) + σ²/(2a²)·(1−e^{-at})² + η²/(2b²)·(1−e^{-bt})²
                                + ρση/(ab)·(1−e^{-at})·(1−e^{-bt})

    This ensures E[r(t)] = f(0,t) when x=y=0, so the model fits initial term structure.
    """
    a, b, s, e, rho = params.a, params.b, params.sigma, params.eta, params.rho
    if t < 1e-10:
        return g2pp_inst_forward(curve, t)
    f0t   = g2pp_inst_forward(curve, t)
    ea    = math.exp(-a * t)
    eb    = math.exp(-b * t)
    phi = (f0t
           + (s**2 / (2.0 * a**2)) * (1.0 - ea)**2
           + (e**2 / (2.0 * b**2)) * (1.0 - eb)**2
           + rho * s * e / (a * b) * (1.0 - ea) * (1.0 - eb))
    return phi


# ── Exact simulation ──────────────────────────────────────────────────────────

def simulate_g2pp(
    curve:      DiscountCurve,
    params:     G2ppParams,
    horizon:    float,
    n_steps:    int    = 100,
    n_paths:    int    = 10_000,
    seed:       int    = 42,
    antithetic: bool   = True,
) -> G2ppSimResult:
    """
    Exact simulation of the G2++ model using the conditional Gaussian law.

    The two factors (x, y) are jointly Gaussian with conditional means and
    covariance:

        E[x(T)|x(t)] = x(t)·e^{-a·Δt}
        E[y(T)|y(t)] = y(t)·e^{-b·Δt}

        Cov(x(T), x(T)|t) = σ²·(1 − e^{-2a·Δt}) / (2a)
        Cov(y(T), y(T)|t) = η²·(1 − e^{-2b·Δt}) / (2b)
        Cov(x(T), y(T)|t) = ρ·σ·η·(1 − e^{-(a+b)·Δt}) / (a+b)

    Antithetic variates: each half of paths uses (-Z₁, -Z₂) for variance
    reduction, respecting the joint structure.

    Parameters
    ----------
    horizon : simulation end time T (years from today)
    n_steps : number of time steps
    n_paths : total number of paths (must be even if antithetic=True)
    seed    : RNG seed for reproducibility

    Returns
    -------
    G2ppSimResult with t_grid, x_paths, y_paths, r_paths
    """
    if antithetic and n_paths % 2 != 0:
        n_paths += 1

    rng    = np.random.default_rng(seed)
    dt     = horizon / n_steps
    t_grid = np.linspace(0.0, horizon, n_steps + 1)

    n_base = n_paths // 2 if antithetic else n_paths
    a, b   = params.a, params.b
    s, e   = params.sigma, params.eta
    rho    = params.rho

    # Conditional standard deviations
    std_x_dt = math.sqrt(s**2 * (1.0 - math.exp(-2.0 * a * dt)) / (2.0 * a))
    std_y_dt = math.sqrt(e**2 * (1.0 - math.exp(-2.0 * b * dt)) / (2.0 * b))
    # Conditional correlation between increments
    cov_xy   = rho * s * e * (1.0 - math.exp(-(a + b) * dt)) / (a + b)
    if std_x_dt * std_y_dt > 1e-14:
        rho_dt = cov_xy / (std_x_dt * std_y_dt)
    else:
        rho_dt = 0.0
    rho_dt = float(np.clip(rho_dt, -0.9999, 0.9999))
    sqrt1_rho2 = math.sqrt(1.0 - rho_dt**2)

    # Pre-compute decay factors
    decay_x = math.exp(-a * dt)
    decay_y = math.exp(-b * dt)

    # Allocate path storage
    x_base = np.zeros((n_base, n_steps + 1))
    y_base = np.zeros((n_base, n_steps + 1))

    for step in range(n_steps):
        Z1 = rng.standard_normal((n_base,))
        Z2 = rng.standard_normal((n_base,))
        # Correlate: W1 = Z1, W2 = rho*Z1 + sqrt(1-rho²)*Z2
        W1 = Z1
        W2 = rho_dt * Z1 + sqrt1_rho2 * Z2

        x_base[:, step + 1] = decay_x * x_base[:, step] + std_x_dt * W1
        y_base[:, step + 1] = decay_y * y_base[:, step] + std_y_dt * W2

    if antithetic:
        x_paths = np.vstack([x_base, -x_base])
        y_paths = np.vstack([y_base, -y_base])
    else:
        x_paths = x_base
        y_paths = y_base

    # Compute r(t) = x(t) + y(t) + φ(t)
    phi_grid = np.array([_g2pp_phi(curve, params, float(t)) for t in t_grid])
    r_paths  = x_paths + y_paths + phi_grid[np.newaxis, :]

    return G2ppSimResult(
        t_grid  = t_grid,
        x_paths = x_paths,
        y_paths = y_paths,
        r_paths = r_paths,
        curve   = curve,
        params  = params,
        n_paths = n_paths,
        horizon = horizon,
    )


# ── European swaption pricing (Brigo-Mercurio) ───────────────────────────────

def g2pp_swaption(
    curve:     DiscountCurve,
    params:    G2ppParams,
    expiry:    float,
    swap_tenor: float,
    strike:    float | None = None,
    notional:  float  = 1_000_000.0,
    pay_receive: str  = "payer",
    freq:      int    = 2,
    n_quad:    int    = 96,
) -> dict:
    """
    G2++ European swaption price via Brigo-Mercurio numerical integration
    (Proposition 4.5, p. 155).

    The swaption payer has payoff max(V_swap(T), 0) at T.

    Under G2++ the swaption price is:

        Payer = Σᵢ cᵢ × E[P(0,Tᵢ) × 1_{V_swap(T) > 0}]

    where each factor is a call on a ZCB.  The joint distribution of
    (x(T), y(T)) is bivariate Gaussian N(0, Σ_T) where:

        Var(x(T)) = σ²(1-e^{-2aT})/(2a)
        Var(y(T)) = η²(1-e^{-2bT})/(2b)
        Cov(x,y)  = ρση(1-e^{-(a+b)T})/(a+b)

    The critical region {V_swap(T) > 0} is a curve x = h(y), so:

        Payer = ∫_{-∞}^{∞} [Σᵢ cᵢ × P_i(y) × Φ(z_i(y))] × φ_y(y) dy

    where P_i(y) = A(T,Tᵢ)·exp(-B_b(T,Tᵢ)·ȳ) is the ZCB conditional on y,
    and z_i is the critical x-value shifted.

    Parameters
    ----------
    expiry      : option expiry T (years)
    swap_tenor  : underlying swap tenor (years)
    strike      : fixed rate; None = ATM
    notional    : face value
    pay_receive : "payer" or "receiver"
    freq        : swap payment frequency
    n_quad      : number of Gauss-Hermite quadrature nodes

    Returns
    -------
    dict with pv, forward_swap_rate_pct, strike_pct, annuity, n_quad
    """
    T  = expiry
    dt = 1.0 / freq
    n  = int(round(swap_tenor * freq))
    Tis = [T + (k + 1) * dt for k in range(n)]  # payment dates

    # Forward swap rate (at t=0 with x=y=0)
    ann = sum(dt * float(curve.df(Ti)) for Ti in Tis)
    S_fwd = (float(curve.df(T)) - float(curve.df(Tis[-1]))) / ann if ann > 1e-14 else np.nan

    if strike is None:
        K = S_fwd
    else:
        K = strike

    # Coupon weights: ci × P(0,Ti); final period includes principal
    c = [K * dt] * n
    c[-1] += 1.0  # add principal repayment at last payment

    # G2++ joint variance of (x(T), y(T))
    a, b, s, e, rho = params.a, params.b, params.sigma, params.eta, params.rho
    var_x  = s**2 * (1.0 - math.exp(-2.0 * a * T)) / (2.0 * a)
    var_y  = e**2 * (1.0 - math.exp(-2.0 * b * T)) / (2.0 * b)
    cov_xy = rho * s * e * (1.0 - math.exp(-(a + b) * T)) / (a + b)

    std_x = math.sqrt(var_x)
    std_y = math.sqrt(var_y)
    if std_x < 1e-14 or std_y < 1e-14:
        return {"pv": 0.0, "forward_swap_rate_pct": S_fwd * 100, "strike_pct": K * 100,
                "annuity": ann, "n_quad": n_quad}

    rho_xy = float(np.clip(cov_xy / (std_x * std_y), -0.9999, 0.9999))
    sqrt1m = math.sqrt(1.0 - rho_xy**2)

    # Conditional mean of x given y: E[x|y] = rho_xy * (std_x/std_y) * y
    beta_xy = rho_xy * std_x / std_y  # regression coefficient x on y

    # Precompute lnA and B for each payment date
    lnA_list = [_ln_A_g2pp(curve, params, T, Ti) for Ti in Tis]
    Ba_list  = [_B(a, T, Ti) for Ti in Tis]
    Bb_list  = [_B(b, T, Ti) for Ti in Tis]

    def _find_x_critical(y_val: float) -> float:
        """Find x* such that Σᵢ cᵢ P(T,Tᵢ;x*,y) = 1 (bond at par = swap ITM)."""
        def f(x: float) -> float:
            bond_pv = sum(
                c[i] * math.exp(lnA_list[i] - Ba_list[i] * x - Bb_list[i] * y_val)
                for i in range(n)
            )
            return bond_pv - 1.0
        try:
            x_lo = -10.0 * std_x
            x_hi = +10.0 * std_x
            # Check sign at boundaries
            if f(x_lo) * f(x_hi) > 0:
                return float("nan")
            return optimize.brentq(f, x_lo, x_hi, xtol=1e-10)
        except Exception:
            return float("nan")

    # Gauss-Hermite quadrature over y ∈ (-∞, +∞)
    # y = std_y × √2 × t, so dy = std_y × √2 dt
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)

    pv = 0.0
    for node, w in zip(nodes, weights):
        y_val = std_y * math.sqrt(2.0) * node

        # Conditional mean of x given y
        mu_x_cond = beta_xy * y_val
        std_x_cond = std_x * sqrt1m

        # Find critical x* for this y slice
        x_star = _find_x_critical(y_val)
        if math.isnan(x_star):
            continue

        # Payer: exercise when V_swap > 0 ↔ x > x_star (for payer)
        # Actually exercise when swap PV > 0; for payer this means rates rose
        # Direction depends on sign of dV/dx — check at ATM
        z = (x_star - mu_x_cond) / std_x_cond

        # Contribution from each coupon: Σᵢ cᵢ P_i(y) × Φ(±(μ_i - x_star)/σ)
        row_pv = 0.0
        for i in range(n):
            Ai_exp = math.exp(lnA_list[i] - Bb_list[i] * y_val)
            mi     = mu_x_cond - Ba_list[i] * std_x_cond**2
            d_i    = (x_star - mi) / std_x_cond
            if pay_receive == "payer":
                row_pv += c[i] * Ai_exp * math.exp(-Ba_list[i] * mu_x_cond
                                                    + 0.5 * Ba_list[i]**2 * std_x_cond**2) * Φ(-d_i)
            else:
                row_pv += c[i] * Ai_exp * math.exp(-Ba_list[i] * mu_x_cond
                                                    + 0.5 * Ba_list[i]**2 * std_x_cond**2) * Φ(d_i)

        # Subtract the DF(T) contribution
        df0T = float(curve.df(T)) * math.exp(
            -_B(b, 0, T) * y_val + 0.5 * (_B(b, 0, T) * std_y)**2
        )

        if pay_receive == "payer":
            slice_pv = row_pv - df0T * Φ(-z)
        else:
            slice_pv = df0T * Φ(z) - row_pv

        # Weight by marginal density of y (Gaussian, absorbed into G-H)
        pv += w * slice_pv / math.sqrt(math.pi)

    pv *= notional

    return {
        "pv":                     max(pv, 0.0),
        "forward_swap_rate_pct":  S_fwd * 100,
        "strike_pct":             K * 100,
        "annuity":                ann,
        "n_quad":                 n_quad,
        "pay_receive":            pay_receive,
    }


# ── MC-based swaption pricing (alternative) ──────────────────────────────────

def g2pp_swaption_mc(
    curve:       DiscountCurve,
    params:      G2ppParams,
    expiry:      float,
    swap_tenor:  float,
    strike:      float | None = None,
    notional:    float = 1_000_000.0,
    pay_receive: str   = "payer",
    freq:        int   = 2,
    n_paths:     int   = 20_000,
    seed:        int   = 42,
) -> dict:
    """
    G2++ European swaption via Monte Carlo (for comparison with analytical).

    Simulates (x(T), y(T)) at expiry, values the swap at each path, then
    averages the positive payoffs discounted back.
    """
    T   = expiry
    dt  = 1.0 / freq
    n   = int(round(swap_tenor * freq))
    Tis = [T + (k + 1) * dt for k in range(n)]

    ann   = sum(dt * float(curve.df(Ti)) for Ti in Tis)
    S_fwd = (float(curve.df(T)) - float(curve.df(Tis[-1]))) / ann if ann > 1e-14 else np.nan
    K     = S_fwd if strike is None else strike

    # Simulate to expiry in one shot (exact Gaussian draw)
    a, b = params.a, params.b
    s, e, rho = params.sigma, params.eta, params.rho

    rng   = np.random.default_rng(seed)
    n_h   = n_paths // 2
    var_x = s**2 * (1.0 - math.exp(-2.0 * a * T)) / (2.0 * a)
    var_y = e**2 * (1.0 - math.exp(-2.0 * b * T)) / (2.0 * b)
    cov   = rho * s * e * (1.0 - math.exp(-(a + b) * T)) / (a + b)

    cov_mat = np.array([[var_x, cov], [cov, var_y]])
    Z       = rng.multivariate_normal([0.0, 0.0], cov_mat, size=n_h)
    x_all   = np.concatenate([Z[:, 0], -Z[:, 0]])
    y_all   = np.concatenate([Z[:, 1], -Z[:, 1]])

    lnA_list = [_ln_A_g2pp(curve, params, T, Ti) for Ti in Tis]
    Ba_list  = [_B(a, T, Ti) for Ti in Tis]
    Bb_list  = [_B(b, T, Ti) for Ti in Tis]

    payoffs = np.zeros(n_paths)
    for i_path in range(n_paths):
        x_t = x_all[i_path]
        y_t = y_all[i_path]
        ann_t = sum(
            dt * math.exp(lnA_list[i] - Ba_list[i] * x_t - Bb_list[i] * y_t)
            for i in range(n)
        )
        dfT_TN = math.exp(lnA_list[-1] - Ba_list[-1] * x_t - Bb_list[-1] * y_t)
        dfT_T  = 1.0   # discount factor P(T,T) = 1 (start of swap at T)
        S_t    = (dfT_T - dfT_TN) / ann_t if ann_t > 1e-14 else 0.0
        if pay_receive == "payer":
            payoffs[i_path] = max(S_t - K, 0.0) * ann_t
        else:
            payoffs[i_path] = max(K - S_t, 0.0) * ann_t

    df0T  = float(curve.df(T))
    pv    = notional * df0T * float(np.mean(payoffs))
    se    = notional * df0T * float(np.std(payoffs) / math.sqrt(n_paths))

    return {
        "pv":                    pv,
        "mc_stderr":             se,
        "forward_swap_rate_pct": S_fwd * 100,
        "strike_pct":            K * 100,
        "annuity":               ann,
        "n_paths":               n_paths,
        "pay_receive":           pay_receive,
    }


# ── Portfolio VaR ─────────────────────────────────────────────────────────────

def g2pp_portfolio_var(
    curve:          DiscountCurve,
    params:         G2ppParams,
    portfolio_dv01: float,
    horizon:        float,
    confidence:     float = 0.99,
    n_paths:        int   = 20_000,
    seed:           int   = 42,
    ref_maturity:   float = 10.0,
) -> dict:
    """
    Portfolio VaR under G2++ using analytical ZCB prices at the scenario horizon.

    Simulates (x, y) at the horizon, computes the yield of a 10Y ZCB in each
    scenario, derives PnL = DV01 × (−Δy_bps), and returns VaR/CVaR.

    Parameters
    ----------
    portfolio_dv01 : dollar DV01 of the portfolio (+ = long duration)
    horizon        : risk horizon in years (e.g. 1/250 for 1 business day)
    confidence     : e.g. 0.99 for 99% VaR
    ref_maturity   : maturity of the reference ZCB for yield change (default 10Y)
    """
    sim = simulate_g2pp(curve, params, horizon, n_steps=1,
                        n_paths=n_paths, seed=seed)

    x_h = sim.x_paths[:, -1]
    y_h = sim.y_paths[:, -1]
    T   = horizon + ref_maturity

    p_scenario = np.array([
        g2pp_zcb(curve, params, horizon, T, float(x_h[i]), float(y_h[i]))
        for i in range(n_paths)
    ])
    y_scenario  = -np.log(p_scenario) / ref_maturity
    # Use forward yield (from horizon to T) as the base; this is what the market
    # prices in today, and the model's x=y=0 path reprices exactly this.
    p_base_fwd  = g2pp_zcb(curve, params, horizon, T, 0.0, 0.0)
    y_base      = -math.log(p_base_fwd) / ref_maturity
    delta_y_bps = (y_scenario - y_base) * 10_000

    pnl  = portfolio_dv01 * (-delta_y_bps)
    conf = 1.0 - confidence
    var  = float(np.percentile(pnl, conf * 100))
    idx  = pnl <= var
    cvar = float(np.mean(pnl[idx])) if idx.sum() > 0 else var

    return {
        "var_usd":         round(var, 2),
        "cvar_usd":        round(cvar, 2),
        "var_bps":         round(float(np.percentile(delta_y_bps, conf * 100)), 2),
        "pnl_mean":        round(float(np.mean(pnl)), 2),
        "pnl_std":         round(float(np.std(pnl)), 2),
        "n_paths":         n_paths,
        "confidence":      confidence,
        "horizon_years":   horizon,
    }


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_g2pp(
    curve:     DiscountCurve,
    expiries:  list[float],
    tenors:    list[float],
    atm_vols:  list[list[float]],
    freq:      int   = 2,
    init:      G2ppParams | None = None,
    n_quad:    int   = 48,
) -> G2ppParams:
    """
    Calibrate G2++ parameters (a, b, σ, η, ρ) to a matrix of ATM swaption vols
    using numerical implied vol inversion.

    The calibration minimises:
        Σᵢⱼ [σ_mkt(Tᵢ, τⱼ) − σ_model(Tᵢ, τⱼ; a,b,σ,η,ρ)]²

    where σ_model is extracted by inverting the G2++ MC swaption PV via
    Black-76 forward price.

    Parameters
    ----------
    expiries  : list of option expiries [T₁, T₂, ...]
    tenors    : list of swap tenors [τ₁, τ₂, ...]
    atm_vols  : matrix of ATM Black vols, shape (len(expiries), len(tenors))
    init      : initial parameter guess; defaults to typical USD swaption params

    Returns
    -------
    Calibrated G2ppParams
    """
    if init is None:
        init = G2ppParams(a=0.05, b=0.10, sigma=0.010, eta=0.008, rho=-0.30)

    x0 = [init.a, init.b, init.sigma, init.eta, init.rho]
    bounds = [(1e-4, 2.0), (1e-4, 2.0), (1e-5, 0.20), (1e-5, 0.20), (-0.9, 0.9)]

    from scipy.optimize import brentq as _brentq
    from scipy.stats import norm as _norm

    def _black76_implied_vol(F: float, K: float, T: float, pv: float,
                              ann: float, N: float, pay_rec: str) -> float:
        if pv <= 0 or ann <= 0:
            return 0.0
        premium = pv / (N * ann)
        if pay_rec == "payer":
            intrinsic = max(F - K, 0.0)
        else:
            intrinsic = max(K - F, 0.0)
        if premium <= intrinsic + 1e-12 or F <= 0 or K <= 0:
            return 0.0
        def f(v: float) -> float:
            sqT = math.sqrt(T)
            d1  = (math.log(F / K) + 0.5 * v**2 * T) / (v * sqT)
            d2  = d1 - v * sqT
            if pay_rec == "payer":
                return F * _norm.cdf(d1) - K * _norm.cdf(d2) - premium
            else:
                return K * _norm.cdf(-d2) - F * _norm.cdf(-d1) - premium
        try:
            return _brentq(f, 1e-6, 10.0)
        except Exception:
            return 0.0

    def objective(x: list) -> float:
        a, b, s, e, r = x
        if a <= 0 or b <= 0 or s <= 0 or e <= 0 or not -1 < r < 1:
            return 1e10
        try:
            p = G2ppParams(a=a, b=b, sigma=s, eta=e, rho=r)
        except ValueError:
            return 1e10
        total = 0.0
        for i, T in enumerate(expiries):
            for j, tau in enumerate(tenors):
                tgt = atm_vols[i][j]
                res = g2pp_swaption_mc(curve, p, T, tau, strike=None,
                                       notional=1.0, pay_receive="payer",
                                       freq=freq, n_paths=2000, seed=42)
                # Extract implied vol
                iv = _black76_implied_vol(
                    res["forward_swap_rate_pct"] / 100,
                    res["forward_swap_rate_pct"] / 100,  # ATM
                    T, res["pv"], res["annuity"], 1.0, "payer",
                )
                total += (iv - tgt) ** 2
        return total

    result = optimize.minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 100, "ftol": 1e-8})
    a, b, s, e, r = result.x
    return G2ppParams(
        a=max(a, 1e-4), b=max(b, 1e-4),
        sigma=max(s, 1e-5), eta=max(e, 1e-5),
        rho=float(np.clip(r, -0.9, 0.9)),
    )

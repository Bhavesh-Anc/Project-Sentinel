"""
sofr_engine/lmm.py — SOFR Libor Market Model (BGM / Brace-Gatarek-Musiela)

Implements a discrete log-normal LMM for a grid of SOFR forward rates.
Under the terminal measure Q^{T_N} the dynamics of each forward rate F_k are:

    dF_k = F_k [- Σ_{m=k+1}^{N-1} ρ_{km} σ_k σ_m α_m F_m / (1+α_m F_m)] dt
         + F_k σ_k dW_k^{T_N}

or equivalently in log-space:

    d ln F_k = (μ_k − σ_k²/2) dt + σ_k dW_k

Drift vector is fully vectorised (no Python loops over paths).

Features
--------
- Exponential correlation  ρ_{ij} = exp(−λ |T_i − T_j|)
- Predictor-corrector Euler in log-space; antithetic variates
- Caplet / Cap: exact Black-76 + Monte-Carlo validation under forward measure
- Swaption MC  under Q^{T_N} with correct measure-change weighting
- Rebonato's approximate swaption vol from LMM vols & correlation
- Cap vol bootstrap: recover caplet vols from flat ATM cap implied vols
- Correlation decay calibration via Rebonato residual least-squares
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

from .curve import DiscountCurve


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LMMParams:
    """
    Parameters for the log-normal SOFR LMM on a discrete tenor grid.

    Attributes
    ----------
    tenors      T_0 < T_1 < … < T_N  (N+1 values in years from today)
                There are N forward rates: F_k covers period [T_k, T_{k+1}].
    vols        flat instantaneous Black vol σ_k for each F_k; shape (N,)
    corr_decay  λ in ρ_{ij} = exp(−λ |T_i − T_j|).  0 → all ρ = 1.
                Ignored when corr_matrix is given.
    corr_matrix optional explicit (N,N) correlation matrix override
    """
    tenors     : np.ndarray
    vols       : np.ndarray
    corr_decay : float                  = 0.10
    corr_matrix: Optional[np.ndarray]  = None

    def __post_init__(self) -> None:
        self.tenors = np.asarray(self.tenors, dtype=float)
        self.vols   = np.asarray(self.vols,   dtype=float)
        n1 = len(self.tenors)
        n  = len(self.vols)
        if n1 < 2:
            raise ValueError("Need ≥ 2 tenors (1 forward rate).")
        if n != n1 - 1:
            raise ValueError(f"vols length {n} must equal len(tenors)−1 = {n1-1}.")
        if not np.all(np.diff(self.tenors) > 0):
            raise ValueError("tenors must be strictly increasing.")
        if np.any(self.vols <= 0):
            raise ValueError("All vols must be positive.")
        if self.corr_decay < 0:
            raise ValueError("corr_decay must be non-negative.")
        if self.corr_matrix is not None:
            cm = np.asarray(self.corr_matrix, dtype=float)
            if cm.shape != (n, n):
                raise ValueError(f"corr_matrix must be ({n}, {n}).")
            self.corr_matrix = cm

    @property
    def N(self) -> int:
        return len(self.vols)

    @property
    def alpha(self) -> np.ndarray:
        """Accrual fractions α_k = T_{k+1} − T_k."""
        return np.diff(self.tenors)

    def correlation(self) -> np.ndarray:
        """Return (N, N) correlation matrix."""
        if self.corr_matrix is not None:
            return self.corr_matrix
        T = self.tenors[:-1]
        return np.exp(-self.corr_decay * np.abs(T[:, None] - T[None, :]))

    def with_vols(self, vols: np.ndarray) -> "LMMParams":
        return LMMParams(
            tenors=self.tenors.copy(),
            vols=np.asarray(vols, dtype=float),
            corr_decay=self.corr_decay,
            corr_matrix=self.corr_matrix,
        )

    def with_corr_decay(self, lam: float) -> "LMMParams":
        return LMMParams(
            tenors=self.tenors.copy(),
            vols=self.vols.copy(),
            corr_decay=float(lam),
            corr_matrix=None,
        )


@dataclass
class LMMSimResult:
    """
    Output of simulate_lmm.

    forward_paths  (n_steps+1, n_paths, N) — F_k at each step, each path
    tenors         (N+1,)
    time_steps     (n_steps+1,)  — simulation time grid
    n_paths        total paths (base + antithetic)
    """
    forward_paths : np.ndarray   # (n_steps+1, n_paths, N)
    tenors        : np.ndarray   # (N+1,)
    time_steps    : np.ndarray   # (n_steps+1,)
    n_paths       : int

    @property
    def N(self) -> int:
        return self.forward_paths.shape[2]

    @property
    def alpha(self) -> np.ndarray:
        return np.diff(self.tenors)

    def step_for_tenor(self, T: float) -> int:
        return int(np.argmin(np.abs(self.time_steps - T)))

    def forwards_at(self, T: float) -> np.ndarray:
        """Return forward rates at time T; shape (n_paths, N)."""
        return self.forward_paths[self.step_for_tenor(T)]

    def swap_rate(self, T: float, k_start: int, k_end: int) -> np.ndarray:
        """
        Swap rate S_{k_start, k_end}(T) from simulated forwards.
        S = (1 − P(T, T_{k_end})) / A_{k_start, k_end}(T)
        Returns shape (n_paths,).
        """
        F   = self.forwards_at(T)
        alp = self.alpha
        P   = np.ones(self.n_paths)
        ann = np.zeros(self.n_paths)
        for i in range(k_start, k_end):
            P   = P / (1.0 + alp[i] * F[:, i])
            ann = ann + alp[i] * P
        return (1.0 - P) / np.maximum(ann, 1e-15)

    def annuity(self, T: float, k_start: int, k_end: int) -> np.ndarray:
        """A_{k_start, k_end}(T); shape (n_paths,)."""
        F   = self.forwards_at(T)
        alp = self.alpha
        P   = np.ones(self.n_paths)
        ann = np.zeros(self.n_paths)
        for i in range(k_start, k_end):
            P   = P / (1.0 + alp[i] * F[:, i])
            ann = ann + alp[i] * P
        return ann


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def initial_forwards(curve: DiscountCurve, tenors: Sequence[float]) -> np.ndarray:
    """
    F_k(0) = (P(0,T_k) / P(0,T_{k+1}) − 1) / α_k.  Shape (N,).
    """
    tenors = np.asarray(tenors, dtype=float)
    alpha  = np.diff(tenors)
    dfs    = np.array([float(curve.df(t)) for t in tenors])
    return (dfs[:-1] / dfs[1:] - 1.0) / alpha


def exponential_correlation(tenors: np.ndarray, decay: float) -> np.ndarray:
    """ρ_{ij} = exp(−λ |T_i − T_j|), shape (N, N), N = len(tenors)−1."""
    T = np.asarray(tenors)[:-1]
    return np.exp(-decay * np.abs(T[:, None] - T[None, :]))


def _drift_weights(params: LMMParams) -> np.ndarray:
    """
    Precompute upper-triangular weight matrix W[k,m] = ρ_{km} σ_k σ_m for m > k.
    Drift: μ_k = −Σ_{m=k+1}^{N-1} W[k,m] α_m F_m/(1+α_m F_m).
    Vectorised as  μ = −(denom @ W.T)  where denom[p,m] = α_m F_{pm}/(1+α_m F_{pm}).
    """
    return np.triu(np.outer(params.vols, params.vols) * params.correlation(), k=1)


def _corr_cholesky(params: LMMParams) -> np.ndarray:
    rho = params.correlation()
    try:
        return np.linalg.cholesky(rho)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(rho + 1e-9 * np.eye(params.N))


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_lmm(
    curve     : DiscountCurve,
    params    : LMMParams,
    n_steps   : int  = 100,
    n_paths   : int  = 2_000,
    seed      : Optional[int] = 42,
    antithetic: bool = True,
) -> LMMSimResult:
    """
    Simulate LMM forward rates under Q^{T_N} (terminal measure).

    Uses predictor-corrector Euler in log-space with antithetic variates.
    Simulation runs from t=0 to T_{N-1} (last fixing date) in n_steps steps.

    Parameters
    ----------
    curve      : initial DiscountCurve (for initial forward rates)
    params     : LMMParams
    n_steps    : number of time steps from 0 to T_{N-1}
    n_paths    : total paths; if antithetic, half are base, half are mirror
    seed       : RNG seed (None for random)
    antithetic : use antithetic variance reduction

    Returns
    -------
    LMMSimResult  with forward_paths shape (n_steps+1, n_paths, N)
    """
    rng    = np.random.default_rng(seed)
    N      = params.N
    tenors = params.tenors
    alpha  = params.alpha
    vols   = params.vols
    W      = _drift_weights(params)    # (N, N) upper-tri weight matrix
    L      = _corr_cholesky(params)    # (N, N) Cholesky of correlation

    T_end     = tenors[-2]             # T_{N-1}: last fixing date
    dt        = T_end / n_steps
    sqrt_dt   = math.sqrt(dt)
    time_steps = np.linspace(0.0, T_end, n_steps + 1)

    F0 = initial_forwards(curve, tenors)
    if np.any(F0 <= 0):
        raise ValueError("Non-positive initial forward rates — check curve.")
    z0     = np.log(F0)
    vols2  = vols ** 2

    n_base  = n_paths // 2 if antithetic else n_paths
    n_total = 2 * n_base  if antithetic else n_base

    forward_paths = np.empty((n_steps + 1, n_total, N))
    forward_paths[0] = F0[None, :]

    z_b = np.tile(z0, (n_base, 1))    # (n_base, N)
    z_a = z_b.copy() if antithetic else None

    def _drift(F_arr: np.ndarray) -> np.ndarray:
        denom = alpha * F_arr / (1.0 + alpha * F_arr)   # (n_base, N)
        return -(denom @ W.T)                             # (n_base, N)

    for s in range(n_steps):
        Z = rng.standard_normal((n_base, N)) @ L.T       # correlated normals

        # ── Base paths (predictor-corrector) ──────────────────────────────────
        F_b   = np.exp(z_b)
        mu_b0 = _drift(F_b)
        z_tmp = z_b + (mu_b0 - 0.5 * vols2) * dt + vols * sqrt_dt * Z
        mu_b1 = _drift(np.exp(z_tmp))
        z_b   = z_b + (0.5 * (mu_b0 + mu_b1) - 0.5 * vols2) * dt + vols * sqrt_dt * Z
        forward_paths[s + 1, :n_base] = np.exp(z_b)

        # ── Antithetic paths (negate Z) ────────────────────────────────────────
        if antithetic:
            F_a   = np.exp(z_a)
            mu_a0 = _drift(F_a)
            z_tmp = z_a + (mu_a0 - 0.5 * vols2) * dt - vols * sqrt_dt * Z
            mu_a1 = _drift(np.exp(z_tmp))
            z_a   = z_a + (0.5 * (mu_a0 + mu_a1) - 0.5 * vols2) * dt - vols * sqrt_dt * Z
            forward_paths[s + 1, n_base:n_total] = np.exp(z_a)

    return LMMSimResult(forward_paths, tenors.copy(), time_steps, n_total)


# ─────────────────────────────────────────────────────────────────────────────
# Black-76 caplet / cap  (closed-form, model-exact)
# ─────────────────────────────────────────────────────────────────────────────

def _black76(F: float, K: float, vol: float, T: float, is_call: bool = True) -> float:
    """Undiscounted Black-76 price for a European option on a forward."""
    if T <= 0 or vol <= 0:
        return max((F - K) if is_call else (K - F), 0.0)
    sq = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / (vol * sq)
    d2 = d1 - vol * sq
    if is_call:
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    return K * norm.cdf(-d2) - F * norm.cdf(-d1)


def caplet_black76(
    curve    : DiscountCurve,
    params   : LMMParams,
    k        : int,
    strike   : float,
    notional : float = 1.0,
    is_cap   : bool  = True,
) -> float:
    """
    Exact Black-76 caplet (is_cap=True) or floorlet on forward rate F_k.
    PV = P(0, T_{k+1}) × α_k × Black76(F_k(0), K, σ_k, T_k)
    """
    tenors  = params.tenors
    alpha   = params.alpha
    T_fix   = tenors[k]
    T_pay   = tenors[k + 1]
    F_k     = initial_forwards(curve, tenors)[k]
    sigma   = params.vols[k]
    df_pay  = float(curve.df(T_pay))
    return notional * alpha[k] * df_pay * _black76(F_k, strike, sigma, T_fix, is_cap)


def cap_black76(
    curve    : DiscountCurve,
    params   : LMMParams,
    strike   : float,
    notional : float = 1.0,
    is_cap   : bool  = True,
) -> float:
    """
    Black-76 cap (or floor) price — sum of individual caplet / floorlet prices.
    """
    return sum(
        caplet_black76(curve, params, k, strike, notional, is_cap)
        for k in range(params.N)
    )


def cap_implied_vol(
    curve    : DiscountCurve,
    params   : LMMParams,
    strike   : float,
    mkt_price: float,
    notional : float = 1.0,
    is_cap   : bool  = True,
    vol_lo   : float = 1e-4,
    vol_hi   : float = 5.0,
) -> float:
    """
    Implied flat vol of the cap/floor from market price.  Uses brentq.
    """
    def f(sigma: float) -> float:
        p = params.with_vols(np.full(params.N, sigma))
        return cap_black76(curve, p, strike, notional, is_cap) - mkt_price

    return float(brentq(f, vol_lo, vol_hi))


# ─────────────────────────────────────────────────────────────────────────────
# Caplet MC under Q^{T_{k+1}} (forward measure) — F_k is a martingale
# ─────────────────────────────────────────────────────────────────────────────

def caplet_lmm_mc(
    curve    : DiscountCurve,
    params   : LMMParams,
    k        : int,
    strike   : float,
    notional : float = 1.0,
    n_paths  : int   = 4_000,
    seed     : Optional[int] = 42,
) -> dict:
    """
    MC caplet under Q^{T_{k+1}} where F_k is a driftless martingale.
    Agrees with Black-76 by construction; useful for convergence tests.

    Returns dict with 'pv', 'std_err', 'black76_pv'.
    """
    tenors  = params.tenors
    T_fix   = tenors[k]
    T_pay   = tenors[k + 1]
    alpha_k = params.alpha[k]
    F0      = initial_forwards(curve, tenors)[k]
    sigma   = params.vols[k]
    df_pay  = float(curve.df(T_pay))

    rng    = np.random.default_rng(seed)
    n_base = n_paths // 2
    Z      = rng.standard_normal(n_base)
    Z      = np.concatenate([Z, -Z])    # antithetic

    F_T    = F0 * np.exp(-0.5 * sigma**2 * T_fix + sigma * math.sqrt(T_fix) * Z)
    payoff = np.maximum(F_T - strike, 0.0) * alpha_k * df_pay * notional

    pv       = float(payoff.mean())
    std_err  = float(payoff.std(ddof=1)) / math.sqrt(len(Z))
    black76  = caplet_black76(curve, params, k, strike, notional, is_cap=True)

    return {"pv": pv, "std_err": std_err, "black76_pv": black76}


# ─────────────────────────────────────────────────────────────────────────────
# Swaption MC under Q^{T_N}
# ─────────────────────────────────────────────────────────────────────────────

def swaption_lmm_mc(
    sim      : LMMSimResult,
    curve    : DiscountCurve,
    k_start  : int,
    k_end    : int,
    strike   : float,
    notional : float = 1.0,
    is_payer : bool  = True,
) -> dict:
    """
    European payer (or receiver) swaption priced from simulation under Q^{T_N}.

    The swaption expires at T_{k_start} on the par swap [T_{k_start}, T_{k_end}].

    Under Q^{T_N}, the correctly discounted payoff is:

        payoff_TN = (S − K)^+ × [A(T_exp) / P(T_exp, T_N)]

    where  A/P(T_exp, T_N) = Σ_{i=k_start}^{k_end-1} α_i × Π_{m=i+1}^{N-1}(1+α_m F_m)

    and the price is:

        V(0) = P(0, T_N) × E^{T_N}[payoff_TN]

    Returns dict: pv, std_err, swap_rate_mean, annuity_mean.
    """
    tenors  = sim.tenors
    alpha   = sim.alpha
    N       = sim.N
    n_paths = sim.n_paths
    T_exp   = tenors[k_start]

    F_exp = sim.forwards_at(T_exp)    # (n_paths, N)

    # Suffix product: suf[p, m] = Π_{j=m}^{N-1} (1+α_j F_j(T_exp))
    suf = np.ones((n_paths, N + 1))
    for m in range(N - 1, k_start - 1, -1):
        suf[:, m] = suf[:, m + 1] * (1.0 + alpha[m] * F_exp[:, m])

    # A(T_exp) / P(T_exp, T_N) = Σ_{i=k_start}^{k_end-1} α_i × suf[:, i+1]
    A_norm = np.zeros(n_paths)
    for i in range(k_start, k_end):
        A_norm += alpha[i] * suf[:, i + 1]

    # Swap rate: S = (1 − P(T_exp, T_{k_end})) / A(T_exp)
    # P(T_exp, T_{k_end}) = suf[:, k_end] / suf[:, k_start]
    # A(T_exp) = A_norm / suf[:, k_start]... but we can simplify:
    # S = (suf[:, k_start] − suf[:, k_end]) / A_norm  (×suf cancels)
    S = (suf[:, k_start] - suf[:, k_end]) / np.maximum(A_norm, 1e-15)

    raw    = np.maximum(S - strike, 0.0) if is_payer else np.maximum(strike - S, 0.0)
    payoff = raw * A_norm               # payoff in Q^{T_N} units

    df_TN   = float(curve.df(tenors[-1]))
    pv      = notional * df_TN * float(payoff.mean())
    std_err = notional * df_TN * float(payoff.std(ddof=1)) / math.sqrt(n_paths)

    # Recover A in real money: A(T_exp) = A_norm / suf[:, k_start]
    A_real  = A_norm / np.maximum(suf[:, k_start], 1e-15)

    return {
        "pv"             : pv,
        "std_err"        : std_err,
        "swap_rate_mean" : float(S.mean()),
        "annuity_mean"   : float(A_real.mean()),
        "is_payer"       : is_payer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rebonato's approximate swaption vol from LMM vols & correlation
# ─────────────────────────────────────────────────────────────────────────────

def rebonato_swaption_vol(
    curve   : DiscountCurve,
    params  : LMMParams,
    k_start : int,
    k_end   : int,
) -> float:
    """
    Rebonato (2002) approximation for the implied Black swaption vol.

    σ_S² = Σ_{i,j=k_start}^{k_end-1} w_i w_j ρ_{ij} σ_i σ_j

    where w_k = α_k F_k(0) P(0,T_{k+1}) / [S_0 × A_0 × P(0,T_{k_start})]

    Returns σ_S (a single float).
    """
    tenors  = params.tenors
    alpha   = params.alpha
    vols    = params.vols
    rho     = params.correlation()
    F0      = initial_forwards(curve, tenors)

    df_ks   = float(curve.df(tenors[k_start]))
    dfs     = np.array([float(curve.df(tenors[j])) for j in range(k_start, k_end + 1)])
    P_norm  = dfs / dfs[0]                 # P(0,T_j)/P(0,T_{k_start})

    A0 = sum(alpha[k_start + i] * P_norm[i + 1] for i in range(k_end - k_start))
    S0 = (1.0 - P_norm[-1]) / max(A0, 1e-15)

    weights = np.zeros(params.N)
    for k in range(k_start, k_end):
        df_k1 = float(curve.df(tenors[k + 1]))
        weights[k] = alpha[k] * F0[k] * df_k1 / (S0 * A0 * df_ks)

    # σ_S² = w^T (ρ * outer(σ,σ)) w  (only k_start..k_end-1 block)
    w = weights[k_start:k_end]
    v = vols[k_start:k_end]
    rho_blk = rho[k_start:k_end, k_start:k_end]
    sigma_sq = float(w @ (rho_blk * np.outer(v, v)) @ w)
    return math.sqrt(max(sigma_sq, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_caplet_vols(
    curve          : DiscountCurve,
    params_template: LMMParams,
    cap_flat_vols  : Sequence[float],
) -> LMMParams:
    """
    Bootstrap caplet vols σ_k from a sequence of flat ATM cap implied vols.

    cap_flat_vols[k] is the Black flat implied vol for the (k+1)-caplet cap
    covering periods 0 … k.  Returns LMMParams with calibrated vols.

    Algorithm: bootstrap — at each step, the new caplet's vol is found such
    that the total cap price with already-calibrated previous caplets equals
    the market price using flat vol cap_flat_vols[k].
    """
    N = params_template.N
    if len(cap_flat_vols) != N:
        raise ValueError(f"Need {N} cap vols, got {len(cap_flat_vols)}.")
    cap_flat_vols = list(cap_flat_vols)
    tenors  = params_template.tenors
    alpha   = params_template.alpha
    F0      = initial_forwards(curve, tenors)
    K       = float(F0[0])              # ATM strike = 1st forward (approximate)

    vols = np.zeros(N)

    for k in range(N):
        sigma_cap = float(cap_flat_vols[k])
        T_fix_k   = tenors[k]
        df_pay_k  = float(curve.df(tenors[k + 1]))

        # Market cap price (flat vol for full 0..k cap)
        mkt_cap = 0.0
        for j in range(k + 1):
            mkt_cap += alpha[j] * float(curve.df(tenors[j+1])) * _black76(
                F0[j], K, sigma_cap, tenors[j], True)

        # Price from already-calibrated caplets 0..k-1
        prev_cap = sum(
            alpha[j] * float(curve.df(tenors[j+1])) * _black76(F0[j], K, vols[j], tenors[j], True)
            for j in range(k)
        )

        caplet_target = mkt_cap - prev_cap

        if caplet_target <= 1e-12:
            vols[k] = sigma_cap
            continue

        def obj(sigma: float) -> float:
            return alpha[k] * df_pay_k * _black76(F0[k], K, sigma, T_fix_k, True) - caplet_target

        try:
            vols[k] = brentq(obj, 1e-4, 5.0)
        except ValueError:
            vols[k] = sigma_cap

    return params_template.with_vols(vols)


def calibrate_corr_decay(
    curve         : DiscountCurve,
    params        : LMMParams,
    swaption_specs: Sequence[Tuple[int, int, float]],
    decay_bounds  : Tuple[float, float] = (0.001, 3.0),
) -> Tuple[float, float]:
    """
    Calibrate the exponential correlation decay λ to swaption implied vols.

    swaption_specs: list of (k_start, k_end, mkt_vol) tuples.
    Uses Rebonato's approximation for the model swaption vol.

    Returns (calibrated_lambda, final_rmse).
    """
    def objective(lam: float) -> float:
        p = params.with_corr_decay(lam)
        err = 0.0
        for k_start, k_end, mkt_vol in swaption_specs:
            model_vol = rebonato_swaption_vol(curve, p, k_start, k_end)
            err += (model_vol - mkt_vol) ** 2
        return err

    res   = minimize_scalar(objective, bounds=decay_bounds, method="bounded")
    lam   = float(res.x)
    rmse  = math.sqrt(res.fun / max(len(swaption_specs), 1))
    return lam, rmse


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: implied vol from MC swaption
# ─────────────────────────────────────────────────────────────────────────────

def swaption_implied_vol(
    pv       : float,
    curve    : DiscountCurve,
    params   : LMMParams,
    k_start  : int,
    k_end    : int,
    strike   : float,
    notional : float = 1.0,
    is_payer : bool  = True,
    vol_lo   : float = 1e-4,
    vol_hi   : float = 5.0,
) -> float:
    """
    Implied Black swaption vol from a dollar PV.  Uses Black-76 swap-rate formula.
    """
    tenors = params.tenors
    F0     = initial_forwards(curve, tenors)
    alpha  = params.alpha

    # Initial swap rate & annuity
    P  = 1.0
    A0 = 0.0
    for i in range(k_start, k_end):
        P   = P / (1.0 + alpha[i] * F0[i])
        A0 += alpha[i] * P
    S0 = (1.0 - P) / max(A0, 1e-15)

    df_ks = float(curve.df(tenors[k_start]))
    ann_0 = A0 * df_ks                  # annuity discounted to today

    def f(sigma: float) -> float:
        raw = _black76(S0, strike, sigma, tenors[k_start], is_payer)
        model_pv = notional * ann_0 * raw
        return model_pv - pv

    try:
        return float(brentq(f, vol_lo, vol_hi))
    except ValueError:
        return float("nan")


def lmm_corr_from_pca(
    rate_changes: np.ndarray,
    n_components: int = 3,
) -> np.ndarray:
    """
    Build an empirical LMM forward-rate correlation matrix from historical
    data via PCA (level / slope / curvature factor decomposition).

    The single-parameter exponential family ``ρ_{ij} = exp(−λ|T_i−T_j|)``
    cannot capture the true factor structure of the SOFR curve.  This
    function estimates the full (N×N) correlation matrix directly from
    observed daily forward-rate changes using truncated SVD, retaining the
    dominant ``n_components`` factors.

    Parameters
    ----------
    rate_changes : (T_days, N) array of daily changes in N forward rates.
                   Columns must correspond to the N forward rates implied by
                   ``LMMParams.tenors`` in the same order.
    n_components : number of PCA factors to retain (default 3:
                   level, slope, curvature).  Higher values capture more
                   idiosyncratic variance but may overfit to historical noise.

    Returns
    -------
    corr : (N, N) positive-semi-definite correlation matrix with unit
           diagonal.  Pass directly to ``LMMParams(corr_matrix=corr)``.

    Notes
    -----
    Algorithm:
      1. Demean the change matrix X (T_days × N).
      2. Compute truncated SVD: X ≈ U Σ V^T, keep first n_components rows of V^T.
      3. Loadings L = V^T[:k].T  — shape (N, k).
      4. Factor correlation: C = L L^T + diag(residual variance).
      5. Normalise to unit diagonal → correlation matrix.
    """
    X = np.asarray(rate_changes, dtype=float)
    if X.ndim != 2:
        raise ValueError("rate_changes must be 2-D: (T_days, N)")
    T, N = X.shape
    k = min(n_components, N, T - 1)

    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    L = Vt[:k].T  # (N, k) PC loadings

    # Factor-model correlation: L L^T + residual diagonal
    C = L @ L.T
    resid = np.maximum(1.0 - np.diag(C), 0.0)
    C += np.diag(resid)

    # Normalise to correlation matrix (unit diagonal)
    d = np.sqrt(np.maximum(np.diag(C), 1e-15))
    corr = C / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


__all__ = [
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
    "lmm_corr_from_pca",
]

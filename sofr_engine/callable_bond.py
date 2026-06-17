"""
Callable / Putable Bond Pricing via Hull-White Trinomial Tree.

Provides:
  - HWTreeParams     : tree construction parameters
  - CallableBond     : bond + optionality schedule
  - CallableBondResult : price, OAS, effective dur/cvx
  - price_callable_bond : backward induction on calibrated HW lattice
  - straight_bond_price : closed-form price without optionality
  - calibrate_oas       : brentq root-find to match market price
  - effective_duration  : finite-difference effective duration
  - effective_convexity : finite-difference effective convexity

References
----------
Hull, J. & White, A. (1994). Numerical procedures for implementing
  term structure models I: Single-factor models. Journal of Derivatives, 2(1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.optimize import brentq

from sofr_engine.curve import DiscountCurve


# ── Parameter dataclasses ─────────────────────────────────────────────────────

@dataclass
class HWTreeParams:
    """
    Hull-White trinomial tree construction parameters.

    Parameters
    ----------
    a     : mean-reversion speed (e.g. 0.05)
    sigma : short-rate volatility (e.g. 0.01)
    dt    : time step in years (e.g. 0.25 for quarterly)
    """
    a:     float
    sigma: float
    dt:    float

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError(f"a must be positive, got {self.a}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")

    @property
    def dr(self) -> float:
        """Node spacing dr = sigma * sqrt(3 * dt)."""
        return self.sigma * math.sqrt(3.0 * self.dt)

    @property
    def j_max(self) -> int:
        """Maximum index j_max = ceil(0.184 / (a * dt))."""
        return max(1, math.ceil(0.184 / (self.a * self.dt)))


@dataclass
class CallableBond:
    """
    Fixed-coupon bond with optional call and/or put schedule.

    Parameters
    ----------
    face          : face/par value (e.g. 100.0)
    coupon        : annual coupon rate (decimal, e.g. 0.05)
    maturity      : maturity in years
    freq          : coupon payments per year (1=annual, 2=semi-annual)
    call_schedule : list of (time, call_price) — issuer can redeem at call_price
    put_schedule  : list of (time, put_price)  — holder can redeem at put_price
    """
    face:          float
    coupon:        float
    maturity:      float
    freq:          int = 2
    call_schedule: List[Tuple[float, float]] = field(default_factory=list)
    put_schedule:  List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.face <= 0:
            raise ValueError("face must be positive")
        if self.coupon < 0:
            raise ValueError("coupon must be non-negative")
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.freq <= 0:
            raise ValueError("freq must be positive")

    @property
    def coupon_times(self) -> List[float]:
        """Regular coupon payment times."""
        dt = 1.0 / self.freq
        times = []
        t = dt
        while t <= self.maturity + 1e-9:
            times.append(round(t, 10))
            t += dt
        return times

    @property
    def coupon_payment(self) -> float:
        """Coupon cash-flow per period = face × annual_rate / freq."""
        return self.face * self.coupon / self.freq


@dataclass
class CallableBondResult:
    """
    Result of callable/putable bond pricing.

    Attributes
    ----------
    price              : model-derived dirty price
    straight_price     : price without any optionality
    option_value       : straight_price - price (call) or price - straight (put)
    oas                : option-adjusted spread (decimal)
    effective_duration : duration via ±1bp parallel shift
    effective_convexity: convexity via ±1bp parallel shift
    """
    price:               float
    straight_price:      float
    option_value:        float
    oas:                 float
    effective_duration:  float
    effective_convexity: float


# ── Tree internals ─────────────────────────────────────────────────────────────

def _branching_probs(j: int, j_max: int, M: float) -> Tuple[int, float, float, float]:
    """
    Return (shift, pu, pm, pd) for node j at a given time step.

    shift = 0 → normal (j-1, j, j+1)
    shift = -1 → upward (j, j+1, j+2)
    shift = +1 → downward (j-2, j-1, j)
    """
    eta = j * M  # = a * j * dt
    if j == j_max:
        # Top boundary — branch downward
        pu = 1.0/6.0 + eta*eta/2.0 - eta/2.0
        pm = -1.0/3.0 - eta*eta + 2.0*eta/2.0
        # ensure valid; clamp
        pu = max(0.0, min(1.0, pu))
        pm_raw = 7.0/6.0 + eta*eta/2.0 - 3.0*eta/2.0
        pm = max(0.0, min(1.0, pm_raw))
        pd = 1.0 - pu - pm
        pd = max(0.0, pd)
        return 1, pu, pm, pd
    elif j == -j_max:
        # Bottom boundary — branch upward
        pu_raw = 1.0/6.0 + eta*eta/2.0 + 3.0*eta/2.0
        pu = max(0.0, min(1.0, pu_raw))
        pm_raw = -1.0/3.0 - eta*eta - 2.0*eta/2.0
        pm = max(0.0, min(1.0, 7.0/6.0 + eta*eta/2.0 + 3.0*eta/2.0 - pu))
        pd = 1.0 - pu - pm
        pd = max(0.0, pd)
        return -1, pu, pm, pd
    else:
        # Normal branching
        pu = 1.0/6.0 + (eta*eta - eta) / 2.0
        pm = 2.0/3.0 - eta*eta
        pd = 1.0/6.0 + (eta*eta + eta) / 2.0
        pu = max(0.0, min(1.0, pu))
        pm = max(0.0, min(1.0, pm))
        pd = max(0.0, 1.0 - pu - pm)
        return 0, pu, pm, pd


def _build_tree(curve: DiscountCurve, params: HWTreeParams, n_steps: int, oas: float = 0.0):
    """
    Build calibrated HW trinomial tree.

    Returns
    -------
    alpha : (n_steps+1,) drift array so that E[r(t)] fits discount curve
    Q     : Arrow-Debreu state prices (list of dicts {j: Q_j})
    """
    dt = params.dt
    dr = params.dr
    M  = params.a * dt
    jmax = params.j_max

    alpha = np.zeros(n_steps + 1)
    Q = [dict() for _ in range(n_steps + 1)]

    # t=0: single node j=0, Q=1
    Q[0][0] = 1.0

    for i in range(n_steps):
        t_i = i * dt
        t_next = (i + 1) * dt

        # Target: P(0, t_{i+1}) from market curve
        P_target = float(curve.df(t_next)) * math.exp(-oas * t_next)

        # Compute alpha[i] so that sum_j Q[i][j] * exp(-(alpha[i] + j*dr)*dt) = P_target
        # => alpha[i] = -1/dt * log(P_target / sum_j Q[i][j] * exp(-j*dr*dt))
        sum_q_exp = sum(qval * math.exp(-j * dr * dt)
                        for j, qval in Q[i].items())
        if sum_q_exp < 1e-14:
            sum_q_exp = 1e-14
        alpha[i] = -math.log(P_target / sum_q_exp) / dt

        # Propagate Arrow-Debreu prices forward
        for j, qval in Q[i].items():
            j = int(j)
            r_j = alpha[i] + j * dr
            disc = math.exp(-r_j * dt)
            shift, pu, pm, pd = _branching_probs(j, jmax, M)

            if shift == 0:
                targets = [(j + 1, pu), (j, pm), (j - 1, pd)]
            elif shift == 1:   # top boundary: branch down
                targets = [(j, pu), (j - 1, pm), (j - 2, pd)]
            else:              # bottom boundary: branch up
                targets = [(j + 2, pu), (j + 1, pm), (j, pd)]

            for jnext, p in targets:
                jnext = max(-jmax - 1, min(jmax + 1, jnext))
                Q[i + 1][jnext] = Q[i + 1].get(jnext, 0.0) + qval * disc * p

    # Set alpha at last step (not used for discounting but needed for short-rate)
    t_last = n_steps * dt
    P_last = float(curve.df(t_last)) * math.exp(-oas * t_last)
    sum_q = sum(Q[n_steps].values())
    if sum_q > 1e-14:
        alpha[n_steps] = -math.log(P_last / (sum_q * dt + 1e-14)) / dt
    else:
        alpha[n_steps] = alpha[n_steps - 1]

    return alpha, Q


# ── Bond pricing ───────────────────────────────────────────────────────────────

def straight_bond_price(curve: DiscountCurve, bond: CallableBond) -> float:
    """
    Analytical price of the straight (non-callable/non-putable) bond.

    P = coupon * sum_k DF(T_k) + face * DF(T_N)
    """
    cpn = bond.coupon_payment
    pv = sum(cpn * float(curve.df(t)) for t in bond.coupon_times)
    pv += bond.face * float(curve.df(bond.maturity))
    return pv


def _price_on_tree(
    curve:    DiscountCurve,
    params:   HWTreeParams,
    bond:     CallableBond,
    oas:      float = 0.0,
) -> float:
    """
    Backward induction on calibrated HW tree to price the callable/putable bond.
    """
    dt    = params.dt
    dr    = params.dr
    M     = params.a * dt
    jmax  = params.j_max

    # Build time grid covering bond maturity
    n_steps = max(1, round(bond.maturity / dt))
    times   = np.array([i * dt for i in range(n_steps + 1)])

    alpha, Q = _build_tree(curve, params, n_steps, oas)

    # Identify coupon, call, put step indices (nearest grid point)
    cpn_times  = bond.coupon_times
    call_dict  = {t: p for t, p in bond.call_schedule}
    put_dict   = {t: p for t, p in bond.put_schedule}

    def nearest_step(t: float) -> int:
        return int(min(range(n_steps + 1), key=lambda i: abs(times[i] - t)))

    cpn_steps  = {nearest_step(t): bond.coupon_payment for t in cpn_times}
    call_steps = {nearest_step(t): p for t, p in bond.call_schedule}
    put_steps  = {nearest_step(t): p for t, p in bond.put_schedule}

    # Bond value at each node at maturity
    V: dict[int, float] = {}
    for j in Q[n_steps]:
        V[j] = bond.face
        if n_steps in cpn_steps:
            V[j] += cpn_steps[n_steps]

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        V_new: dict[int, float] = {}
        for j in Q[i]:
            j = int(j)
            r_j = alpha[i] + j * dr
            disc = math.exp(-r_j * dt)
            shift, pu, pm, pd = _branching_probs(j, jmax, M)

            if shift == 0:
                neighbors = [(j + 1, pu), (j, pm), (j - 1, pd)]
            elif shift == 1:
                neighbors = [(j, pu), (j - 1, pm), (j - 2, pd)]
            else:
                neighbors = [(j + 2, pu), (j + 1, pm), (j, pd)]

            ev = 0.0
            for jnext, p in neighbors:
                jnext = max(-jmax - 1, min(jmax + 1, jnext))
                ev += p * V.get(jnext, bond.face)

            v = disc * ev

            # Add coupon at this step
            if i in cpn_steps:
                v += disc * cpn_steps[i]

            # Apply call constraint (issuer calls if bond value > call price)
            if i in call_steps:
                v = min(v, call_steps[i])

            # Apply put constraint (holder puts if bond value < put price)
            if i in put_steps:
                v = max(v, put_steps[i])

            V_new[j] = v
        V = V_new

    # Price at root (j=0)
    return V.get(0, bond.face)


def price_callable_bond(
    curve:     DiscountCurve,
    hw_params: HWTreeParams,
    bond:      CallableBond,
    oas:       float = 0.0,
) -> float:
    """Price a callable/putable bond on the HW trinomial tree with given OAS."""
    return _price_on_tree(curve, hw_params, bond, oas)


def calibrate_oas(
    curve:        DiscountCurve,
    hw_params:    HWTreeParams,
    bond:         CallableBond,
    market_price: float,
    oas_lo:       float = -0.10,
    oas_hi:       float = 0.10,
) -> float:
    """
    Calibrate the option-adjusted spread so that the model price matches market_price.

    Returns OAS in decimal (e.g. 0.005 = 50 bps).
    """
    def objective(oas: float) -> float:
        return _price_on_tree(curve, hw_params, bond, oas) - market_price

    f_lo = objective(oas_lo)
    f_hi = objective(oas_hi)
    if f_lo * f_hi > 0:
        # Expand search
        for oas_hi2 in [0.20, 0.50, 1.00]:
            if objective(oas_hi2) * f_lo < 0:
                oas_hi = oas_hi2
                break
        for oas_lo2 in [-0.20, -0.50, -1.00]:
            if objective(oas_lo2) * objective(oas_hi) < 0:
                oas_lo = oas_lo2
                break

    return brentq(objective, oas_lo, oas_hi, xtol=1e-8, maxiter=200)


def _shifted_curve(curve: DiscountCurve, shift: float) -> DiscountCurve:
    """Return a new DiscountCurve with all zero rates shifted by `shift`."""
    times = curve._times[1:]          # skip t=0 (DF=1 by convention)
    log_dfs = curve._log_df[1:]       # log(DF) at each pillar
    # zero rate r(t) = -log(DF(t))/t; shifted: r_new = r + shift → DF_new = exp(-(r+shift)*t)
    new_log_dfs = log_dfs - shift * times
    new_dfs = np.exp(new_log_dfs)
    return DiscountCurve(
        ref_date=curve.ref_date,
        times=times,
        dfs=new_dfs,
        label=curve.label,
    )


def effective_duration(
    curve:     DiscountCurve,
    hw_params: HWTreeParams,
    bond:      CallableBond,
    oas:       float = 0.0,
    dy:        float = 0.0001,
) -> float:
    """
    Effective duration via ±dy parallel curve shift.

    ED = (P(-dy) - P(+dy)) / (2 * P * dy)
    """
    p0   = _price_on_tree(curve,               hw_params, bond, oas)
    p_dn = _price_on_tree(_shifted_curve(curve, -dy), hw_params, bond, oas)
    p_up = _price_on_tree(_shifted_curve(curve, +dy), hw_params, bond, oas)

    return (p_dn - p_up) / (2.0 * p0 * dy)


def effective_convexity(
    curve:     DiscountCurve,
    hw_params: HWTreeParams,
    bond:      CallableBond,
    oas:       float = 0.0,
    dy:        float = 0.0001,
) -> float:
    """
    Effective convexity via ±dy parallel curve shift.

    EC = (P(-dy) + P(+dy) - 2P) / (P * dy^2)
    """
    p0   = _price_on_tree(curve,               hw_params, bond, oas)
    p_dn = _price_on_tree(_shifted_curve(curve, -dy), hw_params, bond, oas)
    p_up = _price_on_tree(_shifted_curve(curve, +dy), hw_params, bond, oas)

    return (p_dn + p_up - 2.0 * p0) / (p0 * dy * dy)


def price_callable_bond_full(
    curve:        DiscountCurve,
    hw_params:    HWTreeParams,
    bond:         CallableBond,
    market_price: float | None = None,
) -> CallableBondResult:
    """
    Full callable bond analysis: price, OAS, straight price, effective duration/convexity.

    If market_price is provided, calibrates OAS; otherwise OAS=0.
    """
    straight = straight_bond_price(curve, bond)
    oas = 0.0
    if market_price is not None:
        oas = calibrate_oas(curve, hw_params, bond, market_price)
        price = market_price
    else:
        price = price_callable_bond(curve, hw_params, bond, oas)

    option_value = abs(straight - price)
    ed = effective_duration(curve, hw_params, bond, oas)
    ec = effective_convexity(curve, hw_params, bond, oas)

    return CallableBondResult(
        price=price,
        straight_price=straight,
        option_value=option_value,
        oas=oas,
        effective_duration=ed,
        effective_convexity=ec,
    )


__all__ = [
    "HWTreeParams",
    "CallableBond",
    "CallableBondResult",
    "price_callable_bond",
    "price_callable_bond_full",
    "straight_bond_price",
    "calibrate_oas",
    "effective_duration",
    "effective_convexity",
]

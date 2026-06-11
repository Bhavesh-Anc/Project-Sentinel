"""
Bermudan Swaption Pricing — Longstaff-Schwartz Monte Carlo (LSM)
================================================================
Prices Bermudan (multi-exercise) swaptions by backward induction on
Monte Carlo paths using the Longstaff-Schwartz (2001) least-squares
regression algorithm.

Background
----------
A Bermudan swaption allows the holder to enter a fixed/floating swap at
ANY of a set of discrete exercise dates (typically quarterly or semi-annual
from the first exercise to the swap maturity).

This is more valuable than a European swaption (single exercise) and cannot
be priced analytically — Monte Carlo with backward induction is the industry
standard approach.

LSM Algorithm
-------------
1. Simulate N paths of the short rate under Hull-White (forward measure).
2. At final exercise date T_n: exercise value = max(swap PV, 0).
3. For each earlier exercise date T_k (backward):
   a. Identify in-the-money (ITM) paths.
   b. On ITM paths, regress the discounted continuation value on a set of
      basis functions of the state variable (r_t or swap PV).
   c. Exercise if immediate exercise value > fitted continuation value.
4. Price = discounted expected value of the optimal exercise payoff.

Regression Basis
----------------
Laguerre polynomials L_0, L_1, L_2, L_3 of the state variable (standardised
short rate) — the same choice as in Longstaff & Schwartz (2001).

The Swap PV on Each Path
------------------------
Under Hull-White, the swap PV at exercise date T_k on path i is:
    SwapPV_i(T_k) = annuity(T_k, swap maturities | r_i) × (S_i(T_k) − K)

where S_i is the fair forward rate and annuity is computed analytically
from simulated r_i via the HW bond pricing formula.

References
----------
Longstaff, F.A. & Schwartz, E.S. (2001). "Valuing American options by
  simulation: A simple least-squares approach." RFS 14(1), 113–147.
Andersen, L. & Piterbarg, V. (2010). Interest Rate Modelling, Vol. II, Ch. 12.
Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering, Ch. 8.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from .curve import DiscountCurve
from .monte_carlo import (
    HullWhiteParams, simulate_hw, SimulationResult,
    zcb_price_hw, _B, _ln_A, _inst_forward,
)


# ── Basis functions ───────────────────────────────────────────────────────────

def _laguerre_basis(x: np.ndarray, n_terms: int = 4) -> np.ndarray:
    """
    Laguerre polynomial basis functions for LSM regression.

    L_0(x) = 1
    L_1(x) = 1 − x
    L_2(x) = 1 − 2x + x²/2
    L_3(x) = 1 − 3x + 3x²/2 − x³/6

    Parameters
    ----------
    x      : state variable array, shape (n_paths,)
    n_terms: number of basis terms (max 4)

    Returns
    -------
    Design matrix of shape (n_paths, n_terms).
    """
    e_x  = np.exp(-x / 2)
    L0   = e_x * np.ones_like(x)
    L1   = e_x * (1 - x)
    L2   = e_x * (1 - 2*x + x**2/2)
    L3   = e_x * (1 - 3*x + 3*x**2/2 - x**3/6)
    bases = [L0, L1, L2, L3]
    return np.column_stack(bases[:n_terms])


# ── Analytical swap PV at exercise date ──────────────────────────────────────

def _swap_pv_at_date(
    r_t: np.ndarray,
    t: float,
    swap_maturity: float,
    strike: float,
    notional: float,
    pay_receive: Literal["payer", "receiver"],
    curve: DiscountCurve,
    params: HullWhiteParams,
    freq: int = 2,
) -> np.ndarray:
    """
    Analytical swap PV at exercise date t for each path.

    SwapPV = notional × annuity × (S_t − K) × sign

    where:
    - annuity = Σ τ_i × P(t, T_i)
    - S_t = [P(t, t) − P(t, T)] / annuity = [1 − P(t, T)] / annuity

    sign = +1 for payer (pay fixed, receive float; benefits when S_t > K)
           −1 for receiver (receive fixed, pay float; benefits when S_t < K)

    Parameters
    ----------
    r_t       : short rate at time t, shape (n_paths,)
    t         : exercise/observation date in years
    swap_maturity : swap end date in years from today
    strike    : fixed rate K
    freq      : coupon frequency per year
    """
    dt = 1.0 / freq
    payment_times = np.arange(t + dt, swap_maturity + dt * 0.5, dt)

    annuity = np.zeros(len(r_t))
    for T_pay in payment_times:
        if T_pay > t + 1e-8:
            P = zcb_price_hw(r_t, t, T_pay, curve, params)
            annuity += dt * P

    if len(payment_times) == 0 or (annuity < 1e-12).all():
        return np.zeros(len(r_t))

    P_T = zcb_price_hw(r_t, t, swap_maturity, curve, params)
    S_t = np.where(annuity > 1e-12, (1.0 - P_T) / annuity, strike)

    sign = 1.0 if pay_receive == "payer" else -1.0
    return float(notional) * annuity * np.maximum(sign * (S_t - strike), 0.0)


# ── LSM Bermudan swaption ────────────────────────────────────────────────────

@dataclass
class BermudanSwaptionResult:
    """Output of a Bermudan swaption pricing by LSM."""
    price:              float           # Bermudan swaption PV ($)
    european_lower:     float           # Max European swaption price (lower bound)
    exercise_probs:     list[float]     # P(optimal exercise at each exercise date)
    exercise_dates:     list[float]     # exercise date schedule (years)
    params:             HullWhiteParams
    n_paths:            int
    strike:             float
    swap_maturity:      float
    pay_receive:        str

    @property
    def early_exercise_premium(self) -> float:
        """Price premium over the best European swaption."""
        return self.price - self.european_lower


def price_bermudan_swaption(
    curve: DiscountCurve,
    params: HullWhiteParams,
    first_exercise: float,
    swap_maturity: float,
    strike: float | None = None,
    notional: float = 1_000_000.0,
    pay_receive: Literal["payer", "receiver"] = "payer",
    exercise_freq: int = 2,
    swap_freq: int = 2,
    n_paths: int = 10_000,
    n_steps_per_period: int = 5,
    n_basis: int = 3,
    seed: int = 42,
) -> BermudanSwaptionResult:
    """
    Price a Bermudan swaption using Longstaff-Schwartz LSM Monte Carlo.

    The holder may exercise at any of the dates
    {first_exercise, first_exercise + 1/exercise_freq, ..., swap_maturity − 1/exercise_freq}

    If strike is None, uses ATM (annuity-weighted forward swap rate from initial curve).

    Parameters
    ----------
    curve              : initial SOFR discount curve
    params             : Hull-White model parameters
    first_exercise     : first allowable exercise date in years
    swap_maturity      : swap end date in years
    strike             : fixed coupon rate; None = ATM
    notional           : face value
    pay_receive        : 'payer' (call on rates) or 'receiver' (put on rates)
    exercise_freq      : exercise dates per year (2 = semi-annual)
    swap_freq          : swap coupon frequency per year
    n_paths            : number of MC paths
    n_steps_per_period : simulation steps between exercise dates
    n_basis            : number of Laguerre basis terms for regression
    seed               : random seed

    Returns
    -------
    BermudanSwaptionResult with price, exercise_probs, and early exercise premium.
    """
    dt_exercise = 1.0 / exercise_freq
    exercise_dates = np.arange(first_exercise, swap_maturity - dt_exercise * 0.5,
                                dt_exercise)
    exercise_dates = exercise_dates[exercise_dates < swap_maturity - 1e-6]

    if len(exercise_dates) == 0:
        raise ValueError("No valid exercise dates between first_exercise and swap_maturity")

    # ATM strike
    if strike is None:
        dt_s = 1.0 / swap_freq
        pay_times = np.arange(dt_s, swap_maturity + dt_s * 0.5, dt_s)
        annuity0  = sum(dt_s * float(curve.df(T)) for T in pay_times if T > 1e-8)
        strike    = (1.0 - float(curve.df(swap_maturity))) / annuity0 if annuity0 > 1e-12 else 0.05

    # Build fine time grid covering all exercise dates
    total_horizon = exercise_dates[-1]
    n_total_steps = len(exercise_dates) * n_steps_per_period

    sim = simulate_hw(
        curve, params,
        horizon = total_horizon,
        n_steps = n_total_steps,
        n_paths = n_paths,
        seed    = seed,
        antithetic = True,
    )
    n_paths_actual = sim.n_paths

    # Map exercise dates to time steps
    dt_step  = total_horizon / n_total_steps
    ex_steps = [max(0, min(n_total_steps - 1,
                           int(round(t / dt_step)))) for t in exercise_dates]

    # Backward induction: start from last exercise date
    # continuation_value[i] = discounted continuation value for path i
    continuation = np.zeros(n_paths_actual)

    exercise_probs = []
    european_prices = []

    for k in range(len(exercise_dates) - 1, -1, -1):
        t_k   = float(exercise_dates[k])
        step_k = ex_steps[k]

        r_k      = sim.r_paths[:, step_k]
        iv_k     = _swap_pv_at_date(
            r_k, t_k, swap_maturity, strike, notional,
            pay_receive, curve, params, swap_freq,
        )

        # Discount continuation from next exercise to current (over dt_exercise)
        # Use midpoint of short rates between step_k and step_k + n_steps_per_period
        if k < len(exercise_dates) - 1:
            step_next = ex_steps[k + 1]
            r_seg = sim.r_paths[:, step_k:step_next + 1]
            n_seg = step_next - step_k
            dt_seg = (exercise_dates[k + 1] - t_k) / max(n_seg, 1)
            r_mid_seg = 0.5 * (r_seg[:, :-1] + r_seg[:, 1:])
            disc_seg  = np.exp(-r_mid_seg.sum(axis=1) * dt_seg)
            continuation = continuation * disc_seg   # discount earlier continuation

        # LSM regression on ITM paths
        itm = iv_k > 0

        if itm.sum() > n_basis + 1:
            x_state = r_k[itm]
            x_mean, x_std = x_state.mean(), x_state.std()
            if x_std < 1e-10:
                x_std = 1.0
            x_norm  = (x_state - x_mean) / x_std
            basis   = _laguerre_basis(x_norm, n_basis)
            cont_itm = continuation[itm]

            try:
                coeffs, _, _, _ = np.linalg.lstsq(basis, cont_itm, rcond=None)
                cv_fitted = basis @ coeffs
            except np.linalg.LinAlgError:
                cv_fitted = cont_itm

            # Exercise decision: immediate value > fitted continuation
            exercise_now = iv_k[itm] > cv_fitted
            continuation[itm] = np.where(exercise_now, iv_k[itm], continuation[itm])
        else:
            # Too few ITM paths: always exercise if ITM
            continuation[itm] = iv_k[itm]

        # Track European price at this exercise date (discount back to t=0)
        dt_sim = total_horizon / n_total_steps
        r_full = sim.r_paths[:, :step_k + 1]
        r_mid_full = 0.5 * (r_full[:, :-1] + r_full[:, 1:])
        disc_0k  = np.exp(-r_mid_full.sum(axis=1) * dt_sim)
        euro_pv  = float(np.mean(disc_0k * iv_k))
        european_prices.append(euro_pv)

        ex_prob = float(itm.sum() / n_paths_actual)
        exercise_probs.append(ex_prob)

    # Final Bermudan price: discount total continuation back to t=0
    first_step = ex_steps[0]
    r_0_to_first = sim.r_paths[:, :first_step + 1]
    r_mid_0 = 0.5 * (r_0_to_first[:, :-1] + r_0_to_first[:, 1:])
    dt_sim   = total_horizon / n_total_steps
    disc_0   = np.exp(-r_mid_0.sum(axis=1) * dt_sim)
    bermudan_price = float(np.mean(disc_0 * continuation))

    return BermudanSwaptionResult(
        price           = max(bermudan_price, 0.0),
        european_lower  = max(european_prices),
        exercise_probs  = list(reversed(exercise_probs)),
        exercise_dates  = list(exercise_dates),
        params          = params,
        n_paths         = n_paths_actual,
        strike          = strike,
        swap_maturity   = swap_maturity,
        pay_receive     = pay_receive,
    )


# ── European swaption by HW ──────────────────────────────────────────────────

def price_european_swaption_hw(
    curve: DiscountCurve,
    params: HullWhiteParams,
    expiry: float,
    swap_maturity: float,
    strike: float | None = None,
    notional: float = 1_000_000.0,
    pay_receive: Literal["payer", "receiver"] = "payer",
    swap_freq: int = 2,
    n_paths: int = 20_000,
    seed: int = 42,
) -> dict:
    """
    Price a European swaption by Hull-White Monte Carlo.

    Provides a bridge between the analytical Black-76 and the full Bermudan LSM.
    Useful for convergence and calibration checks.

    Returns
    -------
    dict with pv, forward_swap_rate, annuity, atm_strike, n_paths.
    """
    if strike is None:
        dt_s = 1.0 / swap_freq
        pay_times = np.arange(expiry + dt_s, swap_maturity + dt_s * 0.5, dt_s)
        annuity0  = sum(dt_s * float(curve.df(T)) for T in pay_times if T > expiry - 1e-8)
        P_expiry = float(curve.df(expiry))
        P_mat    = float(curve.df(swap_maturity))
        strike   = (P_expiry - P_mat) / annuity0 if annuity0 > 1e-12 else 0.05

    sim    = simulate_hw(curve, params, expiry, n_steps=50, n_paths=n_paths, seed=seed, antithetic=True)
    h_step = sim.n_steps
    r_exp  = sim.r_paths[:, h_step]

    # Payoff at expiry
    iv = _swap_pv_at_date(r_exp, expiry, swap_maturity, strike, notional,
                           pay_receive, curve, params, swap_freq)

    # Discount path from 0 to expiry
    dt = expiry / sim.n_steps
    r_mid = 0.5 * (sim.r_paths[:, :-1] + sim.r_paths[:, 1:])
    disc  = np.exp(-r_mid.sum(axis=1) * dt)
    pv    = float(np.mean(disc * iv))
    se    = float(np.std(disc * iv) / np.sqrt(sim.n_paths))

    # ATM forward swap rate (from initial curve)
    dt_s = 1.0 / swap_freq
    pay_times = np.arange(expiry + dt_s, swap_maturity + dt_s * 0.5, dt_s)
    annuity_init = sum(dt_s * float(curve.df(T)) for T in pay_times)
    P_0T_exp = float(curve.df(expiry))
    P_0T_mat = float(curve.df(swap_maturity))
    fwd_rate = (P_0T_exp - P_0T_mat) / annuity_init if annuity_init > 1e-12 else 0.05

    return {
        "pv":                  round(max(pv, 0.0), 2),
        "mc_stderr":           round(se, 2),
        "mc_95ci":             (round(pv - 1.96 * se, 2), round(pv + 1.96 * se, 2)),
        "forward_swap_rate_pct": round(fwd_rate * 100, 5),
        "strike_pct":          round(strike * 100, 5),
        "moneyness_bps":       round((fwd_rate - strike) * 10_000, 2),
        "annuity":             round(annuity_init, 8),
        "n_paths":             sim.n_paths,
        "expiry":              expiry,
        "swap_maturity":       swap_maturity,
        "pay_receive":         pay_receive,
    }

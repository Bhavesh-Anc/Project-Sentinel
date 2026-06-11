"""
Tests for sofr_engine/g2pp.py

Coverage
--------
- G2ppParams: validation
- g2pp_zcb: analytical ZCB, t→T=0, recovery of initial curve
- g2pp_inst_forward: consistency with curve forwards
- simulate_g2pp: shapes, mean-reversion, antithetic, determinism
- G2ppSimResult.zcb_prices_at: shape, positivity
- g2pp_swaption_mc: positive PV, key fields, ATM moneyness, ITM > ATM
- g2pp_portfolio_var: VaR negative, CVaR ≤ VaR, key fields
- g2pp_swaption: analytical swaption returns positive PV
"""
from __future__ import annotations

import math
import pytest
import numpy as np
from datetime import date

from sofr_engine.bootstrap import flat_sofr_curve
from sofr_engine.g2pp import (
    G2ppParams, G2ppSimResult,
    g2pp_zcb, g2pp_inst_forward,
    simulate_g2pp,
    g2pp_swaption, g2pp_swaption_mc,
    g2pp_portfolio_var,
    _B, _ln_A_g2pp, _g2pp_phi,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_curve():
    return flat_sofr_curve(date.today(), 0.0433)


@pytest.fixture
def params():
    return G2ppParams(a=0.05, b=0.10, sigma=0.010, eta=0.008, rho=-0.30)


@pytest.fixture
def small_params():
    return G2ppParams(a=0.05, b=0.10, sigma=0.005, eta=0.004, rho=0.0)


# ── G2ppParams validation ─────────────────────────────────────────────────────

class TestG2ppParams:
    def test_valid_construction(self):
        p = G2ppParams(a=0.05, b=0.10, sigma=0.010, eta=0.008, rho=-0.30)
        assert p.a == 0.05

    def test_negative_a_raises(self):
        with pytest.raises(ValueError):
            G2ppParams(a=-0.05, b=0.10, sigma=0.010, eta=0.008, rho=-0.30)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            G2ppParams(a=0.05, b=0.10, sigma=0.0, eta=0.008, rho=-0.30)

    def test_rho_out_of_range_raises(self):
        with pytest.raises(ValueError):
            G2ppParams(a=0.05, b=0.10, sigma=0.010, eta=0.008, rho=1.0)

    def test_rho_minus_one_raises(self):
        with pytest.raises(ValueError):
            G2ppParams(a=0.05, b=0.10, sigma=0.010, eta=0.008, rho=-1.0)

    def test_zero_b_raises(self):
        with pytest.raises(ValueError):
            G2ppParams(a=0.05, b=0.0, sigma=0.010, eta=0.008, rho=-0.30)


# ── B factor ─────────────────────────────────────────────────────────────────

class TestBFactor:
    def test_zero_at_T_equals_t(self):
        assert abs(_B(0.05, 1.0, 1.0)) < 1e-10

    def test_positive_T_greater_t(self):
        assert _B(0.05, 1.0, 5.0) > 0

    def test_approaches_1_over_kappa_for_large_tau(self):
        # As T-t → ∞, B(κ,t,T) → 1/κ  (for κ=0.05, need T very large)
        assert abs(_B(0.05, 0.0, 1000.0) - 1.0 / 0.05) < 0.001

    def test_small_tau(self):
        # B(κ, t, t+ε) ≈ ε (first-order Taylor)
        eps = 1e-4
        assert abs(_B(0.05, 0.0, eps) - eps) < 1e-5


# ── ZCB pricing ───────────────────────────────────────────────────────────────

class TestG2ppZCB:
    def test_boundary_t_equals_T(self, flat_curve, params):
        p = g2pp_zcb(flat_curve, params, t=1.0, T=1.0, x_t=0.01, y_t=0.01)
        assert abs(p - 1.0) < 1e-10

    def test_boundary_t_greater_T(self, flat_curve, params):
        p = g2pp_zcb(flat_curve, params, t=5.0, T=1.0, x_t=0.0, y_t=0.0)
        assert p == 0.0

    def test_recovers_initial_curve_at_x0_y0(self, flat_curve, params):
        # At t→0, x=0, y=0: P(t,T;0,0) → curve.df(T)
        T = 5.0
        p_hw = g2pp_zcb(flat_curve, params, t=0.0, T=T, x_t=0.0, y_t=0.0)
        p_curve = float(flat_curve.df(T))
        assert abs(p_hw / p_curve - 1.0) < 1e-6

    def test_monotone_in_maturity(self, flat_curve, params):
        x_t, y_t = 0.0, 0.0
        T_vals = [1.0, 2.0, 5.0, 10.0]
        prices = [g2pp_zcb(flat_curve, params, 0.0, T, x_t, y_t) for T in T_vals]
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1))

    def test_positive_value(self, flat_curve, params):
        p = g2pp_zcb(flat_curve, params, 0.0, 5.0, 0.01, 0.01)
        assert 0 < p < 1

    def test_higher_x_lower_price(self, flat_curve, params):
        p_low  = g2pp_zcb(flat_curve, params, 1.0, 5.0,  0.0, 0.0)
        p_high = g2pp_zcb(flat_curve, params, 1.0, 5.0,  0.10, 0.0)
        assert p_low > p_high


# ── Instantaneous forward ─────────────────────────────────────────────────────

class TestG2ppInstForward:
    def test_positive_for_normal_curve(self, flat_curve):
        f = g2pp_inst_forward(flat_curve, 1.0)
        assert f > 0

    def test_near_flat_for_flat_curve(self, flat_curve):
        # Flat 4.33% curve → f(0,t) ≈ 4.33% for all t
        f1 = g2pp_inst_forward(flat_curve, 1.0)
        f5 = g2pp_inst_forward(flat_curve, 5.0)
        assert abs(f1 - 0.0433) < 0.005
        assert abs(f5 - 0.0433) < 0.005

    def test_consistent_finite_difference(self, flat_curve):
        dt = 1e-3
        t  = 2.0
        fd = -(math.log(float(flat_curve.df(t + dt))) - math.log(float(flat_curve.df(t)))) / dt
        f  = g2pp_inst_forward(flat_curve, t, dt=dt/2)
        assert abs(f - fd) < 0.001


# ── Simulation ────────────────────────────────────────────────────────────────

class TestSimulateG2pp:
    def test_output_shape(self, flat_curve, params):
        sim = simulate_g2pp(flat_curve, params, horizon=1.0, n_steps=10, n_paths=100, seed=42)
        assert sim.x_paths.shape == (100, 11)
        assert sim.y_paths.shape == (100, 11)
        assert sim.r_paths.shape == (100, 11)

    def test_starts_at_zero(self, flat_curve, params):
        sim = simulate_g2pp(flat_curve, params, horizon=1.0, n_steps=10, n_paths=200, seed=42)
        np.testing.assert_allclose(sim.x_paths[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(sim.y_paths[:, 0], 0.0, atol=1e-12)

    def test_antithetic_symmetry(self, flat_curve, params):
        n = 200
        sim = simulate_g2pp(flat_curve, params, horizon=1.0, n_steps=5, n_paths=n, seed=42)
        half = n // 2
        np.testing.assert_allclose(sim.x_paths[:half], -sim.x_paths[half:], atol=1e-10)
        np.testing.assert_allclose(sim.y_paths[:half], -sim.y_paths[half:], atol=1e-10)

    def test_deterministic_with_seed(self, flat_curve, params):
        s1 = simulate_g2pp(flat_curve, params, horizon=1.0, n_steps=5, n_paths=50, seed=99)
        s2 = simulate_g2pp(flat_curve, params, horizon=1.0, n_steps=5, n_paths=50, seed=99)
        np.testing.assert_array_equal(s1.x_paths, s2.x_paths)

    def test_different_seeds_differ(self, flat_curve, params):
        s1 = simulate_g2pp(flat_curve, params, horizon=1.0, n_steps=5, n_paths=50, seed=1)
        s2 = simulate_g2pp(flat_curve, params, horizon=1.0, n_steps=5, n_paths=50, seed=2)
        assert not np.allclose(s1.x_paths, s2.x_paths)

    def test_short_rate_near_initial(self, flat_curve, small_params):
        # With low vol, r should stay near initial forward rate
        sim = simulate_g2pp(flat_curve, small_params, horizon=0.01,
                            n_steps=5, n_paths=1000, seed=42)
        r_end = sim.r_paths[:, -1]
        f0    = g2pp_inst_forward(flat_curve, 0.01)
        assert abs(float(np.mean(r_end)) - f0) < 0.005

    def test_t_grid_correct(self, flat_curve, params):
        sim = simulate_g2pp(flat_curve, params, horizon=2.0, n_steps=4, n_paths=10, seed=42)
        expected = np.linspace(0.0, 2.0, 5)
        np.testing.assert_allclose(sim.t_grid, expected)

    def test_r_paths_reflect_phi(self, flat_curve, params):
        # r = x + y + phi; at t=0, x=y=0 so r[0] = phi(0)
        sim = simulate_g2pp(flat_curve, params, horizon=1.0, n_steps=10, n_paths=50, seed=1)
        phi0 = _g2pp_phi(flat_curve, params, 0.0)
        np.testing.assert_allclose(sim.r_paths[:, 0], phi0, atol=1e-10)

    def test_zcb_prices_at_shape(self, flat_curve, params):
        sim = simulate_g2pp(flat_curve, params, horizon=1.0, n_steps=5, n_paths=50, seed=1)
        mats = [2.0, 5.0, 10.0]
        prices = sim.zcb_prices_at(step=5, maturities=mats)
        assert prices.shape == (50, 3)

    def test_zcb_prices_positive(self, flat_curve, params):
        sim = simulate_g2pp(flat_curve, params, horizon=1.0, n_steps=5, n_paths=50, seed=1)
        prices = sim.zcb_prices_at(step=5, maturities=[5.0])
        assert (prices > 0).all()


# ── Monte Carlo Swaption ──────────────────────────────────────────────────────

class TestG2ppSwaptionMC:
    def test_pv_positive(self, flat_curve, params):
        res = g2pp_swaption_mc(flat_curve, params, expiry=1.0, swap_tenor=5.0, n_paths=5000, seed=42)
        assert res["pv"] > 0

    def test_keys_present(self, flat_curve, params):
        res = g2pp_swaption_mc(flat_curve, params, expiry=1.0, swap_tenor=5.0, n_paths=1000)
        for k in ["pv", "mc_stderr", "forward_swap_rate_pct", "strike_pct", "annuity", "n_paths"]:
            assert k in res

    def test_atm_strike_moneyness_zero(self, flat_curve, params):
        res = g2pp_swaption_mc(flat_curve, params, expiry=1.0, swap_tenor=5.0,
                                strike=None, n_paths=2000)
        assert abs(res["forward_swap_rate_pct"] - res["strike_pct"]) < 0.01

    def test_itm_greater_than_atm(self, flat_curve, params):
        atm = g2pp_swaption_mc(flat_curve, params, expiry=1.0, swap_tenor=5.0,
                                strike=None, n_paths=5000, seed=42)
        itm = g2pp_swaption_mc(flat_curve, params, expiry=1.0, swap_tenor=5.0,
                                strike=0.01, n_paths=5000, seed=42)
        assert itm["pv"] > atm["pv"]

    def test_receiver_positive(self, flat_curve, params):
        res = g2pp_swaption_mc(flat_curve, params, expiry=1.0, swap_tenor=5.0,
                                pay_receive="receiver", n_paths=3000, seed=42)
        assert res["pv"] > 0

    def test_mc_stderr_positive(self, flat_curve, params):
        res = g2pp_swaption_mc(flat_curve, params, expiry=1.0, swap_tenor=5.0, n_paths=1000)
        assert res["mc_stderr"] > 0

    def test_longer_tenor_larger_annuity(self, flat_curve, params):
        r5  = g2pp_swaption_mc(flat_curve, params, expiry=1.0, swap_tenor=5.0,  n_paths=500)
        r10 = g2pp_swaption_mc(flat_curve, params, expiry=1.0, swap_tenor=10.0, n_paths=500)
        assert r10["annuity"] > r5["annuity"]

    def test_deterministic_with_seed(self, flat_curve, params):
        r1 = g2pp_swaption_mc(flat_curve, params, expiry=1.0, swap_tenor=5.0, n_paths=1000, seed=77)
        r2 = g2pp_swaption_mc(flat_curve, params, expiry=1.0, swap_tenor=5.0, n_paths=1000, seed=77)
        assert abs(r1["pv"] - r2["pv"]) < 1e-8

    def test_higher_vol_higher_pv(self, flat_curve):
        p_low  = G2ppParams(a=0.05, b=0.10, sigma=0.005, eta=0.004, rho=0.0)
        p_high = G2ppParams(a=0.05, b=0.10, sigma=0.020, eta=0.016, rho=0.0)
        r_low  = g2pp_swaption_mc(flat_curve, p_low,  1.0, 5.0, n_paths=5000, seed=42)
        r_high = g2pp_swaption_mc(flat_curve, p_high, 1.0, 5.0, n_paths=5000, seed=42)
        assert r_high["pv"] > r_low["pv"]

    def test_pv_scales_with_notional(self, flat_curve, params):
        r1 = g2pp_swaption_mc(flat_curve, params, expiry=1.0, swap_tenor=5.0,
                               notional=1_000_000.0, n_paths=2000, seed=1)
        r2 = g2pp_swaption_mc(flat_curve, params, expiry=1.0, swap_tenor=5.0,
                               notional=2_000_000.0, n_paths=2000, seed=1)
        assert r1["pv"] > 0
        assert abs(r2["pv"] / r1["pv"] - 2.0) < 1e-8


# ── Portfolio VaR ─────────────────────────────────────────────────────────────

class TestG2ppVaR:
    def test_var_negative_long_duration(self, flat_curve, params):
        res = g2pp_portfolio_var(flat_curve, params, portfolio_dv01=10_000,
                                  horizon=1/250, confidence=0.99, n_paths=2000)
        assert res["var_usd"] < 0

    def test_cvar_le_var(self, flat_curve, params):
        res = g2pp_portfolio_var(flat_curve, params, portfolio_dv01=10_000,
                                  horizon=1/250, confidence=0.99, n_paths=2000)
        assert res["cvar_usd"] <= res["var_usd"]

    def test_result_keys(self, flat_curve, params):
        res = g2pp_portfolio_var(flat_curve, params, portfolio_dv01=10_000,
                                  horizon=1/250, confidence=0.99, n_paths=1000)
        for k in ["var_usd", "cvar_usd", "var_bps", "pnl_mean", "pnl_std",
                  "n_paths", "confidence"]:
            assert k in res

    def test_longer_horizon_wider_var(self, flat_curve, params):
        r1d = g2pp_portfolio_var(flat_curve, params, portfolio_dv01=10_000,
                                  horizon=1/250, confidence=0.99, n_paths=3000, seed=42)
        r10d = g2pp_portfolio_var(flat_curve, params, portfolio_dv01=10_000,
                                   horizon=10/250, confidence=0.99, n_paths=3000, seed=42)
        assert abs(r10d["var_usd"]) > abs(r1d["var_usd"])

    def test_higher_dv01_wider_var(self, flat_curve, params):
        r1 = g2pp_portfolio_var(flat_curve, params, portfolio_dv01=5_000,
                                 horizon=1/250, n_paths=2000, seed=42)
        r2 = g2pp_portfolio_var(flat_curve, params, portfolio_dv01=10_000,
                                 horizon=1/250, n_paths=2000, seed=42)
        assert abs(r2["var_usd"]) > abs(r1["var_usd"])


# ── Analytical swaption ───────────────────────────────────────────────────────

class TestG2ppSwaptionAnalytical:
    def test_pv_positive(self, flat_curve, params):
        res = g2pp_swaption(flat_curve, params, expiry=1.0, swap_tenor=5.0, n_quad=32)
        assert res["pv"] >= 0

    def test_keys_present(self, flat_curve, params):
        res = g2pp_swaption(flat_curve, params, expiry=1.0, swap_tenor=5.0, n_quad=32)
        for k in ["pv", "forward_swap_rate_pct", "strike_pct", "annuity", "n_quad"]:
            assert k in res

    def test_annuity_positive(self, flat_curve, params):
        res = g2pp_swaption(flat_curve, params, expiry=1.0, swap_tenor=5.0, n_quad=32)
        assert res["annuity"] > 0

    def test_forward_rate_reasonable(self, flat_curve, params):
        res = g2pp_swaption(flat_curve, params, expiry=1.0, swap_tenor=5.0, n_quad=32)
        assert 0.5 < res["forward_swap_rate_pct"] < 15.0

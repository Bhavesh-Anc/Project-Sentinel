"""
Tests for sofr_engine/monte_carlo.py

Coverage:
- HullWhiteParams validation
- _inst_forward, _phi, _B, _ln_A helpers
- simulate_hw: shapes, exact moments, antithetic
- zcb_price_hw: vs. initial curve (t=0 limit)
- price_zcb_mc: MC vs analytical, convergence direction
- price_caplet_mc: positive price, forward rate check
- portfolio_var_hw: negative VaR (loss), confidence relationship
- parametric_var: formula check
- convergence_diagnostics: error decreases with more paths
- SimulationResult: expected_path, zcb_prices_at_horizon
"""
import math
import pytest
import numpy as np
from datetime import date

from sofr_engine.bootstrap import flat_sofr_curve
from sofr_engine.monte_carlo import (
    HullWhiteParams,
    SimulationResult,
    simulate_hw,
    zcb_price_hw,
    price_zcb_mc,
    price_caplet_mc,
    portfolio_var_hw,
    parametric_var,
    convergence_diagnostics,
    _inst_forward,
    _phi,
    _B,
    _ln_A,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_curve():
    return flat_sofr_curve(date.today(), 0.0433)


@pytest.fixture
def hw_params():
    return HullWhiteParams(a=0.05, sigma=0.010)


@pytest.fixture
def hw_zero_mr():
    """Zero mean-reversion (a → 0) for simpler formulas."""
    return HullWhiteParams(a=1e-12, sigma=0.010)


# ── HullWhiteParams ───────────────────────────────────────────────────────────

class TestHullWhiteParams:
    def test_default_construction(self):
        p = HullWhiteParams()
        assert p.a == 0.05
        assert p.sigma == 0.010

    def test_custom_values(self):
        p = HullWhiteParams(a=0.10, sigma=0.015)
        assert p.a == 0.10
        assert p.sigma == 0.015

    def test_negative_a_raises(self):
        with pytest.raises(ValueError, match="mean-reversion"):
            HullWhiteParams(a=-0.01, sigma=0.010)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            HullWhiteParams(a=0.05, sigma=0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            HullWhiteParams(a=0.05, sigma=-0.01)

    def test_zero_a_allowed(self):
        p = HullWhiteParams(a=0.0, sigma=0.010)
        assert p.a == 0.0


# ── Helper functions ──────────────────────────────────────────────────────────

class TestHelperFunctions:
    def test_inst_forward_near_sofr_level(self, flat_curve, hw_params):
        f = _inst_forward(flat_curve, 1.0)
        assert 0.03 < f < 0.06

    def test_inst_forward_increases_on_upward_curve(self):
        from sofr_engine.curve import DiscountCurve
        times = np.array([0.01, 1.0, 2.0, 5.0, 10.0])
        rates = np.array([0.03, 0.035, 0.04, 0.045, 0.05])
        dfs   = np.exp(-rates * times)
        curve = DiscountCurve(date.today(), times, dfs)
        f1 = _inst_forward(curve, 1.0)
        f5 = _inst_forward(curve, 5.0)
        assert f5 > f1

    def test_phi_near_forward(self, flat_curve, hw_params):
        phi = _phi(flat_curve, 2.0, hw_params)
        f   = _inst_forward(flat_curve, 2.0)
        assert phi >= f  # φ(t) = f(0,t) + non-negative correction

    def test_phi_zero_mr_equals_forward(self, flat_curve, hw_zero_mr):
        phi = _phi(flat_curve, 2.0, hw_zero_mr)
        f   = _inst_forward(flat_curve, 2.0)
        assert abs(phi - f) < 1e-6

    def test_B_monotone_in_tenor(self, hw_params):
        B1 = _B(0, 1.0, hw_params)
        B5 = _B(0, 5.0, hw_params)
        assert B5 > B1

    def test_B_zero_at_t_equal_T(self, hw_params):
        assert abs(_B(2.0, 2.0, hw_params)) < 1e-10

    def test_B_zero_mr_equals_tau(self, hw_zero_mr):
        assert abs(_B(0, 5.0, hw_zero_mr) - 5.0) < 1e-5

    def test_lnA_zero_at_same_time(self, flat_curve, hw_params):
        lnA = _ln_A(1.0, 1.0, flat_curve, hw_params)
        assert abs(lnA) < 1e-8

    def test_lnA_reproduces_initial_curve_at_small_t(self, flat_curve, hw_params):
        # At a very small t, P(t,T) should be close to P(0,T).
        # Use r_t = φ(t) (the initial-curve-fitting value, so x_t=0).
        T = 5.0
        t = 0.01
        r_t = _phi(flat_curve, t, hw_params)   # x_t=0 → r_t = φ(t)
        B    = _B(t, T, hw_params)
        lnA  = _ln_A(t, T, flat_curve, hw_params)
        p_hw = float(np.exp(lnA - B * r_t))
        # P(t,T) with x_t=0 should exactly reproduce P(0,T)/P(0,t) × P(0,t)
        # i.e., be very close to P(0,T)
        p_init = float(flat_curve.df(T))
        assert abs(p_hw - p_init) / p_init < 0.01  # within 1%


# ── simulate_hw ───────────────────────────────────────────────────────────────

class TestSimulateHW:
    def test_output_shape_basic(self, flat_curve, hw_params):
        sim = simulate_hw(flat_curve, hw_params, horizon=1.0, n_steps=50,
                          n_paths=200, antithetic=False)
        assert sim.r_paths.shape == (200, 51)
        assert sim.times.shape == (51,)

    def test_output_shape_antithetic(self, flat_curve, hw_params):
        sim = simulate_hw(flat_curve, hw_params, horizon=1.0, n_steps=50,
                          n_paths=200, antithetic=True)
        # antithetic doubles: 200 base → 400 total (100 base + 100 anti = 200)
        assert sim.r_paths.shape[0] == 200
        assert sim.n_paths == 200

    def test_time_grid_correct(self, flat_curve, hw_params):
        sim = simulate_hw(flat_curve, hw_params, horizon=2.0, n_steps=20,
                          n_paths=100, antithetic=False)
        assert abs(sim.times[0]) < 1e-10
        assert abs(sim.times[-1] - 2.0) < 1e-10

    def test_initial_r_equals_phi_zero(self, flat_curve, hw_params):
        sim = simulate_hw(flat_curve, hw_params, horizon=1.0, n_steps=10,
                          n_paths=100, antithetic=False)
        # x_0 = 0 for all paths → r_0 = φ(0) for all paths
        phi0 = _phi(flat_curve, 0.0, hw_params)
        assert np.allclose(sim.r_paths[:, 0], phi0, atol=1e-8)

    def test_r_paths_positive(self, flat_curve, hw_params):
        sim = simulate_hw(flat_curve, hw_params, horizon=5.0, n_steps=100,
                          n_paths=500, seed=42)
        frac_neg = (sim.r_paths < 0).mean()
        # Very few negative paths expected for 4.33% SOFR with σ=1%
        assert frac_neg < 0.10

    def test_antithetic_symmetry(self, flat_curve, hw_params):
        sim = simulate_hw(flat_curve, hw_params, horizon=1.0, n_steps=20,
                          n_paths=100, antithetic=True, seed=42)
        n_half = sim.n_paths // 2
        # antithetic x paths: x_anti = -x_base → r_anti ≠ -r_base (due to φ offset)
        # Check x_paths antithetic property
        x_base = sim.x_paths[:n_half]
        x_anti = sim.x_paths[n_half:]
        assert np.allclose(x_base + x_anti, 0, atol=1e-8)

    def test_reproducible_with_seed(self, flat_curve, hw_params):
        s1 = simulate_hw(flat_curve, hw_params, horizon=1.0, n_steps=10,
                         n_paths=50, seed=7, antithetic=False)
        s2 = simulate_hw(flat_curve, hw_params, horizon=1.0, n_steps=10,
                         n_paths=50, seed=7, antithetic=False)
        assert np.allclose(s1.r_paths, s2.r_paths)

    def test_different_seeds_differ(self, flat_curve, hw_params):
        s1 = simulate_hw(flat_curve, hw_params, horizon=1.0, n_steps=10,
                         n_paths=50, seed=1, antithetic=False)
        s2 = simulate_hw(flat_curve, hw_params, horizon=1.0, n_steps=10,
                         n_paths=50, seed=2, antithetic=False)
        assert not np.allclose(s1.r_paths, s2.r_paths)

    def test_expected_path_shape(self, flat_curve, hw_params):
        sim = simulate_hw(flat_curve, hw_params, horizon=2.0, n_steps=20, n_paths=200)
        times, mean, std = sim.expected_path()
        assert len(times) == 21
        assert len(mean) == 21
        assert len(std) == 21

    def test_std_grows_with_time(self, flat_curve, hw_params):
        sim = simulate_hw(flat_curve, hw_params, horizon=5.0, n_steps=50,
                          n_paths=1000, antithetic=True)
        _, _, std = sim.expected_path()
        # Standard deviation of r_t generally grows (then plateaus at long horizon)
        assert std[10] > std[1]


# ── zcb_price_hw ─────────────────────────────────────────────────────────────

class TestZCBPriceHW:
    def test_scalar_input(self, flat_curve, hw_params):
        r0 = _inst_forward(flat_curve, 0.001)
        p  = zcb_price_hw(r0, 0.0, 5.0, flat_curve, hw_params)
        assert 0.5 < float(p) < 1.0

    def test_array_input(self, flat_curve, hw_params):
        r_arr = np.array([0.03, 0.04, 0.05, 0.06])
        p_arr = zcb_price_hw(r_arr, 0.5, 5.0, flat_curve, hw_params)
        assert p_arr.shape == (4,)

    def test_monotone_in_rate(self, flat_curve, hw_params):
        p_low  = float(zcb_price_hw(0.02, 0.5, 5.0, flat_curve, hw_params))
        p_high = float(zcb_price_hw(0.08, 0.5, 5.0, flat_curve, hw_params))
        assert p_low > p_high

    def test_unit_at_maturity(self, flat_curve, hw_params):
        r = np.array([0.04])
        p = zcb_price_hw(r, 5.0, 5.0, flat_curve, hw_params)
        assert np.allclose(p, 1.0, atol=1e-8)


# ── price_zcb_mc ─────────────────────────────────────────────────────────────

class TestPriceZCBMC:
    def test_mc_price_close_to_analytical(self, flat_curve, hw_params):
        result = price_zcb_mc(flat_curve, hw_params, maturity=5.0,
                               n_paths=20_000, n_steps=100, seed=42)
        assert result["error_bps"] < 10.0   # within 10bps of analytical

    def test_analytical_price_matches_curve(self, flat_curve, hw_params):
        result = price_zcb_mc(flat_curve, hw_params, maturity=2.0,
                               n_paths=1000, n_steps=50, seed=0)
        analytic = result["analytical_price"]
        assert abs(analytic - float(flat_curve.df(2.0))) < 1e-6

    def test_result_keys(self, flat_curve, hw_params):
        result = price_zcb_mc(flat_curve, hw_params, maturity=3.0, n_paths=500)
        for key in ["mc_price", "analytical_price", "error_bps", "mc_stderr", "n_paths"]:
            assert key in result

    def test_mc_95ci_contains_true_value(self, flat_curve, hw_params):
        result = price_zcb_mc(flat_curve, hw_params, maturity=5.0,
                               n_paths=10_000, n_steps=50, seed=42)
        lo, hi = result["mc_95ci"]
        true_p = result["analytical_price"]
        assert lo < true_p < hi

    def test_stderr_positive(self, flat_curve, hw_params):
        result = price_zcb_mc(flat_curve, hw_params, maturity=5.0, n_paths=500)
        assert result["mc_stderr"] > 0

    def test_more_paths_lower_stderr(self, flat_curve, hw_params):
        r1 = price_zcb_mc(flat_curve, hw_params, maturity=5.0, n_paths=500, seed=42)
        r2 = price_zcb_mc(flat_curve, hw_params, maturity=5.0, n_paths=5000, seed=42)
        # Roughly: stderr should scale as 1/√n
        assert r2["mc_stderr"] < r1["mc_stderr"]


# ── price_caplet_mc ───────────────────────────────────────────────────────────

class TestPriceCapletMC:
    def test_positive_price(self, flat_curve, hw_params):
        result = price_caplet_mc(flat_curve, hw_params, strike=0.04,
                                  t_reset=1.0, t_pay=1.25, notional=1e6,
                                  n_paths=5000, seed=42)
        assert result["mc_pv"] > 0

    def test_result_keys(self, flat_curve, hw_params):
        result = price_caplet_mc(flat_curve, hw_params, strike=0.04,
                                  t_reset=1.0, t_pay=1.25, n_paths=1000)
        for key in ["mc_pv", "mc_stderr", "black76_pv", "forward_rate_pct", "n_paths"]:
            assert key in result

    def test_forward_rate_in_range(self, flat_curve, hw_params):
        result = price_caplet_mc(flat_curve, hw_params, strike=0.04,
                                  t_reset=1.0, t_pay=1.25, n_paths=1000)
        assert 0.03 < result["forward_rate_pct"] < 0.06 * 100

    def test_otm_lower_pv_than_itm(self, flat_curve, hw_params):
        r_itm = price_caplet_mc(flat_curve, hw_params, strike=0.02,
                                 t_reset=1.0, t_pay=1.25, notional=1e6, n_paths=3000)
        r_otm = price_caplet_mc(flat_curve, hw_params, strike=0.08,
                                 t_reset=1.0, t_pay=1.25, notional=1e6, n_paths=3000)
        assert r_itm["mc_pv"] > r_otm["mc_pv"]

    def test_95ci_non_degenerate(self, flat_curve, hw_params):
        result = price_caplet_mc(flat_curve, hw_params, strike=0.04,
                                  t_reset=1.0, t_pay=1.25, n_paths=2000)
        lo, hi = result["mc_95ci"]
        assert hi > lo


# ── portfolio_var_hw ─────────────────────────────────────────────────────────

class TestPortfolioVaRHW:
    def test_var_negative_for_long_duration(self, flat_curve, hw_params):
        result = portfolio_var_hw(flat_curve, hw_params, portfolio_dv01=10_000,
                                   n_paths=2000, seed=42)
        assert result["var_usd"] < 0  # VaR is a loss

    def test_cvar_worse_than_var(self, flat_curve, hw_params):
        result = portfolio_var_hw(flat_curve, hw_params, portfolio_dv01=10_000,
                                   n_paths=2000, seed=42)
        assert result["cvar_usd"] <= result["var_usd"]

    def test_higher_confidence_larger_var(self, flat_curve, hw_params):
        r99 = portfolio_var_hw(flat_curve, hw_params, portfolio_dv01=10_000,
                                n_paths=2000, seed=42, confidence=0.99)
        r95 = portfolio_var_hw(flat_curve, hw_params, portfolio_dv01=10_000,
                                n_paths=2000, seed=42, confidence=0.95)
        # 99% VaR more extreme than 95% VaR (more negative)
        assert r99["var_usd"] <= r95["var_usd"]

    def test_result_keys(self, flat_curve, hw_params):
        result = portfolio_var_hw(flat_curve, hw_params, portfolio_dv01=5_000,
                                   n_paths=1000, seed=0)
        for key in ["var_usd", "cvar_usd", "var_bps", "pnl_percentiles", "n_paths"]:
            assert key in result

    def test_pnl_percentiles_ordered(self, flat_curve, hw_params):
        result = portfolio_var_hw(flat_curve, hw_params, portfolio_dv01=10_000,
                                   n_paths=2000, seed=42)
        pcts = result["pnl_percentiles"]
        vals = [pcts["p1"], pcts["p5"], pcts["p25"], pcts["p50"],
                pcts["p75"], pcts["p95"], pcts["p99"]]
        assert vals == sorted(vals)

    def test_larger_dv01_larger_var(self, flat_curve, hw_params):
        r1 = portfolio_var_hw(flat_curve, hw_params, portfolio_dv01=5_000, n_paths=1000)
        r2 = portfolio_var_hw(flat_curve, hw_params, portfolio_dv01=10_000, n_paths=1000)
        assert abs(r2["var_usd"]) > abs(r1["var_usd"])

    def test_short_duration_symmetric_var(self, flat_curve, hw_params):
        # Long loses on rate rise, short loses on rate fall — same magnitude by symmetry.
        r_long  = portfolio_var_hw(flat_curve, hw_params, portfolio_dv01=+10_000,
                                    n_paths=4000, seed=42)
        r_short = portfolio_var_hw(flat_curve, hw_params, portfolio_dv01=-10_000,
                                    n_paths=4000, seed=42)
        assert r_long["var_usd"]  < 0   # long loses when rates rise
        assert r_short["var_usd"] < 0   # short loses when rates fall
        # Magnitudes should be roughly equal (symmetric simulation)
        assert abs(abs(r_long["var_usd"]) - abs(r_short["var_usd"])) / abs(r_long["var_usd"]) < 0.15


# ── parametric_var ────────────────────────────────────────────────────────────

class TestParametricVaR:
    def test_positive_var_usd(self):
        result = parametric_var(portfolio_dv01=10_000, yield_vol_bps=80.0)
        assert result["var_usd"] > 0

    def test_formula_check(self):
        from scipy.stats import norm
        dv01, vol, conf = 10_000, 80.0, 0.99
        result = parametric_var(dv01, vol, horizon=1/252, confidence=conf)
        z = norm.ppf(conf)
        daily_vol = vol / np.sqrt(252)
        expected_var_bps = z * daily_vol
        expected_var_usd = dv01 * expected_var_bps
        assert abs(result["var_usd"] - expected_var_usd) < 0.01

    def test_result_keys(self):
        r = parametric_var(10_000, 80.0)
        for k in ["var_usd", "var_bps", "cvar_usd", "z_alpha", "one_day_yield_vol_bps"]:
            assert k in r

    def test_higher_vol_larger_var(self):
        r1 = parametric_var(10_000, 60.0)
        r2 = parametric_var(10_000, 100.0)
        assert r2["var_usd"] > r1["var_usd"]

    def test_longer_horizon_larger_var(self):
        r1 = parametric_var(10_000, 80.0, horizon=1/252)
        r10 = parametric_var(10_000, 80.0, horizon=10/252)
        assert r10["var_usd"] > r1["var_usd"]

    def test_cvar_exceeds_var(self):
        r = parametric_var(10_000, 80.0)
        assert r["cvar_usd"] >= r["var_usd"]


# ── convergence_diagnostics ───────────────────────────────────────────────────

class TestConvergenceDiagnostics:
    def test_returns_list(self, flat_curve, hw_params):
        results = convergence_diagnostics(flat_curve, hw_params,
                                           test_maturity=5.0,
                                           path_counts=[100, 500, 1000], seed=42)
        assert len(results) == 3

    def test_result_keys(self, flat_curve, hw_params):
        results = convergence_diagnostics(flat_curve, hw_params,
                                           path_counts=[200], seed=42)
        for k in ["n_paths", "mc_price", "error_bps", "mc_stderr"]:
            assert k in results[0]

    def test_stderr_decreases(self, flat_curve, hw_params):
        results = convergence_diagnostics(flat_curve, hw_params,
                                           test_maturity=5.0,
                                           path_counts=[200, 500, 2000], seed=42)
        stderrs = [r["mc_stderr"] for r in results]
        assert stderrs[0] > stderrs[2]  # larger n → smaller stderr


# ── SimulationResult methods ──────────────────────────────────────────────────

class TestSimulationResult:
    def test_r_at_returns_correct_shape(self, flat_curve, hw_params):
        sim = simulate_hw(flat_curve, hw_params, horizon=1.0,
                          n_steps=10, n_paths=100, antithetic=False)
        r5 = sim.r_at(5)
        assert r5.shape == (100,)

    def test_zcb_prices_at_horizon_shape(self, flat_curve, hw_params):
        sim = simulate_hw(flat_curve, hw_params, horizon=1.0,
                          n_steps=10, n_paths=100, antithetic=False)
        maturities = [2.0, 5.0, 10.0]
        prices = sim.zcb_prices_at_horizon(horizon_step=10, maturities=maturities)
        assert prices.shape == (100, 3)

    def test_zcb_prices_positive(self, flat_curve, hw_params):
        sim = simulate_hw(flat_curve, hw_params, horizon=1.0,
                          n_steps=10, n_paths=100, antithetic=False, seed=42)
        prices = sim.zcb_prices_at_horizon(horizon_step=10, maturities=[2.0, 5.0])
        assert (prices > 0).all()

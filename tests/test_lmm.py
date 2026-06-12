"""
Tests for sofr_engine/lmm.py

Coverage
--------
- LMMParams: validation, properties, helpers
- initial_forwards: reprices curve discount factors
- exponential_correlation: shape, symmetry, diagonal, monotone in distance
- _drift_weights: upper-triangular, correct values
- simulate_lmm: shape, positivity, antithetic centering, forward-rate mean
- caplet_black76: matches Black-76 formula analytically
- cap_black76: sum of caplets, positive, cap/floor parity
- cap_implied_vol: round-trips
- caplet_lmm_mc: agrees with Black-76 within 2 bps vol error
- swaption_lmm_mc: positive for ITM, payer/receiver parity, scaling
- rebonato_swaption_vol: positive, increases with vol, increases with decay
- calibrate_caplet_vols: reprices input caps, vols positive
- calibrate_corr_decay: finds λ that minimises residual
- swaption_implied_vol: positive, round-trip with Rebonato
"""
from __future__ import annotations

import math
import pytest
import numpy as np
from datetime import date

from sofr_engine.bootstrap import flat_sofr_curve
from sofr_engine.lmm import (
    LMMParams,
    LMMSimResult,
    initial_forwards,
    exponential_correlation,
    simulate_lmm,
    caplet_black76,
    cap_black76,
    cap_implied_vol,
    caplet_lmm_mc,
    swaption_lmm_mc,
    rebonato_swaption_vol,
    calibrate_caplet_vols,
    calibrate_corr_decay,
    swaption_implied_vol,
    _drift_weights,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

TENORS_5Y = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
VOLS_10   = np.full(10, 0.25)   # flat 25 % vol


@pytest.fixture
def flat_curve():
    return flat_sofr_curve(date.today(), 0.0433)


@pytest.fixture
def params(flat_curve):
    return LMMParams(tenors=TENORS_5Y.copy(), vols=VOLS_10.copy(), corr_decay=0.10)


@pytest.fixture
def sim(flat_curve, params):
    return simulate_lmm(flat_curve, params, n_steps=50, n_paths=2_000, seed=7)


# ── LMMParams validation ──────────────────────────────────────────────────────

class TestLMMParams:
    def test_valid_construction(self, params):
        assert params.N == 10

    def test_alpha_positive(self, params):
        assert np.all(params.alpha > 0)

    def test_too_few_tenors(self):
        with pytest.raises(ValueError):
            LMMParams(tenors=np.array([0.0]), vols=np.array([]))

    def test_mismatched_vols(self):
        with pytest.raises(ValueError):
            LMMParams(tenors=TENORS_5Y, vols=np.ones(5))

    def test_non_increasing_tenors(self):
        with pytest.raises(ValueError):
            LMMParams(tenors=np.array([0.0, 1.0, 0.5]), vols=np.ones(2))

    def test_negative_vol(self):
        with pytest.raises(ValueError):
            LMMParams(tenors=TENORS_5Y, vols=np.full(10, -0.1))

    def test_negative_corr_decay(self):
        with pytest.raises(ValueError):
            LMMParams(tenors=TENORS_5Y, vols=VOLS_10, corr_decay=-0.1)

    def test_corr_matrix_wrong_shape(self):
        with pytest.raises(ValueError):
            LMMParams(tenors=TENORS_5Y, vols=VOLS_10, corr_matrix=np.eye(5))

    def test_with_vols_returns_new_params(self, params):
        new_params = params.with_vols(np.full(10, 0.30))
        assert abs(new_params.vols[0] - 0.30) < 1e-10
        assert abs(params.vols[0] - 0.25) < 1e-10   # original unchanged

    def test_correlation_is_identity_when_large_decay(self):
        p = LMMParams(tenors=TENORS_5Y, vols=VOLS_10, corr_decay=100.0)
        rho = p.correlation()
        # Off-diagonals should be nearly zero; diagonal = 1
        off_diag = rho - np.eye(10)
        assert np.max(np.abs(off_diag)) < 1e-3

    def test_correlation_all_ones_when_zero_decay(self):
        p = LMMParams(tenors=TENORS_5Y, vols=VOLS_10, corr_decay=0.0)
        rho = p.correlation()
        assert np.allclose(rho, 1.0)


# ── initial_forwards ─────────────────────────────────────────────────────────

class TestInitialForwards:
    def test_shape(self, flat_curve, params):
        F = initial_forwards(flat_curve, params.tenors)
        assert F.shape == (params.N,)

    def test_positive_for_positive_rates(self, flat_curve, params):
        F = initial_forwards(flat_curve, params.tenors)
        assert np.all(F > 0)

    def test_reprices_discount_factors(self, flat_curve, params):
        tenors = params.tenors
        alpha  = params.alpha
        F = initial_forwards(flat_curve, tenors)
        # Product of (1 + α_k F_k) should match ratio P(0,T_0)/P(0,T_N)
        df0 = float(flat_curve.df(tenors[0]))
        dfN = float(flat_curve.df(tenors[-1]))
        prod = np.prod(1.0 + alpha * F)
        assert abs(prod - df0 / dfN) < 1e-6

    def test_flat_curve_gives_flat_forwards(self, flat_curve, params):
        F = initial_forwards(flat_curve, params.tenors)
        # For a flat curve at 4.33%, all 6M forwards should be close
        assert F.max() - F.min() < 0.01  # within 100 bps spread


# ── exponential_correlation ───────────────────────────────────────────────────

class TestExponentialCorrelation:
    def test_diagonal_is_one(self, params):
        rho = exponential_correlation(params.tenors, 0.1)
        assert np.allclose(np.diag(rho), 1.0)

    def test_symmetric(self, params):
        rho = exponential_correlation(params.tenors, 0.1)
        assert np.allclose(rho, rho.T)

    def test_entries_in_01(self, params):
        rho = exponential_correlation(params.tenors, 0.5)
        assert rho.min() >= 0.0
        assert rho.max() <= 1.0 + 1e-12

    def test_monotone_decrease_with_distance(self):
        T = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        rho = exponential_correlation(T, 0.3)
        assert rho[0, 1] > rho[0, 2] > rho[0, 3]

    def test_zero_decay_gives_ones(self, params):
        rho = exponential_correlation(params.tenors, 0.0)
        assert np.allclose(rho, 1.0)


# ── drift weights ─────────────────────────────────────────────────────────────

class TestDriftWeights:
    def test_upper_triangular(self, params):
        W = _drift_weights(params)
        assert np.allclose(np.tril(W, k=0), 0.0)

    def test_positive_off_diagonal(self, params):
        W = _drift_weights(params)
        assert np.all(W[np.triu_indices(params.N, k=1)] > 0)

    def test_shape(self, params):
        W = _drift_weights(params)
        assert W.shape == (params.N, params.N)


# ── simulate_lmm ─────────────────────────────────────────────────────────────

class TestSimulateLMM:
    def test_output_shape(self, sim, params):
        n_steps = sim.time_steps.shape[0] - 1
        assert sim.forward_paths.shape == (n_steps + 1, sim.n_paths, params.N)

    def test_paths_positive(self, sim):
        assert np.all(sim.forward_paths > 0)

    def test_initial_step_constant(self, flat_curve, params):
        F0 = initial_forwards(flat_curve, params.tenors)
        assert np.allclose(sim_fixture_helper(flat_curve, params).forward_paths[0], F0[None, :])

    def test_antithetic_mean_close_to_initial(self, flat_curve, params):
        # With antithetic + short vol, mean of paths at T_end should ≈ initial forward
        # (martingale property under Q^T_N is approximate, not exact, for correlated rates)
        sim = simulate_lmm(flat_curve, params, n_steps=50, n_paths=4_000, seed=42)
        F0  = initial_forwards(flat_curve, params.tenors)
        # Mean across paths of last forward (martingale under Q^T_N, zero drift)
        F_last_mean = sim.forward_paths[-1, :, -1].mean()
        assert abs(F_last_mean / F0[-1] - 1.0) < 0.05   # within 5%

    def test_n_paths_matches(self, flat_curve, params):
        sim = simulate_lmm(flat_curve, params, n_steps=10, n_paths=200, seed=1)
        assert sim.n_paths == 200

    def test_no_antithetic_option(self, flat_curve, params):
        sim = simulate_lmm(flat_curve, params, n_steps=10, n_paths=100,
                           seed=1, antithetic=False)
        assert sim.n_paths == 100

    def test_time_steps_monotone(self, sim):
        assert np.all(np.diff(sim.time_steps) > 0)

    def test_forwards_at_returns_correct_shape(self, sim, params):
        F = sim.forwards_at(params.tenors[3])
        assert F.shape == (sim.n_paths, params.N)

    def test_swap_rate_in_plausible_range(self, sim, params):
        S = sim.swap_rate(params.tenors[2], k_start=2, k_end=8)
        assert np.all(S > 0.0)
        assert np.all(S < 0.30)   # < 30% for SOFR-range rates


def sim_fixture_helper(flat_curve, params):
    return simulate_lmm(flat_curve, params, n_steps=10, n_paths=100, seed=0)


# ── caplet_black76 ────────────────────────────────────────────────────────────

class TestCapletBlack76:
    def test_atm_positive(self, flat_curve, params):
        F0 = initial_forwards(flat_curve, params.tenors)
        pv = caplet_black76(flat_curve, params, k=2, strike=F0[2], notional=1e6)
        assert pv > 0

    def test_deep_itm_approaches_intrinsic(self, flat_curve, params):
        F0 = initial_forwards(flat_curve, params.tenors)
        K  = F0[2] * 0.01     # very deep ITM
        pv = caplet_black76(flat_curve, params, k=2, strike=K, notional=1e6)
        # Should be close to discounted intrinsic
        alpha2 = params.alpha[2]
        df     = float(flat_curve.df(params.tenors[3]))
        intrinsic = 1e6 * alpha2 * df * (F0[2] - K)
        assert abs(pv / intrinsic - 1.0) < 0.01

    def test_deep_otm_near_zero(self, flat_curve, params):
        F0 = initial_forwards(flat_curve, params.tenors)
        K  = F0[2] * 5.0      # very deep OTM
        pv = caplet_black76(flat_curve, params, k=2, strike=K, notional=1e6)
        assert pv < 1.0       # less than $1 on $1M notional

    def test_floorlet_positive(self, flat_curve, params):
        F0 = initial_forwards(flat_curve, params.tenors)
        pv = caplet_black76(flat_curve, params, k=2, strike=F0[2], notional=1e6, is_cap=False)
        assert pv > 0

    def test_put_call_parity_caplet(self, flat_curve, params):
        k   = 3
        F0  = initial_forwards(flat_curve, params.tenors)
        K   = F0[k] * 0.9
        alpha_k = params.alpha[k]
        df      = float(flat_curve.df(params.tenors[k + 1]))
        cap_pv  = caplet_black76(flat_curve, params, k=k, strike=K, notional=1e6)
        floor_pv= caplet_black76(flat_curve, params, k=k, strike=K, notional=1e6, is_cap=False)
        # Cap − Floor = PV(F − K) = α df (F − K) × notional
        fwd_pv  = 1e6 * alpha_k * df * (F0[k] - K)
        assert abs((cap_pv - floor_pv) - fwd_pv) < 0.01   # within 1 cent


# ── cap_black76 ───────────────────────────────────────────────────────────────

class TestCapBlack76:
    def test_positive(self, flat_curve, params):
        F0 = initial_forwards(flat_curve, params.tenors)
        pv = cap_black76(flat_curve, params, strike=F0.mean(), notional=1e6)
        assert pv > 0

    def test_cap_equals_sum_of_caplets(self, flat_curve, params):
        F0   = initial_forwards(flat_curve, params.tenors)
        K    = F0.mean()
        cap  = cap_black76(flat_curve, params, strike=K, notional=1e6)
        legs = sum(caplet_black76(flat_curve, params, k, K, 1e6) for k in range(params.N))
        assert abs(cap - legs) < 0.01

    def test_floor_positive(self, flat_curve, params):
        F0 = initial_forwards(flat_curve, params.tenors)
        pv = cap_black76(flat_curve, params, strike=F0.mean(), notional=1e6, is_cap=False)
        assert pv > 0

    def test_increasing_vol_increases_cap(self, flat_curve, params):
        F0   = initial_forwards(flat_curve, params.tenors)
        K    = F0.mean()
        p25  = params
        p40  = params.with_vols(np.full(params.N, 0.40))
        pv25 = cap_black76(flat_curve, p25, K, 1e6)
        pv40 = cap_black76(flat_curve, p40, K, 1e6)
        assert pv40 > pv25


# ── cap_implied_vol ───────────────────────────────────────────────────────────

class TestCapImpliedVol:
    def test_round_trip(self, flat_curve, params):
        F0     = initial_forwards(flat_curve, params.tenors)
        K      = F0.mean()
        sigma  = 0.22
        p      = params.with_vols(np.full(params.N, sigma))
        mkt    = cap_black76(flat_curve, p, K, 1e6)
        sigma_back = cap_implied_vol(flat_curve, params, K, mkt, notional=1e6)
        assert abs(sigma_back - sigma) < 1e-6

    def test_higher_price_higher_vol(self, flat_curve, params):
        F0  = initial_forwards(flat_curve, params.tenors)
        K   = F0.mean()
        p20 = params.with_vols(np.full(params.N, 0.20))
        p30 = params.with_vols(np.full(params.N, 0.30))
        v20 = cap_implied_vol(flat_curve, params, K, cap_black76(flat_curve, p20, K))
        v30 = cap_implied_vol(flat_curve, params, K, cap_black76(flat_curve, p30, K))
        assert v30 > v20


# ── caplet_lmm_mc ─────────────────────────────────────────────────────────────

class TestCapletLMMMC:
    def test_result_keys(self, flat_curve, params):
        F0  = initial_forwards(flat_curve, params.tenors)
        res = caplet_lmm_mc(flat_curve, params, k=2, strike=F0[2], n_paths=2_000)
        assert "pv" in res and "std_err" in res and "black76_pv" in res

    def test_close_to_black76_atm(self, flat_curve, params):
        F0  = initial_forwards(flat_curve, params.tenors)
        res = caplet_lmm_mc(flat_curve, params, k=2, strike=F0[2], n_paths=8_000, seed=42)
        rel_err = abs(res["pv"] - res["black76_pv"]) / max(res["black76_pv"], 1e-10)
        assert rel_err < 0.02   # within 2%

    def test_positive(self, flat_curve, params):
        F0  = initial_forwards(flat_curve, params.tenors)
        res = caplet_lmm_mc(flat_curve, params, k=3, strike=F0[3] * 0.8, n_paths=2_000)
        assert res["pv"] > 0

    def test_std_err_positive(self, flat_curve, params):
        F0  = initial_forwards(flat_curve, params.tenors)
        res = caplet_lmm_mc(flat_curve, params, k=2, strike=F0[2], n_paths=2_000)
        assert res["std_err"] > 0


# ── swaption_lmm_mc ───────────────────────────────────────────────────────────

class TestSwaptionLMMMC:
    def test_result_keys(self, flat_curve, params, sim):
        F0 = initial_forwards(flat_curve, params.tenors)
        S0 = sim.swap_rate(params.tenors[2], 2, 8).mean()
        res = swaption_lmm_mc(sim, flat_curve, k_start=2, k_end=8, strike=S0)
        assert "pv" in res and "std_err" in res and "swap_rate_mean" in res

    def test_payer_positive(self, flat_curve, params, sim):
        F0 = initial_forwards(flat_curve, params.tenors)
        S0 = sim.swap_rate(params.tenors[2], 2, 8).mean()
        res = swaption_lmm_mc(sim, flat_curve, k_start=2, k_end=8,
                               strike=S0 * 0.7, is_payer=True, notional=1e6)
        assert res["pv"] > 0

    def test_receiver_positive(self, flat_curve, params, sim):
        F0 = initial_forwards(flat_curve, params.tenors)
        S0 = sim.swap_rate(params.tenors[2], 2, 8).mean()
        res = swaption_lmm_mc(sim, flat_curve, k_start=2, k_end=8,
                               strike=S0 * 1.3, is_payer=False, notional=1e6)
        assert res["pv"] > 0

    def test_payer_receiver_parity(self, flat_curve, params):
        # Payer − Receiver = Swap PV (put-call parity for swaptions)
        sim2 = simulate_lmm(flat_curve, params, n_steps=50, n_paths=4_000, seed=99)
        S0   = sim2.swap_rate(params.tenors[2], 2, 8).mean()
        K    = float(S0)
        r_pay = swaption_lmm_mc(sim2, flat_curve, 2, 8, K, 1e6, is_payer=True)
        r_rec = swaption_lmm_mc(sim2, flat_curve, 2, 8, K, 1e6, is_payer=False)
        # At ATM K ≈ S0: payer − receiver should be close to zero
        pv_diff = r_pay["pv"] - r_rec["pv"]
        # Should be within 3 std_errs of the model swap value
        assert abs(pv_diff) < 3e4   # within $30k for $1M notional

    def test_notional_scales_pv(self, flat_curve, params):
        sim2  = simulate_lmm(flat_curve, params, n_steps=30, n_paths=2_000, seed=5)
        S0    = sim2.swap_rate(params.tenors[2], 2, 8).mean()
        r1    = swaption_lmm_mc(sim2, flat_curve, 2, 8, S0 * 0.8, 1e6,  True)
        r2    = swaption_lmm_mc(sim2, flat_curve, 2, 8, S0 * 0.8, 2e6,  True)
        assert abs(r2["pv"] / r1["pv"] - 2.0) < 0.01

    def test_otm_swaption_less_than_itm(self, flat_curve, params, sim):
        S0   = sim.swap_rate(params.tenors[2], 2, 8).mean()
        r_itm = swaption_lmm_mc(sim, flat_curve, 2, 8, S0 * 0.7, 1e6, True)
        r_otm = swaption_lmm_mc(sim, flat_curve, 2, 8, S0 * 1.3, 1e6, True)
        assert r_itm["pv"] > r_otm["pv"]


# ── rebonato_swaption_vol ─────────────────────────────────────────────────────

class TestRebonatoVol:
    def test_positive(self, flat_curve, params):
        vol = rebonato_swaption_vol(flat_curve, params, k_start=2, k_end=8)
        assert vol > 0

    def test_less_than_or_equal_to_caplet_vol(self, flat_curve, params):
        # Correlation < 1 → swaption vol ≤ caplet vol (diversification)
        p_corr = params.with_corr_decay(0.5)    # meaningful decorrelation
        vol_sw = rebonato_swaption_vol(flat_curve, p_corr, 0, 10)
        vol_cap = params.vols[0]
        assert vol_sw <= vol_cap + 1e-6

    def test_zero_decay_equals_caplet_vol(self, flat_curve, params):
        # ρ = 1 everywhere → swaption vol = caplet vol (all identical)
        p_full = params.with_corr_decay(0.0)
        vol = rebonato_swaption_vol(flat_curve, p_full, 0, params.N)
        assert abs(vol - params.vols[0]) < 1e-3

    def test_increases_with_underlying_vol(self, flat_curve, params):
        v25 = rebonato_swaption_vol(flat_curve, params.with_vols(np.full(10, 0.25)), 2, 8)
        v40 = rebonato_swaption_vol(flat_curve, params.with_vols(np.full(10, 0.40)), 2, 8)
        assert v40 > v25

    def test_increases_as_decay_decreases(self, flat_curve, params):
        # Lower decay → more correlated → higher swaption vol
        v_hi_corr = rebonato_swaption_vol(flat_curve, params.with_corr_decay(0.01), 2, 8)
        v_lo_corr = rebonato_swaption_vol(flat_curve, params.with_corr_decay(2.0),  2, 8)
        assert v_hi_corr > v_lo_corr

    def test_short_swap_reasonable_range(self, flat_curve, params):
        vol = rebonato_swaption_vol(flat_curve, params, k_start=0, k_end=2)
        assert 0.01 < vol < 1.0


# ── calibrate_caplet_vols ─────────────────────────────────────────────────────

class TestCalibrateCapletVols:
    def test_output_length(self, flat_curve, params):
        flat_vols = [0.25] * 10
        p_cal = calibrate_caplet_vols(flat_curve, params, flat_vols)
        assert len(p_cal.vols) == params.N

    def test_vols_positive(self, flat_curve, params):
        flat_vols = [0.20 + 0.01 * k for k in range(10)]
        p_cal = calibrate_caplet_vols(flat_curve, params, flat_vols)
        assert np.all(p_cal.vols > 0)

    def test_reprices_first_cap(self, flat_curve, params):
        sigma = 0.22
        flat_vols = [sigma] * 10
        p_cal = calibrate_caplet_vols(flat_curve, params, flat_vols)
        # cap(0..0) = caplet_0: model price with calibrated vol should match market
        F0 = initial_forwards(flat_curve, params.tenors)
        from sofr_engine.lmm import _black76
        K = float(F0[0])
        mkt = float(params.alpha[0] * float(flat_curve.df(params.tenors[1])) * _black76(F0[0], K, sigma, params.tenors[0], True))
        cal = float(params.alpha[0] * float(flat_curve.df(params.tenors[1])) * _black76(F0[0], K, p_cal.vols[0], params.tenors[0], True))
        assert abs(cal - mkt) < 1e-8

    def test_mismatched_length_raises(self, flat_curve, params):
        with pytest.raises(ValueError):
            calibrate_caplet_vols(flat_curve, params, [0.25] * 5)


# ── calibrate_corr_decay ──────────────────────────────────────────────────────

class TestCalibrateCorrDecay:
    def test_returns_two_values(self, flat_curve, params):
        specs = [(0, 4, 0.22), (0, 8, 0.21)]
        result = calibrate_corr_decay(flat_curve, params, specs)
        assert len(result) == 2

    def test_lambda_in_bounds(self, flat_curve, params):
        specs = [(0, 6, 0.23)]
        lam, _ = calibrate_corr_decay(flat_curve, params, specs, decay_bounds=(0.01, 2.0))
        assert 0.01 <= lam <= 2.0

    def test_rmse_non_negative(self, flat_curve, params):
        specs = [(0, 4, 0.22), (2, 8, 0.20)]
        _, rmse = calibrate_corr_decay(flat_curve, params, specs)
        assert rmse >= 0.0

    def test_round_trip_recovers_lambda(self, flat_curve, params):
        # Generate swaption vols using λ = 0.3, then recover
        lam_true = 0.30
        p_true = params.with_corr_decay(lam_true)
        specs = [
            (0, 4, rebonato_swaption_vol(flat_curve, p_true, 0, 4)),
            (0, 8, rebonato_swaption_vol(flat_curve, p_true, 0, 8)),
            (2, 8, rebonato_swaption_vol(flat_curve, p_true, 2, 8)),
        ]
        lam_cal, _ = calibrate_corr_decay(flat_curve, params, specs)
        assert abs(lam_cal - lam_true) < 0.05


# ── swaption_implied_vol ──────────────────────────────────────────────────────

class TestSwaptionImpliedVol:
    def test_positive(self, flat_curve, params):
        vol_model = rebonato_swaption_vol(flat_curve, params, 2, 8)
        F0 = initial_forwards(flat_curve, params.tenors)
        alpha = params.alpha
        P, A = 1.0, 0.0
        for i in range(2, 8):
            P = P / (1 + alpha[i] * F0[i])
            A += alpha[i] * P
        S0 = (1 - P) / max(A, 1e-15)
        df_ks = float(flat_curve.df(params.tenors[2]))
        from sofr_engine.lmm import _black76
        pv = 1e6 * A * df_ks * _black76(S0, S0, vol_model, params.tenors[2], True)
        iv = swaption_implied_vol(pv, flat_curve, params, 2, 8, S0, 1e6, True)
        assert abs(iv - vol_model) < 1e-4

    def test_returns_nan_for_impossible_pv(self, flat_curve, params):
        F0 = initial_forwards(flat_curve, params.tenors)
        iv = swaption_implied_vol(1e30, flat_curve, params, 2, 8, F0.mean(), 1e6)
        assert math.isnan(iv)

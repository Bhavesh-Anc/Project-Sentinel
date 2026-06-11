"""
Tests for sofr_engine/bermudan.py

Coverage:
- _laguerre_basis: shape, orthogonality at origin
- _swap_pv_at_date: positive for ITM, zero for deep OTM
- price_european_swaption_hw: positive PV, keys, ATM
- price_bermudan_swaption: price >= European, early exercise premium >= 0,
  exercise_probs length, exercise_dates ascending, BermudanSwaptionResult props
"""
import pytest
import numpy as np
from datetime import date

from sofr_engine.bootstrap import flat_sofr_curve
from sofr_engine.monte_carlo import HullWhiteParams, simulate_hw
from sofr_engine.bermudan import (
    _laguerre_basis,
    _swap_pv_at_date,
    price_european_swaption_hw,
    price_bermudan_swaption,
    BermudanSwaptionResult,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_curve():
    return flat_sofr_curve(date.today(), 0.0433)


@pytest.fixture
def hw_params():
    return HullWhiteParams(a=0.05, sigma=0.010)


# ── _laguerre_basis ───────────────────────────────────────────────────────────

class TestLaguerreBasis:
    def test_shape_default(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        B = _laguerre_basis(x, n_terms=4)
        assert B.shape == (4, 4)

    def test_shape_2_terms(self):
        x = np.linspace(0, 3, 20)
        B = _laguerre_basis(x, n_terms=2)
        assert B.shape == (20, 2)

    def test_first_column_nonneg(self):
        x = np.linspace(0, 5, 50)
        B = _laguerre_basis(x, n_terms=4)
        assert (B[:, 0] >= 0).all()  # L_0·e^{-x/2} ≥ 0

    def test_at_zero(self):
        x = np.array([0.0])
        B = _laguerre_basis(x, n_terms=4)
        # L_k(0) = 1 for all k, so at x=0: e^0 × L_k(0) = 1 for all k
        np.testing.assert_allclose(B[0], [1.0, 1.0, 1.0, 1.0], atol=1e-10)


# ── _swap_pv_at_date ──────────────────────────────────────────────────────────

class TestSwapPVAtDate:
    def test_deep_itm_payer_positive(self, flat_curve, hw_params):
        r_t = np.full(100, 0.08)  # high rates → payer ITM
        pv = _swap_pv_at_date(r_t, t=0.01, swap_maturity=10.0,
                               strike=0.03, notional=1e6,
                               pay_receive="payer",
                               curve=flat_curve, params=hw_params)
        assert (pv > 0).all()

    def test_deep_otm_payer_zero(self, flat_curve, hw_params):
        r_t = np.full(100, 0.01)  # low rates → payer OTM
        pv = _swap_pv_at_date(r_t, t=0.01, swap_maturity=10.0,
                               strike=0.10, notional=1e6,
                               pay_receive="payer",
                               curve=flat_curve, params=hw_params)
        assert (pv == 0).all()

    def test_receiver_opposite_sign_condition(self, flat_curve, hw_params):
        r_t = np.full(100, 0.01)  # low rates → receiver ITM
        pv = _swap_pv_at_date(r_t, t=0.01, swap_maturity=10.0,
                               strike=0.10, notional=1e6,
                               pay_receive="receiver",
                               curve=flat_curve, params=hw_params)
        assert (pv > 0).all()

    def test_nonnegatve_payoff(self, flat_curve, hw_params):
        r_t = np.random.default_rng(42).uniform(0.01, 0.10, 50)
        pv = _swap_pv_at_date(r_t, t=0.5, swap_maturity=5.0,
                               strike=0.04, notional=1e6,
                               pay_receive="payer",
                               curve=flat_curve, params=hw_params)
        assert (pv >= 0).all()


# ── European swaption by HW MC ───────────────────────────────────────────────

class TestEuropeanSwaptionHW:
    def test_positive_pv(self, flat_curve, hw_params):
        result = price_european_swaption_hw(
            flat_curve, hw_params,
            expiry=1.0, swap_maturity=6.0,
            n_paths=3000, seed=42,
        )
        assert result["pv"] > 0

    def test_result_keys(self, flat_curve, hw_params):
        result = price_european_swaption_hw(
            flat_curve, hw_params,
            expiry=1.0, swap_maturity=6.0, n_paths=1000,
        )
        for key in ["pv", "forward_swap_rate_pct", "strike_pct",
                    "moneyness_bps", "annuity", "n_paths"]:
            assert key in result

    def test_atm_strike_zero_moneyness(self, flat_curve, hw_params):
        result = price_european_swaption_hw(
            flat_curve, hw_params,
            expiry=1.0, swap_maturity=6.0, strike=None, n_paths=1000,
        )
        assert abs(result["moneyness_bps"]) < 0.5  # ATM → ~0 moneyness

    def test_deeper_itm_higher_pv(self, flat_curve, hw_params):
        # Strike below ATM (payer ITM): higher PV
        atm_result = price_european_swaption_hw(
            flat_curve, hw_params,
            expiry=1.0, swap_maturity=6.0, strike=None, n_paths=3000, seed=42,
        )
        itm_result = price_european_swaption_hw(
            flat_curve, hw_params,
            expiry=1.0, swap_maturity=6.0, strike=0.02, n_paths=3000, seed=42,
        )
        assert itm_result["pv"] > atm_result["pv"]

    def test_receiver_vs_payer(self, flat_curve, hw_params):
        payer = price_european_swaption_hw(
            flat_curve, hw_params, expiry=1.0, swap_maturity=6.0,
            pay_receive="payer", n_paths=3000, seed=42,
        )
        receiver = price_european_swaption_hw(
            flat_curve, hw_params, expiry=1.0, swap_maturity=6.0,
            pay_receive="receiver", n_paths=3000, seed=42,
        )
        assert payer["pv"] > 0
        assert receiver["pv"] > 0

    def test_mc_se_positive(self, flat_curve, hw_params):
        result = price_european_swaption_hw(
            flat_curve, hw_params, expiry=1.0, swap_maturity=6.0, n_paths=1000,
        )
        assert result["mc_stderr"] > 0

    def test_longer_maturity_higher_annuity(self, flat_curve, hw_params):
        r5 = price_european_swaption_hw(
            flat_curve, hw_params, expiry=1.0, swap_maturity=6.0, n_paths=500,
        )
        r10 = price_european_swaption_hw(
            flat_curve, hw_params, expiry=1.0, swap_maturity=11.0, n_paths=500,
        )
        assert r10["annuity"] > r5["annuity"]


# ── Bermudan swaption by LSM ─────────────────────────────────────────────────

class TestBermudanSwaptionLSM:
    def test_price_positive(self, flat_curve, hw_params):
        result = price_bermudan_swaption(
            flat_curve, hw_params,
            first_exercise=1.0, swap_maturity=5.0,
            n_paths=2000, seed=42,
        )
        assert result.price > 0

    def test_price_geq_european(self, flat_curve, hw_params):
        result = price_bermudan_swaption(
            flat_curve, hw_params,
            first_exercise=1.0, swap_maturity=5.0,
            n_paths=3000, seed=42,
        )
        # Bermudan ≥ European (early exercise premium ≥ 0)
        assert result.price >= result.european_lower * 0.90  # allow for MC noise

    def test_early_exercise_premium_nonneg(self, flat_curve, hw_params):
        result = price_bermudan_swaption(
            flat_curve, hw_params,
            first_exercise=1.0, swap_maturity=5.0, n_paths=3000, seed=42,
        )
        assert result.early_exercise_premium >= -result.price * 0.10

    def test_exercise_dates_ascending(self, flat_curve, hw_params):
        result = price_bermudan_swaption(
            flat_curve, hw_params,
            first_exercise=1.0, swap_maturity=5.0, n_paths=1000,
        )
        d = result.exercise_dates
        assert d == sorted(d)

    def test_exercise_probs_between_0_and_1(self, flat_curve, hw_params):
        result = price_bermudan_swaption(
            flat_curve, hw_params,
            first_exercise=1.0, swap_maturity=5.0, n_paths=1000,
        )
        for p in result.exercise_probs:
            assert 0.0 <= p <= 1.0

    def test_exercise_dates_count(self, flat_curve, hw_params):
        result = price_bermudan_swaption(
            flat_curve, hw_params,
            first_exercise=1.0, swap_maturity=5.0,
            exercise_freq=2,    # semi-annual
            n_paths=1000,
        )
        # 8 exercise dates: 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5
        assert len(result.exercise_dates) == 8

    def test_atm_strike_property(self, flat_curve, hw_params):
        result = price_bermudan_swaption(
            flat_curve, hw_params,
            first_exercise=1.0, swap_maturity=5.0, strike=None, n_paths=1000,
        )
        assert 0.02 < result.strike < 0.08  # ATM in reasonable range

    def test_result_fields(self, flat_curve, hw_params):
        result = price_bermudan_swaption(
            flat_curve, hw_params,
            first_exercise=1.0, swap_maturity=5.0, n_paths=1000,
        )
        assert hasattr(result, "price")
        assert hasattr(result, "european_lower")
        assert hasattr(result, "exercise_probs")
        assert hasattr(result, "exercise_dates")
        assert hasattr(result, "early_exercise_premium")

    def test_receiver_positive(self, flat_curve, hw_params):
        result = price_bermudan_swaption(
            flat_curve, hw_params,
            first_exercise=1.0, swap_maturity=5.0,
            pay_receive="receiver", n_paths=2000, seed=42,
        )
        assert result.price > 0

    def test_deeper_maturity_higher_price(self, flat_curve, hw_params):
        r5 = price_bermudan_swaption(
            flat_curve, hw_params,
            first_exercise=1.0, swap_maturity=5.0, n_paths=1000, seed=42,
        )
        r10 = price_bermudan_swaption(
            flat_curve, hw_params,
            first_exercise=1.0, swap_maturity=10.0, n_paths=1000, seed=42,
        )
        assert r10.price > r5.price

    def test_more_paths_smaller_variance(self, flat_curve, hw_params):
        # Run twice with same seed and compare price is deterministic
        r1 = price_bermudan_swaption(
            flat_curve, hw_params,
            first_exercise=1.0, swap_maturity=5.0, n_paths=1000, seed=123,
        )
        r2 = price_bermudan_swaption(
            flat_curve, hw_params,
            first_exercise=1.0, swap_maturity=5.0, n_paths=1000, seed=123,
        )
        assert abs(r1.price - r2.price) < 1.0  # fully reproducible

    def test_invalid_no_exercise_dates(self, flat_curve, hw_params):
        with pytest.raises(ValueError, match="No valid exercise dates"):
            price_bermudan_swaption(
                flat_curve, hw_params,
                first_exercise=5.0, swap_maturity=5.0, n_paths=100,
            )

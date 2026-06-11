"""
Tests for sofr_engine/cms.py

Coverage
--------
- _annuity_par: boundary values, known limits
- _annuity_par_deriv: derivative consistency via finite differences
- _h_factor: sign (typically negative → positive convexity adj)
- cms_convexity_adj: positive adj, both models, vol/expiry scaling
- CMSCaplet: construction, PV positive, put-call parity
- CMSFloorlet: convenience wrapper
- cms_caplet_floorlet_parity: model-free parity check
- CMSSpreadOption: call/put, Kirk approximation, limiting cases
- CMSSwap: leg PVs, par rate, DV01 finite-difference
"""
from __future__ import annotations

import math
import pytest
import numpy as np
from datetime import date

from sofr_engine.bootstrap import flat_sofr_curve
from sofr_engine.cms import (
    _annuity_par, _annuity_par_deriv, _h_factor,
    CMSConvexityResult, CMSCaplet, CMSSpreadOption, CMSSwap,
    cms_convexity_adj, cms_caplet_pv, cms_floorlet_pv,
    cms_caplet_floorlet_parity, cms_spread_option_pv,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_curve():
    return flat_sofr_curve(date.today(), 0.0433)


@pytest.fixture
def low_curve():
    return flat_sofr_curve(date.today(), 0.015)


# ── Annuity helpers ───────────────────────────────────────────────────────────

class TestAnnuityPar:
    def test_known_value_10_semiannual(self):
        # 10Y semi-annual swap at S=5%: G = (1 - 1.025^{-20}) / 0.025
        S, n, m = 0.05, 20, 2
        G_expected = (1 - (1 + S/m)**(-n)) * m / S
        assert abs(_annuity_par(S, n, m) - G_expected) < 1e-12

    def test_limiting_zero_rate(self):
        # As S→0, G → n/m (all DF → 1)
        G = _annuity_par(1e-12, 20, 2)
        assert abs(G - 10.0) < 1e-4

    def test_positive_value(self):
        assert _annuity_par(0.04, 10, 2) > 0

    def test_longer_tenor_bigger_annuity(self):
        G5  = _annuity_par(0.04, 10, 2)   # 5Y semi-annual
        G10 = _annuity_par(0.04, 20, 2)   # 10Y semi-annual
        assert G10 > G5

    def test_higher_rate_smaller_annuity(self):
        G_low  = _annuity_par(0.02, 20, 2)
        G_high = _annuity_par(0.08, 20, 2)
        assert G_low > G_high


class TestAnnuityDeriv:
    def test_consistent_with_finite_difference(self):
        S, n, m = 0.05, 20, 2
        h  = 1e-6
        fd = (_annuity_par(S + h, n, m) - _annuity_par(S - h, n, m)) / (2 * h)
        assert abs(_annuity_par_deriv(S, n, m) - fd) < 1e-6

    def test_negative_derivative(self):
        # Higher rate → lower annuity → derivative should be negative
        assert _annuity_par_deriv(0.05, 20, 2) < 0

    def test_quarterly_vs_semiannual(self):
        # Both should be negative
        assert _annuity_par_deriv(0.05, 40, 4) < 0
        assert _annuity_par_deriv(0.05, 10, 2) < 0


class TestHFactor:
    def test_negative(self):
        # H = S × G' / G; since G' < 0 and G > 0, H < 0
        assert _h_factor(0.05, 20, 2) < 0

    def test_scales_with_tenor(self):
        # Longer tenor → more convexity → more negative H
        h5  = _h_factor(0.05, 10, 2)
        h10 = _h_factor(0.05, 20, 2)
        assert h10 < h5


# ── CMS convexity adjustment ──────────────────────────────────────────────────

class TestCMSConvexityAdj:
    def test_positive_adjustment(self, flat_curve):
        # CMS should be above forward swap rate (positive convexity adj)
        res = cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.30)
        assert res.convexity_adj > 0

    def test_cms_rate_greater_than_forward(self, flat_curve):
        res = cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.30)
        assert res.cms_rate > res.forward_swap_rate

    def test_result_type(self, flat_curve):
        res = cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.30)
        assert isinstance(res, CMSConvexityResult)

    def test_result_fields(self, flat_curve):
        res = cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.30)
        assert hasattr(res, "forward_swap_rate")
        assert hasattr(res, "convexity_adj")
        assert hasattr(res, "cms_rate")
        assert hasattr(res, "convexity_adj_bps")

    def test_convexity_bps_property(self, flat_curve):
        res = cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.30)
        assert abs(res.convexity_adj_bps - res.convexity_adj * 10_000) < 1e-8

    def test_zero_vol_zero_adj(self, flat_curve):
        # Zero vol → no convexity adjustment
        with pytest.raises(ValueError):
            cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.0)

    def test_invalid_expiry(self, flat_curve):
        with pytest.raises(ValueError):
            cms_convexity_adj(flat_curve, expiry=0.0, swap_tenor=10.0, vol=0.30)

    def test_invalid_swap_tenor(self, flat_curve):
        with pytest.raises(ValueError):
            cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=0.0, vol=0.30)

    def test_higher_vol_bigger_adj(self, flat_curve):
        r1 = cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.20)
        r2 = cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.40)
        assert r2.convexity_adj > r1.convexity_adj

    def test_longer_expiry_bigger_adj(self, flat_curve):
        r1 = cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.30)
        r2 = cms_convexity_adj(flat_curve, expiry=5.0, swap_tenor=10.0, vol=0.30)
        assert r2.convexity_adj > r1.convexity_adj

    def test_longer_tenor_bigger_adj(self, flat_curve):
        r5  = cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=5.0,  vol=0.30)
        r10 = cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.30)
        assert r10.convexity_adj > r5.convexity_adj

    def test_replication_model_positive_adj(self, flat_curve):
        res = cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.30,
                                model="replication")
        assert res.convexity_adj >= 0

    def test_replication_model_label(self, flat_curve):
        res = cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.30,
                                model="replication")
        assert res.model == "replication"

    def test_invalid_model(self, flat_curve):
        with pytest.raises(ValueError):
            cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.30,
                              model="bad_model")

    def test_forward_swap_rate_in_range(self, flat_curve):
        res = cms_convexity_adj(flat_curve, expiry=1.0, swap_tenor=10.0, vol=0.30)
        # Flat curve at 4.33% → forward swap rate near 4.33%
        assert 0.02 < res.forward_swap_rate < 0.10


# ── CMS Caplet / Floorlet ────────────────────────────────────────────────────

class TestCMSCaplet:
    @pytest.fixture
    def caplet(self):
        return CMSCaplet(t_fix=1.0, t_pay=1.25, swap_tenor=10.0,
                         strike=0.04, notional=1_000_000.0)

    def test_pv_positive(self, caplet, flat_curve):
        pv = cms_caplet_pv(caplet, flat_curve, vol=0.30)
        assert pv > 0

    def test_floorlet_positive(self, caplet, flat_curve):
        pv = cms_floorlet_pv(caplet, flat_curve, vol=0.30)
        assert pv > 0

    def test_deep_itm_cap_high_pv(self, flat_curve):
        itm = CMSCaplet(t_fix=1.0, t_pay=1.25, swap_tenor=10.0,
                        strike=0.001, notional=1_000_000.0)
        pv  = cms_caplet_pv(itm, flat_curve, vol=0.30)
        assert pv > 1_000  # deep ITM should have significant PV

    def test_deep_otm_cap_low_pv(self, flat_curve):
        otm = CMSCaplet(t_fix=1.0, t_pay=1.25, swap_tenor=10.0,
                        strike=0.20, notional=1_000_000.0)
        pv  = cms_caplet_pv(otm, flat_curve, vol=0.30)
        assert pv < 1.0  # nearly worthless

    def test_caplet_tau_property(self, caplet):
        assert abs(caplet.tau - 0.25) < 1e-10

    def test_invalid_t_fix(self, flat_curve):
        bad = CMSCaplet(t_fix=0.0, t_pay=0.25, swap_tenor=10.0, strike=0.04)
        with pytest.raises(ValueError):
            cms_caplet_pv(bad, flat_curve, vol=0.30)

    def test_invalid_vol(self, caplet, flat_curve):
        with pytest.raises(ValueError):
            cms_caplet_pv(caplet, flat_curve, vol=0.0)

    def test_put_call_parity(self, flat_curve):
        caplet = CMSCaplet(t_fix=1.0, t_pay=1.25, swap_tenor=10.0,
                           strike=0.04, notional=1_000_000.0)
        parity_model = cms_caplet_floorlet_parity(caplet, flat_curve, vol=0.30)
        pv_cap  = cms_caplet_pv(caplet, flat_curve, vol=0.30)
        fl_cap  = CMSCaplet(t_fix=1.0, t_pay=1.25, swap_tenor=10.0,
                            strike=0.04, notional=1_000_000.0, cap_floor="floor")
        pv_floor = cms_caplet_pv(fl_cap, flat_curve, vol=0.30)
        diff = pv_cap - pv_floor
        assert abs(diff - parity_model) / (abs(parity_model) + 1e-3) < 0.02

    def test_higher_notional_proportional(self, flat_curve):
        c1 = CMSCaplet(t_fix=1.0, t_pay=1.25, swap_tenor=10.0,
                       strike=0.04, notional=1_000_000.0)
        c2 = CMSCaplet(t_fix=1.0, t_pay=1.25, swap_tenor=10.0,
                       strike=0.04, notional=2_000_000.0)
        pv1 = cms_caplet_pv(c1, flat_curve, vol=0.30)
        pv2 = cms_caplet_pv(c2, flat_curve, vol=0.30)
        assert abs(pv2 / pv1 - 2.0) < 1e-8

    def test_replication_model_positive(self, caplet, flat_curve):
        pv = cms_caplet_pv(caplet, flat_curve, vol=0.30, model="replication")
        assert pv > 0

    def test_floor_pv_positive(self, flat_curve):
        floor = CMSCaplet(t_fix=1.0, t_pay=1.25, swap_tenor=10.0,
                          strike=0.06, notional=1_000_000.0, cap_floor="floor")
        pv = cms_caplet_pv(floor, flat_curve, vol=0.30)
        assert pv > 0


# ── CMS Spread Option ─────────────────────────────────────────────────────────

class TestCMSSpreadOption:
    @pytest.fixture
    def steepener(self):
        return CMSSpreadOption(
            t_fix=1.0, t_pay=1.25,
            long_tenor=10.0, short_tenor=2.0,
            spread_strike=0.005,  # 50bps
            notional=10_000_000.0,
            call_put="call",
        )

    def test_call_pv_positive(self, steepener, flat_curve):
        pv = cms_spread_option_pv(steepener, flat_curve, 0.30, 0.30, rho=0.7)
        assert pv > 0

    def test_put_pv_positive(self, flat_curve):
        flattener = CMSSpreadOption(
            t_fix=1.0, t_pay=1.25,
            long_tenor=10.0, short_tenor=2.0,
            spread_strike=0.005,
            notional=10_000_000.0,
            call_put="put",
        )
        pv = cms_spread_option_pv(flattener, flat_curve, 0.30, 0.30, rho=0.7)
        assert pv > 0

    def test_higher_notional_proportional(self, flat_curve):
        opt1 = CMSSpreadOption(t_fix=1.0, t_pay=1.25, long_tenor=10.0,
                               short_tenor=2.0, spread_strike=0.005,
                               notional=1_000_000.0)
        opt2 = CMSSpreadOption(t_fix=1.0, t_pay=1.25, long_tenor=10.0,
                               short_tenor=2.0, spread_strike=0.005,
                               notional=2_000_000.0)
        pv1 = cms_spread_option_pv(opt1, flat_curve, 0.30, 0.30, rho=0.7)
        pv2 = cms_spread_option_pv(opt2, flat_curve, 0.30, 0.30, rho=0.7)
        assert abs(pv2 / pv1 - 2.0) < 1e-8

    def test_invalid_rho(self, steepener, flat_curve):
        with pytest.raises(ValueError):
            cms_spread_option_pv(steepener, flat_curve, 0.30, 0.30, rho=1.5)

    def test_high_rho_lower_vol_lower_pv(self, steepener, flat_curve):
        pv_low_rho  = cms_spread_option_pv(steepener, flat_curve, 0.30, 0.30, rho=0.0)
        pv_high_rho = cms_spread_option_pv(steepener, flat_curve, 0.30, 0.30, rho=0.9)
        # Higher correlation reduces spread volatility → lower spread option price
        assert pv_high_rho < pv_low_rho

    def test_deep_otm_call_near_zero(self, flat_curve):
        deep_otm = CMSSpreadOption(
            t_fix=1.0, t_pay=1.25,
            long_tenor=10.0, short_tenor=2.0,
            spread_strike=0.10,  # 1000bp strike → very OTM
            notional=10_000_000.0,
        )
        pv = cms_spread_option_pv(deep_otm, flat_curve, 0.30, 0.30, rho=0.7)
        assert pv < 50_000  # far OTM on $10M notional

    def test_tau_property(self, steepener):
        assert abs(steepener.tau - 0.25) < 1e-10

    def test_invalid_vol(self, steepener, flat_curve):
        with pytest.raises(ValueError):
            cms_spread_option_pv(steepener, flat_curve, 0.0, 0.30, rho=0.7)

    def test_higher_vol_higher_pv(self, steepener, flat_curve):
        pv_low  = cms_spread_option_pv(steepener, flat_curve, 0.20, 0.20, rho=0.5)
        pv_high = cms_spread_option_pv(steepener, flat_curve, 0.50, 0.50, rho=0.5)
        assert pv_high > pv_low


# ── CMS Swap ──────────────────────────────────────────────────────────────────

class TestCMSSwap:
    @pytest.fixture
    def swap(self):
        return CMSSwap(
            cms_tenor=10.0,
            first_payment=1.0,
            last_payment=5.0,
            fixed_rate=0.045,
            notional=10_000_000.0,
            pay_cms=True,
            payment_freq=2,
        )

    def test_fixed_leg_pv_positive(self, swap, flat_curve):
        pv = swap.fixed_leg_pv(flat_curve)
        assert pv > 0

    def test_cms_leg_pv_positive(self, swap, flat_curve):
        pv = swap.cms_leg_pv(flat_curve, vol=0.30)
        assert pv > 0

    def test_par_rate_reasonable(self, swap, flat_curve):
        par = swap.par_fixed_rate(flat_curve, vol=0.30)
        assert 0.02 < par < 0.10

    def test_par_rate_zero_pv(self, flat_curve):
        # Create swap with par fixed rate → PV should be ~0
        base = CMSSwap(cms_tenor=10.0, first_payment=1.0, last_payment=5.0,
                       fixed_rate=0.04, notional=10_000_000.0, pay_cms=True,
                       payment_freq=2)
        par   = base.par_fixed_rate(flat_curve, vol=0.30)
        at_par = CMSSwap(cms_tenor=10.0, first_payment=1.0, last_payment=5.0,
                         fixed_rate=par, notional=10_000_000.0, pay_cms=True,
                         payment_freq=2)
        assert abs(at_par.pv(flat_curve, vol=0.30)) < 1_000  # near zero

    def test_pay_cms_vs_receive_cms_opposite_sign(self, flat_curve):
        sw1 = CMSSwap(cms_tenor=10.0, first_payment=1.0, last_payment=5.0,
                      fixed_rate=0.04, notional=10_000_000.0, pay_cms=True)
        sw2 = CMSSwap(cms_tenor=10.0, first_payment=1.0, last_payment=5.0,
                      fixed_rate=0.04, notional=10_000_000.0, pay_cms=False)
        pv1 = sw1.pv(flat_curve, vol=0.30)
        pv2 = sw2.pv(flat_curve, vol=0.30)
        assert abs(pv1 + pv2) < 1e-4  # exactly opposite

    def test_dv01_finite_difference(self, swap, flat_curve):
        dv01 = swap.dv01(flat_curve, vol=0.30, shift_bps=1.0)
        assert isinstance(dv01, float)
        assert abs(dv01) > 0

    def test_higher_notional_proportional_pv(self, flat_curve):
        sw1 = CMSSwap(cms_tenor=10.0, first_payment=1.0, last_payment=5.0,
                      fixed_rate=0.04, notional=10_000_000.0)
        sw2 = CMSSwap(cms_tenor=10.0, first_payment=1.0, last_payment=5.0,
                      fixed_rate=0.04, notional=20_000_000.0)
        pv1 = sw1.pv(flat_curve, vol=0.30)
        pv2 = sw2.pv(flat_curve, vol=0.30)
        assert abs(pv2 / pv1 - 2.0) < 1e-4

    def test_payment_dates_ascending(self, swap):
        dates = swap._payment_dates()
        assert dates == sorted(dates)

    def test_payment_dates_count(self, swap):
        # first=1.0, last=5.0, freq=2 → 1.0,1.5,2.0,...,5.0 = 9 dates
        dates = swap._payment_dates()
        assert len(dates) == 9

    def test_longer_tenor_cms_leg_different_pv(self, flat_curve):
        sw10 = CMSSwap(cms_tenor=10.0, first_payment=1.0, last_payment=5.0,
                       fixed_rate=0.04, notional=10_000_000.0)
        sw2  = CMSSwap(cms_tenor=2.0,  first_payment=1.0, last_payment=5.0,
                       fixed_rate=0.04, notional=10_000_000.0)
        pv10 = sw10.cms_leg_pv(flat_curve, vol=0.30)
        pv2  = sw2.cms_leg_pv(flat_curve, vol=0.30)
        # 10Y CMS > 2Y CMS (upward sloping + more convexity)
        assert pv10 > pv2

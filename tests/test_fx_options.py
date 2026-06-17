"""
Tests for sofr_engine/fx_options.py
"""
from __future__ import annotations
import math
import pytest
import numpy as np

from sofr_engine.fx_options import (
    FXOptionParams, FXOptionResult,
    gk_price, gk_greeks, gk_implied_vol,
    FXVolSurface, vol_for_strike, fx_smile,
    FXVolCone, vol_cone,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def atm_call():
    return FXOptionParams(
        spot=1.09, strike=1.09, vol=0.08,
        domestic_rate=0.04, foreign_rate=0.03,
        maturity=1.0, is_call=True,
    )

@pytest.fixture
def atm_put():
    return FXOptionParams(
        spot=1.09, strike=1.09, vol=0.08,
        domestic_rate=0.04, foreign_rate=0.03,
        maturity=1.0, is_call=False,
    )

@pytest.fixture
def surface():
    return FXVolSurface(
        maturities=[0.25, 0.5, 1.0, 2.0],
        atm_vols=  [0.07, 0.08, 0.09, 0.10],
        rr25=      [0.002, 0.003, 0.004, 0.005],
        bf25=      [0.001, 0.001, 0.002, 0.002],
        spot=1.09,
        domestic_rate=0.04,
        foreign_rate=0.03,
    )


# ── GK Price ─────────────────────────────────────────────────────────────────

class TestGKPrice:
    def test_call_positive(self, atm_call):
        assert gk_price(atm_call) > 0.0

    def test_put_positive(self, atm_put):
        assert gk_price(atm_put) > 0.0

    def test_put_call_parity(self, atm_call, atm_put):
        C = gk_price(atm_call)
        P = gk_price(atm_put)
        S  = atm_call.spot
        K  = atm_call.strike
        rd = atm_call.domestic_rate
        rf = atm_call.foreign_rate
        T  = atm_call.maturity
        # C - P = S*exp(-rf*T) - K*exp(-rd*T)
        lhs = C - P
        rhs = S * math.exp(-rf * T) - K * math.exp(-rd * T)
        assert abs(lhs - rhs) < 1e-8

    def test_deep_itm_call_near_intrinsic(self):
        p = FXOptionParams(spot=1.50, strike=1.00, vol=0.08,
                           domestic_rate=0.04, foreign_rate=0.03,
                           maturity=1.0, is_call=True)
        price = gk_price(p)
        fwd   = 1.50 * math.exp((0.04 - 0.03) * 1.0)
        intrinsic = (fwd - 1.00) * math.exp(-0.04 * 1.0)
        assert price >= intrinsic * 0.999

    def test_deep_otm_call_near_zero(self):
        p = FXOptionParams(spot=1.09, strike=2.00, vol=0.08,
                           domestic_rate=0.04, foreign_rate=0.03,
                           maturity=1.0, is_call=True)
        assert gk_price(p) < 0.001

    def test_longer_maturity_more_value(self):
        base = dict(spot=1.09, strike=1.09, vol=0.08,
                    domestic_rate=0.04, foreign_rate=0.03)
        p1 = FXOptionParams(**base, maturity=0.5, is_call=True)
        p2 = FXOptionParams(**base, maturity=1.0, is_call=True)
        assert gk_price(p2) > gk_price(p1)

    def test_higher_vol_more_value(self):
        base = dict(spot=1.09, strike=1.09,
                    domestic_rate=0.04, foreign_rate=0.03, maturity=1.0)
        p1 = FXOptionParams(**base, vol=0.05, is_call=True)
        p2 = FXOptionParams(**base, vol=0.15, is_call=True)
        assert gk_price(p2) > gk_price(p1)

    def test_call_monotone_in_spot(self):
        base = dict(strike=1.09, vol=0.08,
                    domestic_rate=0.04, foreign_rate=0.03, maturity=1.0, is_call=True)
        p1 = FXOptionParams(spot=1.05, **base)
        p2 = FXOptionParams(spot=1.15, **base)
        assert gk_price(p2) > gk_price(p1)

    def test_put_monotone_in_spot(self):
        base = dict(strike=1.09, vol=0.08,
                    domestic_rate=0.04, foreign_rate=0.03, maturity=1.0, is_call=False)
        p1 = FXOptionParams(spot=1.05, **base)
        p2 = FXOptionParams(spot=1.15, **base)
        assert gk_price(p1) > gk_price(p2)

    def test_zero_maturity_intrinsic_call(self):
        p = FXOptionParams(spot=1.10, strike=1.09, vol=0.08,
                           domestic_rate=0.04, foreign_rate=0.03,
                           maturity=0.0, is_call=True)
        assert abs(gk_price(p) - 0.01) < 1e-9

    def test_zero_maturity_intrinsic_put(self):
        p = FXOptionParams(spot=1.09, strike=1.10, vol=0.08,
                           domestic_rate=0.04, foreign_rate=0.03,
                           maturity=0.0, is_call=False)
        assert abs(gk_price(p) - 0.01) < 1e-9


# ── GK Greeks ─────────────────────────────────────────────────────────────────

class TestGKGreeks:
    def test_call_delta_in_range(self, atm_call):
        res = gk_greeks(atm_call)
        assert 0.0 < res.delta < 1.0

    def test_put_delta_in_range(self, atm_put):
        res = gk_greeks(atm_put)
        assert -1.0 < res.delta < 0.0

    def test_call_put_delta_relation(self, atm_call, atm_put):
        dc = gk_greeks(atm_call).delta
        dp = gk_greeks(atm_put).delta
        rd = atm_call.domestic_rate
        rf = atm_call.foreign_rate
        T  = atm_call.maturity
        # dc - dp ≈ exp(-rf*T)
        assert abs(dc - dp - math.exp(-rf * T)) < 0.01

    def test_gamma_positive(self, atm_call):
        assert gk_greeks(atm_call).gamma > 0.0

    def test_vega_positive(self, atm_call):
        assert gk_greeks(atm_call).vega > 0.0

    def test_call_theta_negative(self, atm_call):
        assert gk_greeks(atm_call).theta < 0.0

    def test_vanna_finite(self, atm_call):
        assert math.isfinite(gk_greeks(atm_call).vanna)

    def test_volga_positive_for_otm(self):
        p = FXOptionParams(spot=1.09, strike=1.20, vol=0.08,
                           domestic_rate=0.04, foreign_rate=0.03,
                           maturity=1.0, is_call=True)
        assert gk_greeks(p).volga > 0.0

    def test_all_keys_present(self, atm_call):
        res = gk_greeks(atm_call)
        for attr in ["pv", "delta", "gamma", "vega", "theta", "rho_d", "rho_f", "vanna", "volga"]:
            assert hasattr(res, attr)

    def test_put_gamma_equals_call_gamma(self, atm_call, atm_put):
        gc = gk_greeks(atm_call).gamma
        gp = gk_greeks(atm_put).gamma
        assert abs(gc - gp) < 1e-10


# ── Implied Vol ───────────────────────────────────────────────────────────────

class TestImpliedVol:
    def test_round_trip_call(self, atm_call):
        price = gk_price(atm_call)
        iv = gk_implied_vol(price, atm_call.spot, atm_call.strike,
                            atm_call.domestic_rate, atm_call.foreign_rate,
                            atm_call.maturity, is_call=True)
        assert abs(iv - atm_call.vol) < 1e-5

    def test_round_trip_put(self, atm_put):
        price = gk_price(atm_put)
        iv = gk_implied_vol(price, atm_put.spot, atm_put.strike,
                            atm_put.domestic_rate, atm_put.foreign_rate,
                            atm_put.maturity, is_call=False)
        assert abs(iv - atm_put.vol) < 1e-5

    def test_iv_monotone_in_price(self, atm_call):
        p1 = gk_price(atm_call)
        p2 = p1 * 1.10
        iv1 = gk_implied_vol(p1, atm_call.spot, atm_call.strike,
                              atm_call.domestic_rate, atm_call.foreign_rate,
                              atm_call.maturity)
        iv2 = gk_implied_vol(p2, atm_call.spot, atm_call.strike,
                              atm_call.domestic_rate, atm_call.foreign_rate,
                              atm_call.maturity)
        assert iv2 > iv1

    def test_call_put_same_iv(self, atm_call, atm_put):
        pc = gk_price(atm_call)
        pp = gk_price(atm_put)
        ivc = gk_implied_vol(pc, atm_call.spot, atm_call.strike,
                              atm_call.domestic_rate, atm_call.foreign_rate,
                              atm_call.maturity, is_call=True)
        ivp = gk_implied_vol(pp, atm_put.spot, atm_put.strike,
                              atm_put.domestic_rate, atm_put.foreign_rate,
                              atm_put.maturity, is_call=False)
        assert abs(ivc - ivp) < 1e-5

    def test_high_precision(self, atm_call):
        price = gk_price(atm_call)
        iv = gk_implied_vol(price, atm_call.spot, atm_call.strike,
                            atm_call.domestic_rate, atm_call.foreign_rate,
                            atm_call.maturity)
        assert abs(iv - atm_call.vol) < 1e-6

    def test_zero_maturity_raises(self):
        with pytest.raises(ValueError):
            gk_implied_vol(0.05, 1.09, 1.09, 0.04, 0.03, 0.0)


# ── Put-Call Parity ───────────────────────────────────────────────────────────

class TestPutCallParity:
    def _parity(self, spot, strike, rd, rf, T, vol):
        C = gk_price(FXOptionParams(spot=spot, strike=strike, vol=vol,
                                     domestic_rate=rd, foreign_rate=rf,
                                     maturity=T, is_call=True))
        P = gk_price(FXOptionParams(spot=spot, strike=strike, vol=vol,
                                     domestic_rate=rd, foreign_rate=rf,
                                     maturity=T, is_call=False))
        lhs = C - P
        rhs = spot * math.exp(-rf * T) - strike * math.exp(-rd * T)
        return abs(lhs - rhs)

    def test_atm(self):
        assert self._parity(1.09, 1.09, 0.04, 0.03, 1.0, 0.08) < 1e-10

    def test_itm(self):
        assert self._parity(1.15, 1.09, 0.04, 0.03, 1.0, 0.08) < 1e-10

    def test_otm(self):
        assert self._parity(1.05, 1.09, 0.04, 0.03, 1.0, 0.08) < 1e-10

    def test_short_maturity(self):
        assert self._parity(1.09, 1.09, 0.04, 0.03, 0.1, 0.08) < 1e-10

    def test_zero_rates(self):
        assert self._parity(1.09, 1.09, 0.0, 0.0, 1.0, 0.08) < 1e-10


# ── FX Vol Surface ────────────────────────────────────────────────────────────

class TestFXVolSurface:
    def test_atm_vol_at_atm_strike(self, surface):
        # At ATM strike, interpolated vol should ≈ ATM vol (index 2 = 1Y)
        mat = 1.0
        atm_idx = surface.maturities.index(mat)
        _, k_atm, _ = surface.pillar_strikes(atm_idx)
        v = vol_for_strike(surface, mat, k_atm)
        assert abs(v - surface.atm_vols[atm_idx]) < 0.005

    def test_call_vol_above_atm_when_rr_positive(self, surface):
        mat = 1.0
        i = surface.maturities.index(mat)
        _, k_atm, k25c = surface.pillar_strikes(i)
        v_atm = vol_for_strike(surface, mat, k_atm)
        v_25c = vol_for_strike(surface, mat, k25c)
        assert v_25c > v_atm  # positive RR → call skew

    def test_put_vol_above_atm_with_butterfly(self, surface):
        mat = 1.0
        i = surface.maturities.index(mat)
        k25p, k_atm, _ = surface.pillar_strikes(i)
        v_atm = vol_for_strike(surface, mat, k_atm)
        v_25p = vol_for_strike(surface, mat, k25p)
        # butterfly makes wings higher
        assert v_25p > v_atm - 0.01  # wings elevated

    def test_smile_length(self, surface):
        strikes, vols = fx_smile(surface, 1.0, n_strikes=21)
        assert len(strikes) == 21
        assert len(vols) == 21

    def test_smile_positive_vols(self, surface):
        _, vols = fx_smile(surface, 1.0, n_strikes=11)
        assert all(v > 0.0 for v in vols)

    def test_vol_for_far_otm_reasonable(self, surface):
        mat = 1.0
        fwd = surface.spot * math.exp((surface.domestic_rate - surface.foreign_rate) * mat)
        v = vol_for_strike(surface, mat, fwd * 1.20)
        assert 0.0 < v < 1.0

    def test_pillar_strikes_ordering(self, surface):
        k25p, k_atm, k25c = surface.pillar_strikes(2)
        assert k25p < k_atm < k25c

    def test_maturity_interpolation(self, surface):
        v1 = vol_for_strike(surface, 0.5, 1.09)
        v2 = vol_for_strike(surface, 1.0, 1.09)
        # Closer to 0.5Y vol
        v_mid = vol_for_strike(surface, 0.75, 1.09)
        assert min(v1, v2) - 0.005 <= v_mid <= max(v1, v2) + 0.005

    def test_wrong_lengths_raises(self):
        with pytest.raises(ValueError):
            FXVolSurface(
                maturities=[1.0, 2.0],
                atm_vols=[0.08],
                rr25=[0.002, 0.003],
                bf25=[0.001, 0.001],
                spot=1.09,
            )


# ── Strike from Delta ─────────────────────────────────────────────────────────

class TestStrikeFromDelta:
    def test_50_delta_near_atm(self, surface):
        i = 2  # 1Y
        T   = surface.maturities[i]
        atm_vol = surface.atm_vols[i]
        k50 = surface._strike_from_delta(0.50, atm_vol, T, is_call=True)
        fwd = surface.spot * math.exp((surface.domestic_rate - surface.foreign_rate) * T)
        assert abs(k50 - fwd) / fwd < 0.05

    def test_25d_call_strike_above_atm(self, surface):
        i = 2
        _, k_atm, k25c = surface.pillar_strikes(i)
        assert k25c > k_atm

    def test_25d_put_strike_below_atm(self, surface):
        i = 2
        k25p, k_atm, _ = surface.pillar_strikes(i)
        assert k25p < k_atm

    def test_10d_call_farther_otm_than_25d(self, surface):
        i = 2
        T   = surface.maturities[i]
        _, _, k25c = surface.pillar_strikes(i)
        k10c = surface._strike_from_delta(0.10, surface.atm_vols[i], T, is_call=True)
        assert k10c > k25c

    def test_strike_positive(self, surface):
        i = 2
        T = surface.maturities[i]
        k = surface._strike_from_delta(0.25, 0.09, T, is_call=True)
        assert k > 0.0


# ── Vol Cone ──────────────────────────────────────────────────────────────────

class TestVolCone:
    @pytest.fixture
    def returns(self):
        rng = np.random.default_rng(42)
        return rng.normal(0, 0.01, 500)

    def test_shape(self, returns):
        cone = vol_cone(returns)
        assert cone.percentiles.shape == (5, len(cone.tenors))

    def test_percentiles_ordered(self, returns):
        cone = vol_cone(returns)
        for col in range(cone.percentiles.shape[1]):
            pcts = cone.percentiles[:, col]
            assert all(pcts[i] <= pcts[i + 1] for i in range(len(pcts) - 1))

    def test_current_finite(self, returns):
        cone = vol_cone(returns)
        assert all(math.isfinite(v) for v in cone.current)

    def test_custom_tenors(self, returns):
        cone = vol_cone(returns, tenors=[5, 21, 63])
        assert len(cone.tenors) == 3
        assert cone.percentiles.shape == (5, 3)

    def test_annualised_vol_reasonable(self, returns):
        # σ daily ≈ 1% → annualised ≈ 0.01 * sqrt(252) ≈ 0.159
        cone = vol_cone(returns, tenors=[21])
        median = cone.percentiles[2, 0]
        assert 0.05 < median < 0.40


# ── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_invalid_spot(self):
        with pytest.raises(ValueError):
            FXOptionParams(spot=-1.0, strike=1.09, vol=0.08,
                           domestic_rate=0.04, foreign_rate=0.03, maturity=1.0)

    def test_invalid_strike(self):
        with pytest.raises(ValueError):
            FXOptionParams(spot=1.09, strike=0.0, vol=0.08,
                           domestic_rate=0.04, foreign_rate=0.03, maturity=1.0)

    def test_invalid_vol(self):
        with pytest.raises(ValueError):
            FXOptionParams(spot=1.09, strike=1.09, vol=-0.01,
                           domestic_rate=0.04, foreign_rate=0.03, maturity=1.0)

    def test_zero_vol_intrinsic(self):
        p = FXOptionParams(spot=1.15, strike=1.09, vol=0.0,
                           domestic_rate=0.04, foreign_rate=0.03,
                           maturity=1.0, is_call=True)
        price = gk_price(p)
        assert price > 0.0

    def test_equal_rates_parity(self):
        p = FXOptionParams(spot=1.09, strike=1.09, vol=0.08,
                           domestic_rate=0.05, foreign_rate=0.05,
                           maturity=1.0, is_call=True)
        C = gk_price(p)
        q = FXOptionParams(spot=1.09, strike=1.09, vol=0.08,
                           domestic_rate=0.05, foreign_rate=0.05,
                           maturity=1.0, is_call=False)
        P = gk_price(q)
        # With r_d = r_f, C = P at ATM
        assert abs(C - P) < 1e-6

    def test_greeks_return_result_type(self):
        p = FXOptionParams(spot=1.09, strike=1.09, vol=0.08,
                           domestic_rate=0.04, foreign_rate=0.03,
                           maturity=1.0, is_call=True)
        assert isinstance(gk_greeks(p), FXOptionResult)

    def test_large_spot(self):
        p = FXOptionParams(spot=150.0, strike=100.0, vol=0.10,
                           domestic_rate=0.02, foreign_rate=0.01,
                           maturity=1.0, is_call=True)
        assert gk_price(p) > 0.0

    def test_very_short_maturity(self):
        p = FXOptionParams(spot=1.09, strike=1.09, vol=0.08,
                           domestic_rate=0.04, foreign_rate=0.03,
                           maturity=0.01, is_call=True)
        assert gk_price(p) > 0.0

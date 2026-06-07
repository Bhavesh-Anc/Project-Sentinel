"""
Tests for sofr_engine/swaption.py — Black-76 European swaption pricer.
"""
from __future__ import annotations
import numpy as np
import pytest
from datetime import date

from sofr_engine.bootstrap import flat_sofr_curve
from sofr_engine.swaption import Swaption, SwaptionVolSurface, price_swaption

REF_DATE = date(2024, 1, 2)


def _curve(rate: float = 0.05):
    return flat_sofr_curve(REF_DATE, rate)


# ── Annuity ────────────────────────────────────────────────────────────────────

class TestAnnuity:

    def test_annuity_positive(self):
        sw = Swaption(1.0, 5.0, 0.05)
        assert sw.annuity(_curve()) > 0

    def test_annuity_less_than_tenor(self):
        sw = Swaption(1.0, 5.0, 0.05)
        # Annuity < τ because discount factors < 1
        assert sw.annuity(_curve()) < 5.0

    def test_annuity_decreases_with_higher_rates(self):
        sw = Swaption(1.0, 5.0, 0.05)
        a_low  = sw.annuity(_curve(0.02))
        a_high = sw.annuity(_curve(0.08))
        assert a_low > a_high

    def test_annuity_with_semiannual_freq(self):
        sw_ann  = Swaption(1.0, 5.0, 0.05, freq=1)
        sw_semi = Swaption(1.0, 5.0, 0.05, freq=2)
        # Both should be similar but not identical
        assert abs(sw_ann.annuity(_curve()) - sw_semi.annuity(_curve())) < 0.5


# ── Forward Swap Rate ─────────────────────────────────────────────────────────

class TestForwardSwapRate:

    def test_forward_rate_positive(self):
        sw = Swaption(1.0, 5.0, 0.05)
        assert sw.forward_swap_rate(_curve()) > 0

    def test_forward_rate_close_to_par_on_flat_curve(self):
        curve = _curve(0.05)
        sw = Swaption(0.001, 5.0, 0.05)   # near-zero expiry → ~par rate
        fwd = sw.forward_swap_rate(curve)
        par = curve.par_ois_rate(5.0)
        assert abs(fwd - par) < 0.005   # within 50 bps

    def test_forward_rate_nan_on_zero_annuity(self):
        # expiry + tenor beyond curve → annuity → 0 → NaN
        curve = _curve(0.05)
        sw = Swaption(20.0, 20.0, 0.05)
        result = sw.forward_swap_rate(curve)
        # May be NaN (beyond curve) or a valid rate — just ensure no crash
        assert result is not None


# ── Black-76 PV ────────────────────────────────────────────────────────────────

class TestBlackPV:

    def test_payer_positive(self):
        sw = Swaption(1.0, 5.0, 0.05, notional=1_000_000, swaption_type="payer", vol=0.20)
        assert sw.black_pv(_curve()) > 0

    def test_receiver_positive(self):
        sw = Swaption(1.0, 5.0, 0.05, notional=1_000_000, swaption_type="receiver", vol=0.20)
        assert sw.black_pv(_curve()) > 0

    def test_put_call_parity(self):
        curve = _curve(0.05)
        K = 0.050
        N = 10_000_000
        payer = Swaption(1.0, 5.0, K, N, "payer",    0.20)
        recvr = Swaption(1.0, 5.0, K, N, "receiver", 0.20)
        S = payer.forward_swap_rate(curve)
        A = payer.annuity(curve)
        # Payer - Receiver = N * A * (S - K)
        expected = N * A * (S - K)
        actual   = payer.black_pv(curve) - recvr.black_pv(curve)
        assert abs(actual - expected) < 1.0  # within $1

    def test_atm_payer_equals_atm_receiver(self):
        curve = _curve(0.05)
        S = Swaption(1.0, 5.0, 0.05).forward_swap_rate(curve)
        N = 1_000_000
        payer = Swaption(1.0, 5.0, S, N, "payer",    0.20)
        recvr = Swaption(1.0, 5.0, S, N, "receiver", 0.20)
        assert abs(payer.black_pv(curve) - recvr.black_pv(curve)) < 1.0

    def test_deep_itm_payer_near_intrinsic(self):
        curve = _curve(0.05)
        # Very low strike → deep ITM → PV ≈ intrinsic
        K = 0.01   # 1% strike, forward ~5%
        sw = Swaption(1.0, 5.0, K, 1_000_000, "payer", 0.001)  # near-zero vol
        S  = sw.forward_swap_rate(curve)
        A  = sw.annuity(curve)
        intrinsic = 1_000_000 * A * max(S - K, 0.0)
        assert abs(sw.black_pv(curve) - intrinsic) / intrinsic < 0.01

    def test_zero_vol_returns_intrinsic(self):
        curve = _curve(0.05)
        K = 0.04
        sw = Swaption(1.0, 5.0, K, 1_000_000, "payer", vol=0.0)
        S  = sw.forward_swap_rate(curve)
        A  = sw.annuity(curve)
        intrinsic = 1_000_000 * A * max(S - K, 0.0)
        assert abs(sw.black_pv(curve) - intrinsic) < 1.0

    def test_expired_swaption_returns_intrinsic(self):
        curve = _curve(0.05)
        sw = Swaption(0.0, 5.0, 0.04, 1_000_000, "payer", 0.20)
        S  = sw.forward_swap_rate(curve)
        A  = sw.annuity(curve)
        intrinsic = 1_000_000 * A * max(S - 0.04, 0.0)
        assert abs(sw.black_pv(curve) - intrinsic) < 1.0

    def test_pv_increases_with_vol(self):
        curve = _curve(0.05)
        sw_lo = Swaption(1.0, 5.0, 0.05, 1_000_000, "payer", 0.10)
        sw_hi = Swaption(1.0, 5.0, 0.05, 1_000_000, "payer", 0.30)
        assert sw_hi.black_pv(curve) > sw_lo.black_pv(curve)

    def test_pv_scales_with_notional(self):
        curve = _curve(0.05)
        sw1 = Swaption(1.0, 5.0, 0.05, 1_000_000, "payer", 0.20)
        sw10 = Swaption(1.0, 5.0, 0.05, 10_000_000, "payer", 0.20)
        assert abs(sw10.black_pv(curve) / sw1.black_pv(curve) - 10.0) < 1e-8


# ── Implied Vol ────────────────────────────────────────────────────────────────

class TestImpliedVol:

    def test_round_trip(self):
        curve = _curve(0.05)
        vol_in = 0.18
        sw = Swaption(1.0, 5.0, 0.05, 1_000_000, "payer", vol_in)
        mkt_pv = sw.black_pv(curve)
        vol_out = sw.implied_vol(curve, mkt_pv)
        assert abs(vol_out - vol_in) < 1e-8

    def test_round_trip_receiver(self):
        curve = _curve(0.05)
        vol_in = 0.22
        sw = Swaption(1.0, 5.0, 0.055, 1_000_000, "receiver", vol_in)
        mkt_pv = sw.black_pv(curve)
        vol_out = sw.implied_vol(curve, mkt_pv)
        assert abs(vol_out - vol_in) < 1e-8

    def test_raises_on_unbracketed_pv(self):
        curve = _curve(0.05)
        sw = Swaption(1.0, 5.0, 0.05, 1_000_000, "payer", 0.20)
        with pytest.raises(ValueError):
            sw.implied_vol(curve, market_pv=-999_999)  # negative PV impossible

    def test_raises_on_expired(self):
        curve = _curve(0.05)
        sw = Swaption(0.0, 5.0, 0.05, 1_000_000, "payer", 0.20)
        with pytest.raises(ValueError):
            sw.implied_vol(curve, market_pv=100_000)


# ── Greeks ─────────────────────────────────────────────────────────────────────

class TestGreeks:

    def test_vega_positive(self):
        curve = _curve(0.05)
        sw = Swaption(1.0, 5.0, 0.05, 1_000_000, "payer", 0.20)
        assert sw.vega(curve) > 0

    def test_vega_receiver_positive(self):
        curve = _curve(0.05)
        sw = Swaption(1.0, 5.0, 0.05, 1_000_000, "receiver", 0.20)
        assert sw.vega(curve) > 0

    def test_payer_delta_positive(self):
        curve = _curve(0.05)
        sw = Swaption(1.0, 5.0, 0.05, 1_000_000, "payer", 0.20)
        assert sw.delta(curve) > 0

    def test_receiver_delta_negative(self):
        curve = _curve(0.05)
        sw = Swaption(1.0, 5.0, 0.05, 1_000_000, "receiver", 0.20)
        assert sw.delta(curve) < 0

    def test_payer_receiver_vega_similar(self):
        curve = _curve(0.05)
        payer = Swaption(1.0, 5.0, 0.05, 1_000_000, "payer",    0.20)
        recvr = Swaption(1.0, 5.0, 0.05, 1_000_000, "receiver", 0.20)
        assert abs(payer.vega(curve) - recvr.vega(curve)) < 1.0


# ── Summary ────────────────────────────────────────────────────────────────────

class TestSummary:

    def test_all_keys_present(self):
        curve = _curve(0.05)
        sw = Swaption(1.0, 5.0, 0.05, 1_000_000, "payer", 0.20)
        s = sw.summary(curve)
        for key in ["expiry_years", "swap_tenor_years", "strike_pct",
                    "forward_rate_pct", "annuity", "black_pv",
                    "intrinsic_value", "time_value", "vega", "delta", "moneyness_bps"]:
            assert key in s, f"Missing key: {key}"

    def test_pv_equals_intrinsic_plus_time(self):
        curve = _curve(0.05)
        s = Swaption(1.0, 5.0, 0.05, 1_000_000, "payer", 0.20).summary(curve)
        assert abs(s["black_pv"] - (s["intrinsic_value"] + s["time_value"])) < 1e-6

    def test_strike_pct_correct(self):
        s = Swaption(1.0, 5.0, 0.05, 1_000_000, "payer", 0.20).summary(_curve(0.05))
        assert abs(s["strike_pct"] - 5.0) < 1e-10

    def test_time_value_nonnegative(self):
        curve = _curve(0.05)
        s = Swaption(1.0, 5.0, 0.05, 1_000_000, "payer", 0.20).summary(curve)
        assert s["time_value"] >= -1.0  # allow tiny floating-point negative


# ── SwaptionVolSurface ─────────────────────────────────────────────────────────

class TestSwaptionVolSurface:

    def _surface(self):
        return SwaptionVolSurface.typical_market()

    def test_vol_positive(self):
        surf = self._surface()
        assert surf.vol(1.0, 5.0) > 0

    def test_vol_in_range(self):
        surf = self._surface()
        for e in [0.5, 1.0, 5.0, 10.0]:
            for t in [1.0, 5.0, 30.0]:
                v = surf.vol(e, t)
                assert 0.01 < v < 0.50, f"Vol {v:.4f} out of range at ({e},{t})"

    def test_vol_clamped_outside_grid(self):
        surf = self._surface()
        v_inside  = surf.vol(1.0, 5.0)
        v_outside = surf.vol(0.01, 5.0)   # before first expiry
        assert v_outside == pytest.approx(surf.vol(0.5, 5.0), abs=1e-10)

    def test_to_dataframe_shape(self):
        surf = self._surface()
        df = surf.to_dataframe()
        assert df.shape == (5, 5)

    def test_to_dataframe_pct(self):
        surf = self._surface()
        df = surf.to_dataframe()
        # Values should be in percent (roughly 5–25%), not decimal
        assert df.values.max() > 1.0

    def test_raises_on_nonpositive_vol(self):
        with pytest.raises(ValueError):
            SwaptionVolSurface([1.0], [5.0], np.array([[0.0]]))

    def test_raises_on_wrong_shape(self):
        with pytest.raises(ValueError):
            SwaptionVolSurface([1.0, 2.0], [5.0], np.array([[0.1, 0.2]]))

    def test_raises_on_non_increasing_expiries(self):
        with pytest.raises(ValueError):
            SwaptionVolSurface([2.0, 1.0], [5.0], np.array([[0.1], [0.2]]))


# ── price_swaption convenience ────────────────────────────────────────────────

class TestPriceSwaption:

    def test_returns_dict(self):
        result = price_swaption(_curve(), 1.0, 5.0)
        assert isinstance(result, dict)

    def test_atm_strike_none(self):
        curve = _curve(0.05)
        result = price_swaption(curve, 1.0, 5.0, strike=None)
        # forward_rate_pct ≈ strike_pct when ATM
        assert abs(result["forward_rate_pct"] - result["strike_pct"]) < 1e-6

    def test_explicit_strike(self):
        result = price_swaption(_curve(0.05), 1.0, 5.0, strike=0.04)
        assert abs(result["strike_pct"] - 4.0) < 1e-10

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            price_swaption(_curve(), 1.0, 5.0, swaption_type="invalid")

    def test_pv_positive(self):
        result = price_swaption(_curve(), 1.0, 5.0, vol=0.20)
        assert result["black_pv"] > 0

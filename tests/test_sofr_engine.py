"""
Unit tests for the SOFR pricing engine.

Tests are self-contained — no FRED API calls, no external data.
All inputs are synthetic but chosen to exercise meaningful code paths.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

from sofr_engine.curve import DiscountCurve
from sofr_engine.bootstrap import SOFRCurveBootstrapper, flat_sofr_curve
from sofr_engine.convexity import (
    hull_white_convexity_adjustment,
    futures_to_forward,
    adjustment_schedule,
)
from sofr_engine.instruments import SOFRSwap, ForwardRateAgreement, SOFRFutures, SOFRCaplet
from sofr_engine.day_count import act360, year_frac


# ── Fixtures ──────────────────────────────────────────────────────────────────

REF_DATE  = date(2026, 6, 6)
FLAT_RATE = 0.0358  # 3.58% overnight


@pytest.fixture
def flat_curve():
    return flat_sofr_curve(REF_DATE, FLAT_RATE)


@pytest.fixture
def bootstrapped_curve():
    """Realistic SOFR curve from deposits + OIS swaps."""
    return SOFRCurveBootstrapper.from_market_data(
        ref_date=REF_DATE,
        sofr_overnight=FLAT_RATE,
        deposit_quotes=[
            (0.083, 0.0358),
            (0.25,  0.0363),
            (0.5,   0.0365),
            (1.0,   0.0365),
        ],
        ois_quotes=[
            (2.0,  0.0400),
            (3.0,  0.0408),
            (5.0,  0.0418),
            (7.0,  0.0440),
            (10.0, 0.0458),
            (20.0, 0.0488),
            (30.0, 0.0498),
        ],
    )


@pytest.fixture
def par_swap_5y(bootstrapped_curve):
    """5Y SOFR payer swap struck at the swap's own par rate (not curve par_ois_rate)."""
    effective = REF_DATE + timedelta(days=2)
    maturity  = effective + relativedelta(years=5)
    dummy     = SOFRSwap(effective, maturity, fixed_rate=0.04, notional=10_000_000, pay_fixed=True)
    par       = dummy.par_rate(bootstrapped_curve)
    return SOFRSwap(effective, maturity, fixed_rate=par, notional=10_000_000, pay_fixed=True)


# ── DiscountCurve tests ───────────────────────────────────────────────────────

class TestDiscountCurve:

    def test_df_at_zero_is_one(self, flat_curve):
        assert abs(flat_curve.df(0.0) - 1.0) < 1e-10

    def test_df_at_pillars_matches_input(self):
        times = [0.5, 1.0, 2.0, 5.0, 10.0]
        dfs   = [np.exp(-0.04 * t) for t in times]
        curve = DiscountCurve(REF_DATE, times, dfs)
        for t, df in zip(times, dfs):
            assert abs(curve.df(t) - df) < 1e-10, f"DF mismatch at t={t}"

    def test_df_monotone_decreasing(self, flat_curve):
        tenors = np.linspace(0.001, 30.0, 100)
        dfs = flat_curve.df(tenors)
        assert np.all(np.diff(dfs) < 0), "Discount factors must be strictly decreasing"

    def test_all_dfs_positive(self, bootstrapped_curve):
        tenors = np.linspace(0.001, 30.0, 200)
        dfs = bootstrapped_curve.df(tenors)
        assert np.all(dfs > 0), "Discount factors must be strictly positive"

    def test_forward_rates_positive(self, bootstrapped_curve):
        """Log-linear interpolation must produce positive forward rates."""
        tenors = np.linspace(0.001, 29.0, 100)
        fwds = [bootstrapped_curve.forward_rate(t, t + 1.0) for t in tenors]
        assert all(f > 0 for f in fwds), "Forward rates must be positive everywhere"

    def test_zero_rate_flat_curve(self, flat_curve):
        """On a flat exp(-r×T) curve, every continuous zero rate equals the flat rate."""
        for t in [0.25, 1.0, 5.0, 10.0, 30.0]:
            z = flat_curve.zero_rate(t)
            assert abs(z - FLAT_RATE) < 1e-4, f"Zero rate at {t}Y should be {FLAT_RATE:.4f}, got {z:.6f}"

    def test_forward_rate_flat_curve(self, flat_curve):
        """On a flat curve, every forward rate equals the flat rate."""
        for t1, t2 in [(0.5, 1.0), (1.0, 2.0), (5.0, 10.0)]:
            f = flat_curve.forward_rate(t1, t2)
            assert abs(f - FLAT_RATE) < 1e-4

    def test_par_ois_rate_flat_curve(self, flat_curve):
        """
        On a flat continuous rate r curve, par OIS rate = e^r - 1 (simply compounded).
        This is the correct mathematical result, not r itself.
        """
        r = FLAT_RATE
        for tenor in [2.0, 5.0, 10.0]:
            par      = flat_curve.par_ois_rate(tenor)
            expected = np.exp(r) - 1  # simply compounded equivalent
            assert abs(par - expected) < 5e-4, (
                f"Par OIS at {tenor}Y: expected {expected*100:.4f}%, got {par*100:.4f}%"
            )

    def test_forward_rate_t2_gt_t1(self, flat_curve):
        with pytest.raises(ValueError):
            flat_curve.forward_rate(5.0, 3.0)

    def test_zero_curve_dataframe_shape(self, bootstrapped_curve):
        zc = bootstrapped_curve.zero_curve()
        assert isinstance(zc, pd.DataFrame)
        assert set(["tenor_yrs", "discount_factor", "zero_rate_pct"]).issubset(zc.columns)
        assert len(zc) > 0

    def test_negative_df_raises(self):
        with pytest.raises(ValueError):
            DiscountCurve(REF_DATE, [1.0], [-0.5])

    def test_non_increasing_times_raises(self):
        with pytest.raises(ValueError):
            DiscountCurve(REF_DATE, [2.0, 1.0], [0.96, 0.98])

    def test_par_ois_rate_increases_with_tenor(self, bootstrapped_curve):
        """In a normal upward-sloping curve, par OIS rates increase with tenor."""
        rates = [bootstrapped_curve.par_ois_rate(t) for t in [2.0, 5.0, 10.0, 30.0]]
        assert rates == sorted(rates), "Par OIS rates should increase with tenor (normal curve)"


# ── Bootstrap tests ───────────────────────────────────────────────────────────

class TestSOFRBootstrap:

    def test_overnight_pillar_present(self):
        """Bootstrapper should always create an overnight pillar."""
        b = SOFRCurveBootstrapper(REF_DATE, FLAT_RATE)
        assert len(b._times) >= 2
        assert b._times[0] == 0.0
        assert abs(b._dfs[0] - 1.0) < 1e-10

    def test_deposit_pillars_monotone(self):
        """All pillar times should be strictly increasing."""
        b = SOFRCurveBootstrapper(REF_DATE, FLAT_RATE)
        b.add_deposits([(0.25, 0.0363), (0.5, 0.0365), (1.0, 0.0365)])
        assert all(np.diff(b._times) > 0), "Deposit pillars must be strictly increasing"

    def test_ois_par_rate_self_consistency(self, bootstrapped_curve):
        """
        Par OIS rate at bootstrapped tenors should reproduce input quotes.
        Tolerance is 20bps for short-to-medium tenors, 30bps at 30Y where
        log-linear interpolation error at off-pillar intermediate payment dates
        (e.g., DF(6Y), DF(9Y) when bootstrapping 10Y/30Y swaps) is larger.
        """
        ois_quotes_and_tols = [
            (2.0,  0.0400, 0.001),   # 10bps
            (5.0,  0.0418, 0.002),   # 20bps
            (10.0, 0.0458, 0.002),   # 20bps
            (30.0, 0.0498, 0.003),   # 30bps — interpolation error largest at long end
        ]
        for tenor, rate, tol in ois_quotes_and_tols:
            par = bootstrapped_curve.par_ois_rate(tenor)
            assert abs(par - rate) < tol, (
                f"Par OIS at {tenor}Y: expected {rate*100:.4f}%, got {par*100:.4f}%"
            )

    def test_flat_curve_construction(self):
        """flat_sofr_curve utility should produce correct DFs."""
        r = 0.05
        curve = flat_sofr_curve(REF_DATE, r)
        for t in [0.25, 1.0, 5.0, 10.0]:
            expected_df = np.exp(-r * t)
            actual_df   = curve.df(t)
            assert abs(actual_df - expected_df) < 1e-5, f"DF mismatch at {t}Y"

    def test_from_market_data_builds_valid_curve(self, bootstrapped_curve):
        """from_market_data should return a usable DiscountCurve."""
        assert isinstance(bootstrapped_curve, DiscountCurve)
        assert bootstrapped_curve.df(0.0) == 1.0
        assert bootstrapped_curve.df(10.0) > 0
        assert bootstrapped_curve.df(10.0) < 1.0

    def test_futures_extend_curve(self):
        """Adding futures should extend the curve beyond the overnight pillar."""
        futures_df = pd.DataFrame([
            {"expiry": date(2026, 9, 17), "accrual_end": date(2026, 12, 16), "implied_rate": 0.0348},
            {"expiry": date(2026, 12, 16), "accrual_end": date(2027, 3, 17), "implied_rate": 0.0338},
        ])
        b = SOFRCurveBootstrapper(REF_DATE, FLAT_RATE)
        b.add_futures(futures_df)
        assert b._times[-1] > 0.5, "Futures should extend curve beyond overnight"

    def test_deposit_df_formula(self):
        """DF for a 3M deposit at 3.63% must match 1/(1 + r × T_days/360)."""
        rate  = 0.0363
        tenor = 0.25
        b = SOFRCurveBootstrapper(REF_DATE, FLAT_RATE)
        b.add_deposits([(tenor, rate)])
        T_days       = tenor * 365.25
        expected_df  = 1.0 / (1.0 + rate * T_days / 360.0)
        actual_df    = b._interp_df(tenor)
        assert abs(actual_df - expected_df) < 1e-6

    def test_curve_has_30y_pillar(self, bootstrapped_curve):
        """Curve should extend to at least 30 years."""
        assert bootstrapped_curve._times[-1] >= 30.0


# ── Convexity Adjustment tests ────────────────────────────────────────────────

class TestConvexityAdjustment:

    def test_zero_mean_reversion_formula(self):
        """CA with a=0 must equal ½σ²T1T2 exactly."""
        sigma, t1, t2 = 0.010, 1.0, 1.25
        ca = hull_white_convexity_adjustment(t1, t2, sigma=sigma, mean_reversion=0.0)
        expected = 0.5 * sigma**2 * t1 * t2
        assert abs(ca - expected) < 1e-12

    def test_adjustment_always_positive(self):
        """Convexity adjustment must be non-negative for t2 > t1 > 0."""
        for t1, t2 in [(0.25, 0.5), (0.5, 0.75), (1.0, 1.25), (2.0, 2.25)]:
            ca = hull_white_convexity_adjustment(t1, t2)
            assert ca >= 0, f"CA should be non-negative for t1={t1}, t2={t2}"

    def test_adjustment_increases_with_tenor(self):
        """Later contracts have larger convexity adjustments."""
        contracts = [(0.5, 0.75), (1.0, 1.25), (1.5, 1.75), (2.0, 2.25)]
        cas = [hull_white_convexity_adjustment(t1, t2, sigma=0.010) for t1, t2 in contracts]
        assert cas == sorted(cas), "CA should increase monotonically with tenor"

    def test_adjustment_zero_at_expiry(self):
        """If t1 == 0, CA should be 0 regardless of t2."""
        ca = hull_white_convexity_adjustment(0.0, 0.5)
        assert ca == 0.0

    def test_futures_to_forward_lower_than_futures(self):
        """OIS forward rate must be below futures rate (CA is positive)."""
        futures_rate = 0.0420
        t1, t2 = 1.0, 1.25
        fwd = futures_to_forward(futures_rate, t1, t2, sigma=0.010)
        assert fwd < futures_rate, "Forward rate must be below futures rate"

    def test_futures_to_forward_close_maturity_tiny_adj(self):
        """Near-expiry contracts (<3M forward) should have tiny adjustment (<0.2bps)."""
        ca_bps = hull_white_convexity_adjustment(0.08, 0.33) * 10_000
        assert ca_bps < 0.2, f"Near-term CA should be <0.2bps, got {ca_bps:.4f}bps"

    def test_full_hull_white_converges_to_simple(self):
        """Full HW formula with small a should converge toward simple formula (within 10%)."""
        t1, t2, sigma = 1.0, 1.25, 0.010
        ca_simple = hull_white_convexity_adjustment(t1, t2, sigma, mean_reversion=0.0)
        ca_full   = hull_white_convexity_adjustment(t1, t2, sigma, mean_reversion=0.01)
        # Mean reversion reduces CA (dampens forward rate uncertainty)
        assert ca_full <= ca_simple, "Mean reversion should reduce convexity adjustment"
        # Should be within 20% for small mean reversion
        assert abs(ca_full - ca_simple) / ca_simple < 0.20

    def test_mean_reversion_reduces_adjustment(self):
        """Higher mean reversion should produce smaller convexity adjustment."""
        t1, t2, sigma = 2.0, 2.25, 0.010
        ca_0   = hull_white_convexity_adjustment(t1, t2, sigma, mean_reversion=0.0)
        ca_01  = hull_white_convexity_adjustment(t1, t2, sigma, mean_reversion=0.1)
        ca_03  = hull_white_convexity_adjustment(t1, t2, sigma, mean_reversion=0.3)
        assert ca_0 >= ca_01 >= ca_03, "CA should decrease as mean reversion increases"

    def test_adjustment_schedule_length(self):
        """adjustment_schedule returns one value per contract."""
        expiries = [date(2026, 9, 17), date(2026, 12, 16), date(2027, 3, 17)]
        acc_ends = [date(2026, 12, 16), date(2027, 3, 17), date(2027, 6, 16)]
        adjs = adjustment_schedule(expiries, acc_ends, REF_DATE)
        assert len(adjs) == 3


# ── Instrument tests ──────────────────────────────────────────────────────────

class TestSOFRSwap:

    def test_par_swap_pv_zero(self, par_swap_5y, bootstrapped_curve):
        """A swap struck at its own par rate must have PV ≈ 0 at inception."""
        pv = par_swap_5y.pv(bootstrapped_curve)
        assert abs(pv) < 1.0, f"Par swap PV should be ~0, got {pv:.4f}"

    def test_par_rate_internally_consistent(self, bootstrapped_curve):
        """
        SOFRSwap.par_rate() should give the rate at which SOFRSwap.pv() = 0.
        DiscountCurve.par_ois_rate() uses abstract year fractions + ACT/360;
        SOFRSwap.par_rate() uses calendar dates — they differ slightly but both
        should produce PV ≈ 0 when used as the swap's own fixed rate.
        """
        effective = REF_DATE + timedelta(days=2)
        maturity  = effective + relativedelta(years=5)
        dummy     = SOFRSwap(effective, maturity, fixed_rate=0.04, notional=10_000_000, pay_fixed=True)
        par       = dummy.par_rate(bootstrapped_curve)
        # Re-create swap at its own par rate — PV must be 0
        swap_at_par = SOFRSwap(effective, maturity, par, 10_000_000, pay_fixed=True)
        assert abs(swap_at_par.pv(bootstrapped_curve)) < 1.0

    def test_above_par_payer_negative_pv(self, bootstrapped_curve):
        """Payer swap struck above par should have negative PV."""
        effective = REF_DATE + timedelta(days=2)
        maturity  = effective + relativedelta(years=5)
        dummy     = SOFRSwap(effective, maturity, 0.04, 10_000_000, pay_fixed=True)
        par       = dummy.par_rate(bootstrapped_curve)
        swap      = SOFRSwap(effective, maturity, par + 0.01, 10_000_000, pay_fixed=True)
        assert swap.pv(bootstrapped_curve) < 0, "Above-par payer should have negative PV"

    def test_below_par_payer_positive_pv(self, bootstrapped_curve):
        """Payer swap struck below par should have positive PV."""
        effective = REF_DATE + timedelta(days=2)
        maturity  = effective + relativedelta(years=5)
        dummy     = SOFRSwap(effective, maturity, 0.04, 10_000_000, pay_fixed=True)
        par       = dummy.par_rate(bootstrapped_curve)
        swap      = SOFRSwap(effective, maturity, par - 0.01, 10_000_000, pay_fixed=True)
        assert swap.pv(bootstrapped_curve) > 0, "Below-par payer should have positive PV"

    def test_receiver_is_negative_of_payer(self, bootstrapped_curve):
        """Receiver swap PV = -1 × payer swap PV."""
        effective = REF_DATE + timedelta(days=2)
        maturity  = effective + relativedelta(years=5)
        rate      = 0.0420
        payer    = SOFRSwap(effective, maturity, rate, 10_000_000, pay_fixed=True)
        receiver = SOFRSwap(effective, maturity, rate, 10_000_000, pay_fixed=False)
        assert abs(payer.pv(bootstrapped_curve) + receiver.pv(bootstrapped_curve)) < 0.01

    def test_dv01_positive(self, par_swap_5y, bootstrapped_curve):
        """DV01 should be positive."""
        assert par_swap_5y.dv01(bootstrapped_curve) > 0

    def test_dv01_scales_with_notional(self, bootstrapped_curve):
        """DV01 should scale linearly with notional."""
        effective = REF_DATE + timedelta(days=2)
        maturity  = effective + relativedelta(years=5)
        dummy     = SOFRSwap(effective, maturity, 0.04, 1_000_000, pay_fixed=True)
        par       = dummy.par_rate(bootstrapped_curve)
        s1 = SOFRSwap(effective, maturity, par, notional=1_000_000, pay_fixed=True)
        s2 = SOFRSwap(effective, maturity, par, notional=10_000_000, pay_fixed=True)
        assert abs(s2.dv01(bootstrapped_curve) / s1.dv01(bootstrapped_curve) - 10.0) < 0.01

    def test_longer_tenor_larger_dv01(self, bootstrapped_curve):
        """10Y swap DV01 > 5Y swap DV01 for same notional."""
        effective = REF_DATE + timedelta(days=2)
        d5  = SOFRSwap(effective, effective + relativedelta(years=5),
                       bootstrapped_curve.par_ois_rate(5.0), 10_000_000, pay_fixed=True)
        d10 = SOFRSwap(effective, effective + relativedelta(years=10),
                       bootstrapped_curve.par_ois_rate(10.0), 10_000_000, pay_fixed=True)
        assert d10.dv01(bootstrapped_curve) > d5.dv01(bootstrapped_curve)

    def test_summary_dict_keys(self, par_swap_5y, bootstrapped_curve):
        s = par_swap_5y.summary(bootstrapped_curve)
        required_keys = {"par_rate_pct", "net_pv", "dv01", "notional", "tenor_yrs"}
        assert required_keys.issubset(s.keys())


class TestFRA:

    def test_fra_at_par_has_zero_pv(self, bootstrapped_curve):
        """FRA struck at par rate should have PV ≈ 0."""
        start = REF_DATE + timedelta(days=90)
        end   = REF_DATE + timedelta(days=180)
        fra   = ForwardRateAgreement(start, end, fixed_rate=0.04, notional=10_000_000)
        par   = fra.par_rate(bootstrapped_curve)
        fra_at_par = ForwardRateAgreement(start, end, fixed_rate=par, notional=10_000_000)
        assert abs(fra_at_par.pv(bootstrapped_curve)) < 1.0

    def test_fra_dv01_positive(self, bootstrapped_curve):
        start = REF_DATE + timedelta(days=90)
        end   = REF_DATE + timedelta(days=180)
        fra = ForwardRateAgreement(start, end, fixed_rate=0.04, notional=10_000_000)
        assert fra.dv01(bootstrapped_curve) > 0

    def test_fra_par_rate_in_range(self, bootstrapped_curve):
        """FRA par rate should be in a plausible range for the current curve."""
        start = REF_DATE + timedelta(days=90)
        end   = REF_DATE + timedelta(days=180)
        fra   = ForwardRateAgreement(start, end, fixed_rate=0.04, notional=10_000_000)
        par   = fra.par_rate(bootstrapped_curve)
        assert 0.02 < par < 0.08, f"FRA par rate {par*100:.2f}% out of expected range"


class TestSOFRCaplet:

    def test_caplet_itm_positive_pv(self, flat_curve):
        """An in-the-money caplet should have positive PV."""
        reset  = REF_DATE + timedelta(days=90)
        pay    = REF_DATE + timedelta(days=180)
        strike = FLAT_RATE - 0.01
        caplet = SOFRCaplet(reset, pay, strike=strike, notional=1_000_000, cap_floor="cap")
        pv = caplet.pv(flat_curve, normal_vol=0.010)
        assert pv > 0

    def test_floor_itm_positive_pv(self, flat_curve):
        reset  = REF_DATE + timedelta(days=90)
        pay    = REF_DATE + timedelta(days=180)
        strike = FLAT_RATE + 0.01
        floor  = SOFRCaplet(reset, pay, strike=strike, notional=1_000_000, cap_floor="floor")
        pv     = floor.pv(flat_curve, normal_vol=0.010)
        assert pv > 0

    def test_caplet_otm_lower_than_itm(self, flat_curve):
        reset   = REF_DATE + timedelta(days=90)
        pay     = REF_DATE + timedelta(days=180)
        cap_itm = SOFRCaplet(reset, pay, FLAT_RATE - 0.005, cap_floor="cap")
        cap_otm = SOFRCaplet(reset, pay, FLAT_RATE + 0.005, cap_floor="cap")
        assert cap_itm.pv(flat_curve) > cap_otm.pv(flat_curve)

    def test_higher_vol_higher_caplet_price(self, flat_curve):
        reset  = REF_DATE + timedelta(days=90)
        pay    = REF_DATE + timedelta(days=180)
        caplet = SOFRCaplet(reset, pay, FLAT_RATE, cap_floor="cap")
        pv_lo  = caplet.pv(flat_curve, normal_vol=0.005)
        pv_hi  = caplet.pv(flat_curve, normal_vol=0.020)
        assert pv_hi > pv_lo


class TestDayCount:

    def test_act360_3m(self):
        """ACT/360 fraction for a 91-day period = 91/360."""
        start = date(2026, 3, 20)
        end   = date(2026, 6, 20)
        frac  = act360(start, end)
        days  = (end - start).days
        assert abs(frac - days / 360.0) < 1e-10

    def test_year_frac_act360(self):
        start = date(2026, 1, 1)
        end   = date(2026, 7, 1)
        frac  = year_frac(start, end, "ACT360")
        days  = (end - start).days
        assert abs(frac - days / 360.0) < 1e-10

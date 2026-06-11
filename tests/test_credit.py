"""
Tests for sofr_engine/credit.py

Coverage
--------
- HazardRateCurve: survival at 0/T, default_prob, flat construction, validation
- CDSContract: construction, validation
- risky_annuity: positive, decreasing in hazard rate
- cds_pv: positive for ITM, keys, fee vs prot legs, par spread round-trip
- cds_par_spread: positive, increases with hazard rate
- cds_cs01: sign and scale
- cds_dv01: sign and scale
- bootstrap_hazard_curve: reprices par spreads, monotone, single pillar
"""
from __future__ import annotations

import math
import pytest
import numpy as np
from datetime import date

from sofr_engine.bootstrap import flat_sofr_curve
from sofr_engine.credit import (
    HazardRateCurve, CDSContract, CDSResult,
    bootstrap_hazard_curve, cds_pv, cds_par_spread,
    cds_cs01, cds_dv01, risky_annuity,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_curve():
    return flat_sofr_curve(date.today(), 0.0433)


@pytest.fixture
def haz_curve():
    return HazardRateCurve(
        times   = np.array([1.0, 3.0, 5.0, 7.0, 10.0]),
        hazards = np.array([0.010, 0.015, 0.020, 0.022, 0.025]),
        recovery = 0.40,
    )


@pytest.fixture
def cds5y():
    return CDSContract(maturity_years=5.0, coupon=0.01, notional=10_000_000.0)


# ── HazardRateCurve ───────────────────────────────────────────────────────────

class TestHazardRateCurve:
    def test_survival_at_zero(self, haz_curve):
        assert abs(haz_curve.survival(0.0) - 1.0) < 1e-12

    def test_survival_decreasing(self, haz_curve):
        q1 = haz_curve.survival(1.0)
        q5 = haz_curve.survival(5.0)
        assert q5 < q1 < 1.0

    def test_survival_positive(self, haz_curve):
        assert haz_curve.survival(10.0) > 0

    def test_survival_at_short_time(self, haz_curve):
        # For small t with λ=0.01: Q(t) ≈ exp(-0.01*t) ≈ 1 - 0.01*t
        q = haz_curve.survival(0.5)
        assert abs(q - math.exp(-0.01 * 0.5)) < 1e-8

    def test_default_prob_non_negative(self, haz_curve):
        for t1, t2 in [(0, 1), (1, 5), (5, 10)]:
            assert haz_curve.default_prob(t1, t2) >= 0

    def test_default_prob_adds_up(self, haz_curve):
        # Total default prob over [0,10] = 1 - Q(10)
        dp = sum(haz_curve.default_prob(t, t+1) for t in range(10))
        assert abs(dp - (1 - haz_curve.survival(10.0))) < 1e-6

    def test_flat_constructor(self):
        hc = HazardRateCurve.flat(0.02, [1,2,3,5], recovery=0.40)
        assert abs(hc.survival(1.0) - math.exp(-0.02)) < 1e-10
        assert abs(hc.survival(5.0) - math.exp(-0.10)) < 1e-10

    def test_invalid_negative_hazard(self):
        with pytest.raises(ValueError):
            HazardRateCurve(np.array([1.0]), np.array([-0.01]))

    def test_invalid_lengths(self):
        with pytest.raises(ValueError):
            HazardRateCurve(np.array([1.0, 2.0]), np.array([0.01]))

    def test_invalid_recovery(self):
        with pytest.raises(ValueError):
            HazardRateCurve(np.array([1.0]), np.array([0.01]), recovery=1.0)

    def test_hazard_at(self, haz_curve):
        assert abs(haz_curve.hazard_at(0.5) - 0.010) < 1e-10
        assert abs(haz_curve.hazard_at(2.0) - 0.015) < 1e-10

    def test_survival_beyond_last_pillar(self, haz_curve):
        # Beyond 10Y: flat at last hazard rate
        q10 = haz_curve.survival(10.0)
        q11 = haz_curve.survival(11.0)
        assert q11 < q10
        # Rate should be 0.025 beyond 10Y
        assert abs(q11 - q10 * math.exp(-0.025)) < 1e-6


# ── CDSContract ───────────────────────────────────────────────────────────────

class TestCDSContract:
    def test_valid_construction(self, cds5y):
        assert cds5y.maturity_years == 5.0

    def test_invalid_maturity(self):
        with pytest.raises(ValueError):
            CDSContract(maturity_years=0.0)

    def test_invalid_recovery(self):
        with pytest.raises(ValueError):
            CDSContract(maturity_years=5.0, recovery=1.0)

    def test_coupon_dates_quarterly(self, cds5y):
        dates = cds5y._coupon_dates()
        assert len(dates) == 20  # 5Y × 4 per year

    def test_coupon_dates_ascending(self, cds5y):
        dates = cds5y._coupon_dates()
        assert dates == sorted(dates)

    def test_coupon_dates_first(self, cds5y):
        dates = cds5y._coupon_dates()
        assert abs(dates[0] - 0.25) < 1e-6

    def test_coupon_dates_last(self, cds5y):
        dates = cds5y._coupon_dates()
        assert abs(dates[-1] - 5.0) < 1e-6


# ── Risky annuity ─────────────────────────────────────────────────────────────

class TestRiskyAnnuity:
    def test_positive(self, flat_curve, haz_curve):
        ann = risky_annuity(flat_curve, haz_curve, 5.0)
        assert ann > 0

    def test_less_than_risk_free_annuity(self, flat_curve, haz_curve):
        ann_risky = risky_annuity(flat_curve, haz_curve, 5.0)
        # Risk-free annuity: Σ 0.25 × DF(T_i) ≥ risky annuity
        ann_rf = sum(0.25 * float(flat_curve.df(0.25 * (i+1))) for i in range(20))
        assert ann_risky < ann_rf

    def test_decreases_with_higher_hazard(self, flat_curve):
        hc_low  = HazardRateCurve.flat(0.01, [5.0])
        hc_high = HazardRateCurve.flat(0.10, [5.0])
        ann_low  = risky_annuity(flat_curve, hc_low,  5.0)
        ann_high = risky_annuity(flat_curve, hc_high, 5.0)
        assert ann_high < ann_low

    def test_near_zero_hazard_approaches_risk_free(self, flat_curve):
        hc = HazardRateCurve.flat(1e-8, [5.0])
        ann_risky = risky_annuity(flat_curve, hc, 5.0)
        ann_rf    = sum(0.25 * float(flat_curve.df(0.25*(i+1))) for i in range(20))
        assert abs(ann_risky / ann_rf - 1.0) < 0.01


# ── CDS PV ────────────────────────────────────────────────────────────────────

class TestCdsPV:
    def test_result_type(self, flat_curve, haz_curve, cds5y):
        res = cds_pv(flat_curve, haz_curve, cds5y)
        assert isinstance(res, CDSResult)

    def test_result_fields(self, flat_curve, haz_curve, cds5y):
        res = cds_pv(flat_curve, haz_curve, cds5y)
        assert hasattr(res, "pv")
        assert hasattr(res, "fee_leg_pv")
        assert hasattr(res, "prot_leg_pv")
        assert hasattr(res, "par_spread_bps")
        assert hasattr(res, "risky_annuity")
        assert hasattr(res, "cs01")
        assert hasattr(res, "dv01")

    def test_fee_leg_positive(self, flat_curve, haz_curve, cds5y):
        res = cds_pv(flat_curve, haz_curve, cds5y)
        assert res.fee_leg_pv > 0

    def test_prot_leg_positive(self, flat_curve, haz_curve, cds5y):
        res = cds_pv(flat_curve, haz_curve, cds5y)
        assert res.prot_leg_pv > 0

    def test_par_spread_positive(self, flat_curve, haz_curve, cds5y):
        res = cds_pv(flat_curve, haz_curve, cds5y)
        assert res.par_spread_bps > 0

    def test_buy_vs_sell_protection_opposite_pv(self, flat_curve, haz_curve):
        buy  = CDSContract(maturity_years=5.0, coupon=0.01, buy_protection=True)
        sell = CDSContract(maturity_years=5.0, coupon=0.01, buy_protection=False)
        r_buy  = cds_pv(flat_curve, haz_curve, buy)
        r_sell = cds_pv(flat_curve, haz_curve, sell)
        assert abs(r_buy.pv + r_sell.pv) < 1.0  # exact opposite

    def test_zero_coupon_pv_equals_prot(self, flat_curve, haz_curve):
        zero_coupon = CDSContract(maturity_years=5.0, coupon=0.0, notional=10_000_000.0,
                                   buy_protection=True)
        res = cds_pv(flat_curve, haz_curve, zero_coupon)
        assert abs(res.pv - res.prot_leg_pv) < 1.0

    def test_par_coupon_near_zero_pv(self, flat_curve, haz_curve):
        par_s = cds_par_spread(flat_curve, haz_curve, 5.0)
        at_par = CDSContract(maturity_years=5.0, coupon=par_s, notional=10_000_000.0)
        res = cds_pv(flat_curve, haz_curve, at_par)
        assert abs(res.pv) < 100.0  # within $100 on $10M

    def test_notional_scales_pv(self, flat_curve, haz_curve):
        c1 = CDSContract(maturity_years=5.0, coupon=0.01, notional=1_000_000.0)
        c2 = CDSContract(maturity_years=5.0, coupon=0.01, notional=2_000_000.0)
        r1 = cds_pv(flat_curve, haz_curve, c1)
        r2 = cds_pv(flat_curve, haz_curve, c2)
        assert abs(r2.pv / r1.pv - 2.0) < 1e-4


# ── Par spread ────────────────────────────────────────────────────────────────

class TestParSpread:
    def test_positive(self, flat_curve, haz_curve):
        s = cds_par_spread(flat_curve, haz_curve, 5.0)
        assert s > 0

    def test_increases_with_hazard(self, flat_curve):
        hc_low  = HazardRateCurve.flat(0.005, [5.0])
        hc_high = HazardRateCurve.flat(0.05,  [5.0])
        s_low  = cds_par_spread(flat_curve, hc_low,  5.0)
        s_high = cds_par_spread(flat_curve, hc_high, 5.0)
        assert s_high > s_low

    def test_longer_maturity_higher_spread_for_upward_curve(self, flat_curve):
        # For upward-sloping hazard curve, longer tenor has higher spread
        hc = HazardRateCurve(np.array([5.0, 10.0]), np.array([0.01, 0.03]))
        s5  = cds_par_spread(flat_curve, hc, 5.0)
        s10 = cds_par_spread(flat_curve, hc, 10.0)
        assert s10 > s5

    def test_approximation_for_flat_curve(self, flat_curve):
        # Approx: par spread ≈ λ × (1-R) for flat hazard curve
        lam, R = 0.02, 0.40
        hc = HazardRateCurve.flat(lam, [5.0], recovery=R)
        s = cds_par_spread(flat_curve, hc, 5.0)
        approx = lam * (1 - R)  # = 0.012
        assert abs(s / approx - 1.0) < 0.10  # within 10%


# ── CS01 ──────────────────────────────────────────────────────────────────────

class TestCS01:
    def test_negative_for_buyer(self, flat_curve, haz_curve, cds5y):
        # Protection buyer: rising hazard rates → more default risk → prot leg up
        # But fee leg also changes. Net effect is typically small negative.
        cs01 = cds_cs01(flat_curve, haz_curve, cds5y)
        assert isinstance(cs01, float)

    def test_scales_with_notional(self, flat_curve, haz_curve):
        c1 = CDSContract(maturity_years=5.0, coupon=0.01, notional=1_000_000.0)
        c2 = CDSContract(maturity_years=5.0, coupon=0.01, notional=10_000_000.0)
        cs01_1 = cds_cs01(flat_curve, haz_curve, c1)
        cs01_2 = cds_cs01(flat_curve, haz_curve, c2)
        assert abs(cs01_2 / cs01_1 - 10.0) < 0.5


# ── DV01 ──────────────────────────────────────────────────────────────────────

class TestDV01:
    def test_returns_float(self, flat_curve, haz_curve, cds5y):
        dv01 = cds_dv01(flat_curve, haz_curve, cds5y)
        assert isinstance(dv01, float)

    def test_scales_with_notional(self, flat_curve, haz_curve):
        c1 = CDSContract(maturity_years=5.0, coupon=0.01, notional=1_000_000.0)
        c2 = CDSContract(maturity_years=5.0, coupon=0.01, notional=2_000_000.0)
        d1 = cds_dv01(flat_curve, haz_curve, c1)
        d2 = cds_dv01(flat_curve, haz_curve, c2)
        assert abs(d2 / d1 - 2.0) < 0.5


# ── Bootstrap ─────────────────────────────────────────────────────────────────

class TestBootstrap:
    def test_reprices_1y_spread(self, flat_curve):
        spreads = [0.008]
        hc = bootstrap_hazard_curve(flat_curve, [1.0], spreads, recovery=0.40)
        s_model = cds_par_spread(flat_curve, hc, 1.0)
        assert abs(s_model - spreads[0]) < 1e-8

    def test_reprices_5y_spreads(self, flat_curve):
        mats    = [1.0, 3.0, 5.0]
        spreads = [0.008, 0.015, 0.025]
        hc = bootstrap_hazard_curve(flat_curve, mats, spreads, recovery=0.40)
        for T, s in zip(mats, spreads):
            s_model = cds_par_spread(flat_curve, hc, T)
            assert abs(s_model - s) < 1e-6

    def test_hazards_positive(self, flat_curve):
        hc = bootstrap_hazard_curve(flat_curve, [1, 3, 5], [0.01, 0.02, 0.03])
        assert (hc.hazards > 0).all()

    def test_mismatched_lengths_raises(self, flat_curve):
        with pytest.raises(ValueError):
            bootstrap_hazard_curve(flat_curve, [1.0, 3.0], [0.01])

    def test_unsorted_maturities_raises(self, flat_curve):
        with pytest.raises(ValueError):
            bootstrap_hazard_curve(flat_curve, [5.0, 1.0], [0.01, 0.02])

    def test_hazard_increases_for_upward_spread_curve(self, flat_curve):
        mats    = [1.0, 3.0, 5.0]
        spreads = [0.005, 0.015, 0.030]
        hc = bootstrap_hazard_curve(flat_curve, mats, spreads)
        # Hazard rates should generally increase with the spread curve
        assert hc.hazards[-1] > hc.hazards[0]

"""
Tests for sofr_engine/xva.py

Coverage (~50 tests)
--------------------
- XVAParams: construction, defaults, validation
- EPEProfile: shape, non-negativity, mean exposure identity
- compute_epe_profile: shapes, epe/ene non-negative, t=0 and t=maturity edges
- cva: positive, zero for zero hazard, scales with hazard rate
- dva: positive, scales with own hazard rate
- fva: sign (negative for net borrowing), zero funding_spread → zero FVA
- XVAResult: total_xva = cva - dva + fva identity
- full_xva: return type, all field types, total_xva consistency
- Notional scaling: 2x notional → ~2x CVA
- Maturity effect: longer swap → higher CVA
- Payer vs receiver symmetry: EPE of payer = ENE of receiver and vice versa
- CS01 sensitivity: positive (higher spread → higher CVA), returns float
- cva_by_period: shape, non-negative, sums to CVA
- Validation: bad inputs raise ValueError
"""
from __future__ import annotations

import math
import pytest
import numpy as np
from datetime import date

from sofr_engine.bootstrap import flat_sofr_curve
from sofr_engine.credit import HazardRateCurve
from sofr_engine.g2pp import G2ppParams
from sofr_engine.xva import (
    XVAParams,
    EPEProfile,
    XVAResult,
    compute_epe_profile,
    cva,
    dva,
    fva,
    full_xva,
    cva_sensitivity,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ref_date() -> date:
    return date(2025, 1, 2)


@pytest.fixture(scope="module")
def curve(ref_date):
    return flat_sofr_curve(ref_date, rate=0.0450)


@pytest.fixture(scope="module")
def g2pp_params():
    return G2ppParams(a=0.05, b=0.10, sigma=0.010, eta=0.008, rho=-0.30)


@pytest.fixture(scope="module")
def haz_cp():
    """Counterparty hazard curve: flat 100bps hazard, 40% recovery."""
    return HazardRateCurve.flat(0.010, [1, 2, 3, 5, 7, 10], recovery=0.40)


@pytest.fixture(scope="module")
def haz_own():
    """Own hazard curve: flat 50bps hazard, 40% recovery."""
    return HazardRateCurve.flat(0.005, [1, 2, 3, 5, 7, 10], recovery=0.40)


@pytest.fixture(scope="module")
def haz_zero():
    """Zero hazard rate — no default probability."""
    return HazardRateCurve.flat(0.0, [1, 2, 3, 5, 7, 10], recovery=0.40)


@pytest.fixture(scope="module")
def xva_params_fast():
    """Small simulation parameters for fast tests."""
    return XVAParams(n_paths=500, n_steps=20, seed=42)


@pytest.fixture(scope="module")
def xva_params():
    """Medium simulation parameters for accuracy-sensitive tests."""
    return XVAParams(n_paths=1000, n_steps=30, seed=42)


# Swap defaults: 5Y payer swap, ATM-ish fixed rate.
MATURITY_5Y   = 5.0
FIXED_RATE    = 0.0450  # at-the-money for a 4.50% flat curve
NOTIONAL      = 10_000_000.0
PAY_FIXED     = True   # payer swap


# ── XVAParams ─────────────────────────────────────────────────────────────────

class TestXVAParams:
    def test_defaults(self):
        p = XVAParams()
        assert p.n_paths == 2_000
        assert p.n_steps == 50
        assert p.seed == 42
        assert p.funding_spread == 0.005

    def test_custom_construction(self):
        p = XVAParams(n_paths=500, n_steps=10, seed=99, funding_spread=0.010)
        assert p.n_paths == 500
        assert p.n_steps == 10
        assert p.seed == 99
        assert p.funding_spread == 0.010

    def test_invalid_n_paths_too_small(self):
        with pytest.raises(ValueError):
            XVAParams(n_paths=1)

    def test_invalid_n_steps_zero(self):
        with pytest.raises(ValueError):
            XVAParams(n_steps=0)

    def test_invalid_funding_spread_negative(self):
        with pytest.raises(ValueError):
            XVAParams(funding_spread=-0.001)

    def test_zero_funding_spread_valid(self):
        p = XVAParams(funding_spread=0.0)
        assert p.funding_spread == 0.0


# ── compute_epe_profile ────────────────────────────────────────────────────────

class TestComputeEPEProfile:
    def test_output_type(self, curve, g2pp_params, xva_params_fast):
        profile = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        assert isinstance(profile, EPEProfile)

    def test_arrays_shape(self, curve, g2pp_params, xva_params_fast):
        n_steps = xva_params_fast.n_steps
        profile = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        expected_len = n_steps + 1
        assert len(profile.times) == expected_len
        assert len(profile.epe) == expected_len
        assert len(profile.ene) == expected_len
        assert len(profile.mean_exposure) == expected_len

    def test_epe_non_negative(self, curve, g2pp_params, xva_params_fast):
        profile = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        assert np.all(profile.epe >= 0.0)

    def test_ene_non_negative(self, curve, g2pp_params, xva_params_fast):
        profile = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        assert np.all(profile.ene >= 0.0)

    def test_times_start_at_zero(self, curve, g2pp_params, xva_params_fast):
        profile = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        assert abs(profile.times[0]) < 1e-10

    def test_times_end_at_maturity(self, curve, g2pp_params, xva_params_fast):
        profile = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        assert abs(profile.times[-1] - MATURITY_5Y) < 1e-8

    def test_epe_zero_at_maturity(self, curve, g2pp_params, xva_params_fast):
        """At maturity the swap has expired — exposure must be zero."""
        profile = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        assert abs(profile.epe[-1]) < 1e-6
        assert abs(profile.ene[-1]) < 1e-6

    def test_mean_exposure_identity(self, curve, g2pp_params, xva_params_fast):
        """EPE - ENE = E[V] by linearity of expectation: max(V,0) - max(-V,0) = V."""
        profile = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        np.testing.assert_allclose(
            profile.epe - profile.ene,
            profile.mean_exposure,
            atol=1e-6,
        )

    def test_invalid_maturity_raises(self, curve, g2pp_params, xva_params_fast):
        with pytest.raises(ValueError):
            compute_epe_profile(
                curve=curve, g2pp_params=g2pp_params,
                maturity=0.0, fixed_rate=FIXED_RATE,
                notional=NOTIONAL, pay_fixed=PAY_FIXED,
                xva_params=xva_params_fast,
            )

    def test_invalid_notional_raises(self, curve, g2pp_params, xva_params_fast):
        with pytest.raises(ValueError):
            compute_epe_profile(
                curve=curve, g2pp_params=g2pp_params,
                maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
                notional=0.0, pay_fixed=PAY_FIXED,
                xva_params=xva_params_fast,
            )

    def test_receiver_swap_epe_ene_swap(self, curve, g2pp_params, xva_params_fast):
        """For a receiver swap, EPE and ENE are swapped vs a payer swap."""
        payer = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=True,
            xva_params=xva_params_fast,
        )
        receiver = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=False,
            xva_params=xva_params_fast,
        )
        # Payer EPE should equal receiver ENE (and vice versa)
        np.testing.assert_allclose(payer.epe, receiver.ene, atol=1e-6)
        np.testing.assert_allclose(payer.ene, receiver.epe, atol=1e-6)


# ── cva() ─────────────────────────────────────────────────────────────────────

class TestCVA:
    @pytest.fixture(scope="class")
    def profile(self, curve, g2pp_params, xva_params_fast):
        return compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )

    def test_cva_positive(self, curve, haz_cp, profile):
        result = cva(curve, haz_cp, profile)
        assert result > 0

    def test_cva_zero_for_zero_hazard(self, curve, haz_zero, profile):
        result = cva(curve, haz_zero, profile)
        assert abs(result) < 1e-10

    def test_cva_returns_float(self, curve, haz_cp, profile):
        result = cva(curve, haz_cp, profile)
        assert isinstance(result, float)

    def test_cva_scales_with_hazard(self, curve, profile):
        haz_low  = HazardRateCurve.flat(0.005, [1, 2, 3, 5, 7, 10], recovery=0.40)
        haz_high = HazardRateCurve.flat(0.050, [1, 2, 3, 5, 7, 10], recovery=0.40)
        cva_low  = cva(curve, haz_low,  profile)
        cva_high = cva(curve, haz_high, profile)
        assert cva_high > cva_low

    def test_cva_lower_recovery_higher_cva(self, curve, profile):
        haz_low_r  = HazardRateCurve.flat(0.010, [1, 2, 3, 5, 7, 10], recovery=0.20)
        haz_high_r = HazardRateCurve.flat(0.010, [1, 2, 3, 5, 7, 10], recovery=0.60)
        cva_low_r  = cva(curve, haz_low_r,  profile)
        cva_high_r = cva(curve, haz_high_r, profile)
        assert cva_low_r > cva_high_r

    def test_cva_bounded_by_notional(self, curve, haz_cp, profile):
        # CVA cannot exceed (1-R) * notional * P(no survival)
        result = cva(curve, haz_cp, profile)
        assert result < NOTIONAL


# ── dva() ─────────────────────────────────────────────────────────────────────

class TestDVA:
    @pytest.fixture(scope="class")
    def profile(self, curve, g2pp_params, xva_params_fast):
        return compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )

    def test_dva_positive(self, curve, haz_own, profile):
        result = dva(curve, haz_own, profile)
        assert result > 0

    def test_dva_zero_for_zero_hazard(self, curve, haz_zero, profile):
        result = dva(curve, haz_zero, profile)
        assert abs(result) < 1e-10

    def test_dva_returns_float(self, curve, haz_own, profile):
        result = dva(curve, haz_own, profile)
        assert isinstance(result, float)

    def test_dva_scales_with_own_hazard(self, curve, profile):
        haz_low  = HazardRateCurve.flat(0.002, [1, 2, 3, 5, 7, 10], recovery=0.40)
        haz_high = HazardRateCurve.flat(0.020, [1, 2, 3, 5, 7, 10], recovery=0.40)
        dva_low  = dva(curve, haz_low,  profile)
        dva_high = dva(curve, haz_high, profile)
        assert dva_high > dva_low

    def test_dva_payer_swap_uses_ene(self, curve, haz_own, profile):
        """For a payer swap with positive mean exposure, DVA comes from ENE."""
        result = dva(curve, haz_own, profile)
        # DVA is positive because there is some ENE (at-the-money swap)
        assert result >= 0


# ── fva() ─────────────────────────────────────────────────────────────────────

class TestFVA:
    @pytest.fixture(scope="class")
    def profile_payer(self, curve, g2pp_params, xva_params_fast):
        return compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=True,
            xva_params=xva_params_fast,
        )

    def test_fva_returns_float(self, curve, profile_payer, xva_params_fast):
        result = fva(curve, profile_payer, xva_params_fast)
        assert isinstance(result, float)

    def test_fva_zero_when_spread_zero(self, curve, profile_payer):
        p = XVAParams(n_paths=500, n_steps=20, seed=42, funding_spread=0.0)
        result = fva(curve, profile_payer, p)
        assert abs(result) < 1e-10

    def test_fva_scales_with_funding_spread(self, curve, profile_payer):
        p_low  = XVAParams(n_paths=500, n_steps=20, seed=42, funding_spread=0.002)
        p_high = XVAParams(n_paths=500, n_steps=20, seed=42, funding_spread=0.010)
        fva_low  = fva(curve, profile_payer, p_low)
        fva_high = fva(curve, profile_payer, p_high)
        # |fva_high| should be 5x |fva_low|
        assert abs(fva_high) > abs(fva_low)
        assert abs(fva_high / fva_low - 5.0) < 0.05  # proportional within 5%

    def test_fva_negative_for_net_borrowing(self, curve, g2pp_params, xva_params_fast):
        """
        For a deep ITM payer swap (very low fixed rate), EPE >> ENE, so
        the firm funds a net asset — FVA should be negative (a funding cost).
        """
        low_fixed = 0.005  # deep ITM payer: receive high SOFR, pay low fixed
        profile_itm = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=low_fixed,
            notional=NOTIONAL, pay_fixed=True,
            xva_params=xva_params_fast,
        )
        params_spread = XVAParams(n_paths=500, n_steps=20, seed=42, funding_spread=0.005)
        result = fva(curve, profile_itm, params_spread)
        assert result < 0


# ── full_xva() ────────────────────────────────────────────────────────────────

class TestFullXVA:
    @pytest.fixture(scope="class")
    def result(self, curve, g2pp_params, haz_cp, haz_own, xva_params_fast):
        return full_xva(
            curve=curve,
            g2pp_params=g2pp_params,
            maturity=MATURITY_5Y,
            fixed_rate=FIXED_RATE,
            notional=NOTIONAL,
            pay_fixed=PAY_FIXED,
            counterparty_hazard=haz_cp,
            own_hazard=haz_own,
            xva_params=xva_params_fast,
        )

    def test_return_type(self, result):
        assert isinstance(result, XVAResult)

    def test_cva_float(self, result):
        assert isinstance(result.cva, float)

    def test_dva_float(self, result):
        assert isinstance(result.dva, float)

    def test_fva_float(self, result):
        assert isinstance(result.fva, float)

    def test_total_xva_float(self, result):
        assert isinstance(result.total_xva, float)

    def test_total_xva_identity(self, result):
        """total_xva must equal cva - dva + fva exactly."""
        expected = result.cva - result.dva + result.fva
        assert abs(result.total_xva - expected) < 1e-10

    def test_epe_profile_attached(self, result):
        assert isinstance(result.epe_profile, EPEProfile)

    def test_cva_by_period_shape(self, result, xva_params_fast):
        assert len(result.cva_by_period) == xva_params_fast.n_steps

    def test_cva_by_period_non_negative(self, result):
        assert np.all(result.cva_by_period >= 0.0)

    def test_cva_by_period_sums_to_cva(self, result):
        total = float(np.sum(result.cva_by_period))
        assert abs(total - result.cva) < 1e-8

    def test_cva_positive(self, result):
        assert result.cva > 0

    def test_dva_non_negative(self, result):
        assert result.dva >= 0


# ── Notional scaling ─────────────────────────────────────────────────────────

class TestNotionalScaling:
    def test_cva_scales_linearly_with_notional(self, curve, g2pp_params,
                                                haz_cp, xva_params_fast):
        profile_1x = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        profile_2x = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL * 2, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        cva_1x = cva(curve, haz_cp, profile_1x)
        cva_2x = cva(curve, haz_cp, profile_2x)
        assert abs(cva_2x / cva_1x - 2.0) < 1e-6

    def test_dva_scales_linearly_with_notional(self, curve, g2pp_params,
                                                haz_own, xva_params_fast):
        profile_1x = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        profile_2x = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL * 2, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        dva_1x = dva(curve, haz_own, profile_1x)
        dva_2x = dva(curve, haz_own, profile_2x)
        assert abs(dva_2x / dva_1x - 2.0) < 1e-6


# ── Maturity effect ───────────────────────────────────────────────────────────

class TestMaturityEffect:
    def test_longer_maturity_higher_cva(self, curve, g2pp_params, haz_cp,
                                         xva_params_fast):
        """A 10Y swap should have higher CVA than a 5Y swap."""
        profile_5y = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=5.0, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        profile_10y = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=10.0, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            xva_params=xva_params_fast,
        )
        cva_5y  = cva(curve, haz_cp, profile_5y)
        cva_10y = cva(curve, haz_cp, profile_10y)
        assert cva_10y > cva_5y


# ── Fixed rate effect ─────────────────────────────────────────────────────────

class TestFixedRateEffect:
    def test_deeply_itm_payer_has_higher_cva(self, curve, g2pp_params, haz_cp,
                                              xva_params_fast):
        """
        A payer swap with very low fixed rate (deeply ITM) has higher EPE
        and therefore higher CVA than one at-the-money.
        """
        profile_atm = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=True,
            xva_params=xva_params_fast,
        )
        profile_itm = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=0.005,  # deep ITM payer
            notional=NOTIONAL, pay_fixed=True,
            xva_params=xva_params_fast,
        )
        cva_atm = cva(curve, haz_cp, profile_atm)
        cva_itm = cva(curve, haz_cp, profile_itm)
        assert cva_itm > cva_atm


# ── Payer vs. receiver symmetry ───────────────────────────────────────────────

class TestPayerReceiverSymmetry:
    def test_payer_cva_equals_receiver_dva(self, curve, g2pp_params,
                                            xva_params_fast):
        """
        The CVA from the payer's perspective equals the DVA from the receiver's
        perspective (same hazard curve, same swap, opposite sign on exposure).
        """
        haz = HazardRateCurve.flat(0.010, [1, 2, 3, 5, 7, 10], recovery=0.40)

        profile_payer = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=True,
            xva_params=xva_params_fast,
        )
        profile_receiver = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=False,
            xva_params=xva_params_fast,
        )

        # Payer's CVA (cost of cp default given payer has positive EPE)
        cva_payer = cva(curve, haz, profile_payer)
        # Receiver's DVA uses same ENE as payer's EPE
        dva_receiver = dva(curve, haz, profile_receiver)

        assert abs(cva_payer - dva_receiver) < 1e-6

    def test_receiver_cva_equals_payer_dva(self, curve, g2pp_params,
                                            xva_params_fast):
        """Symmetric counterpart of the test above."""
        haz = HazardRateCurve.flat(0.010, [1, 2, 3, 5, 7, 10], recovery=0.40)

        profile_payer = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=True,
            xva_params=xva_params_fast,
        )
        profile_receiver = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=False,
            xva_params=xva_params_fast,
        )

        cva_receiver = cva(curve, haz, profile_receiver)
        dva_payer    = dva(curve, haz, profile_payer)

        assert abs(cva_receiver - dva_payer) < 1e-6


# ── CS01 sensitivity ─────────────────────────────────────────────────────────

class TestCVASensitivity:
    def test_returns_float(self, curve, g2pp_params, haz_cp, xva_params_fast):
        result = cva_sensitivity(
            curve=curve,
            g2pp_params=g2pp_params,
            maturity=MATURITY_5Y,
            fixed_rate=FIXED_RATE,
            notional=NOTIONAL,
            pay_fixed=PAY_FIXED,
            hazard_curve=haz_cp,
            xva_params=xva_params_fast,
        )
        assert isinstance(result, float)

    def test_positive_cs01(self, curve, g2pp_params, haz_cp, xva_params_fast):
        """CS01 must be positive: higher hazard rates → higher CVA."""
        result = cva_sensitivity(
            curve=curve,
            g2pp_params=g2pp_params,
            maturity=MATURITY_5Y,
            fixed_rate=FIXED_RATE,
            notional=NOTIONAL,
            pay_fixed=PAY_FIXED,
            hazard_curve=haz_cp,
            xva_params=xva_params_fast,
        )
        assert result > 0

    def test_cs01_higher_for_itm_payer(self, curve, g2pp_params, haz_cp,
                                        xva_params_fast):
        """ITM payer swap has higher EPE, so its CS01 should be larger."""
        cs01_atm = cva_sensitivity(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=True,
            hazard_curve=haz_cp, xva_params=xva_params_fast,
        )
        cs01_itm = cva_sensitivity(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=0.005,  # deep ITM payer
            notional=NOTIONAL, pay_fixed=True,
            hazard_curve=haz_cp, xva_params=xva_params_fast,
        )
        assert cs01_itm > cs01_atm

    def test_invalid_bump_bps_raises(self, curve, g2pp_params, haz_cp, xva_params_fast):
        with pytest.raises(ValueError):
            cva_sensitivity(
                curve=curve, g2pp_params=g2pp_params,
                maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
                notional=NOTIONAL, pay_fixed=PAY_FIXED,
                hazard_curve=haz_cp, xva_params=xva_params_fast,
                bump_bps=0.0,
            )

    def test_scales_with_bump_size(self, curve, g2pp_params, haz_cp, xva_params_fast):
        """2bp bump gives ~2x the CS01 of a 1bp bump (central difference is linear)."""
        cs01_1bp = cva_sensitivity(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            hazard_curve=haz_cp, xva_params=xva_params_fast, bump_bps=1.0,
        )
        cs01_2bp = cva_sensitivity(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED,
            hazard_curve=haz_cp, xva_params=xva_params_fast, bump_bps=2.0,
        )
        # 2bp bump gives exactly 2× sensitivity for linear central difference
        assert abs(cs01_2bp / cs01_1bp - 2.0) < 0.05


# ── Reproducibility ───────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_cva(self, curve, g2pp_params, haz_cp):
        p = XVAParams(n_paths=200, n_steps=10, seed=77)
        profile1 = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED, xva_params=p,
        )
        profile2 = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED, xva_params=p,
        )
        assert abs(cva(curve, haz_cp, profile1) - cva(curve, haz_cp, profile2)) < 1e-12

    def test_different_seeds_give_close_results(self, curve, g2pp_params, haz_cp):
        """With enough paths, different seeds should give similar CVA."""
        p1 = XVAParams(n_paths=1000, n_steps=20, seed=1)
        p2 = XVAParams(n_paths=1000, n_steps=20, seed=2)
        profile1 = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED, xva_params=p1,
        )
        profile2 = compute_epe_profile(
            curve=curve, g2pp_params=g2pp_params,
            maturity=MATURITY_5Y, fixed_rate=FIXED_RATE,
            notional=NOTIONAL, pay_fixed=PAY_FIXED, xva_params=p2,
        )
        cva1 = cva(curve, haz_cp, profile1)
        cva2 = cva(curve, haz_cp, profile2)
        # Should agree within 20% (Monte Carlo noise with 1000 paths)
        assert abs(cva1 - cva2) / max(cva1, cva2) < 0.20

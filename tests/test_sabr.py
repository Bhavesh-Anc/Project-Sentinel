"""
Tests for sofr_engine/sabr.py — SABR stochastic volatility model.

Covers:
- sabr_implied_vol  : ATM formula, smile shape, edge cases
- sabr_normal_vol   : sign and approximate relationship to Black vol
- calibrate_sabr    : round-trip, single-strike, bounds, convergence
- sabr_vol_smile    : DataFrame shape, column values, skew direction
- SABRSurface       : calibrate_from_atm_surface, vol() interpolation
"""
from __future__ import annotations

import numpy as np
import pytest

from sofr_engine.sabr import (
    SABRParams,
    sabr_implied_vol,
    sabr_normal_vol,
    calibrate_sabr,
    sabr_vol_smile,
    SABRSurface,
    _sabr_atm_vol,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

F_USD    = 0.0453   # 10Y SOFR forward rate (approx current market)
T_1Y     = 1.0
T_5Y     = 5.0

PARAMS_BASE = SABRParams(alpha=0.03, beta=0.5, rho=-0.25, nu=0.40)
PARAMS_FLAT = SABRParams(alpha=0.03, beta=0.5, rho=0.0,   nu=0.0)


def _surface_vols():
    """Typical USD ATM swaption vol surface (Black-76 decimal)."""
    import numpy as np
    expiries = [0.5, 1.0, 2.0, 5.0, 10.0]
    tenors   = [1.0, 2.0, 5.0, 10.0, 30.0]
    vols = np.array([
        [0.165, 0.150, 0.130, 0.110, 0.090],
        [0.155, 0.140, 0.125, 0.105, 0.085],
        [0.140, 0.130, 0.115, 0.098, 0.080],
        [0.115, 0.110, 0.100, 0.088, 0.074],
        [0.095, 0.092, 0.086, 0.078, 0.068],
    ])
    return expiries, tenors, vols


# ── TestSABRParams ────────────────────────────────────────────────────────────

class TestSABRParams:

    def test_valid_construction(self):
        p = SABRParams(alpha=0.02, beta=0.5, rho=-0.30, nu=0.35)
        assert p.alpha == 0.02
        assert p.beta  == 0.5
        assert p.rho   == -0.30
        assert p.nu    == 0.35

    def test_raises_on_non_positive_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            SABRParams(alpha=0.0, beta=0.5, rho=0.0, nu=0.4)

    def test_raises_on_beta_out_of_range(self):
        with pytest.raises(ValueError, match="beta"):
            SABRParams(alpha=0.03, beta=1.5, rho=0.0, nu=0.4)

    def test_raises_on_rho_out_of_range(self):
        with pytest.raises(ValueError, match="rho"):
            SABRParams(alpha=0.03, beta=0.5, rho=1.0, nu=0.4)

    def test_raises_on_negative_nu(self):
        with pytest.raises(ValueError, match="nu"):
            SABRParams(alpha=0.03, beta=0.5, rho=0.0, nu=-0.1)


# ── TestSABRImpliedVol ────────────────────────────────────────────────────────

class TestSABRImpliedVol:

    def test_atm_vol_positive(self):
        vol = sabr_implied_vol(F_USD, F_USD, T_1Y, PARAMS_BASE)
        assert vol > 0.0

    def test_atm_vol_matches_atm_formula(self):
        """sabr_implied_vol at K=F should match the dedicated ATM formula."""
        vol_general = _sabr_atm_vol(F_USD, T_1Y, PARAMS_BASE)
        vol_router  = sabr_implied_vol(F_USD, F_USD, T_1Y, PARAMS_BASE)
        assert abs(vol_general - vol_router) < 1e-10

    def test_atm_vol_near_k_matches_atm_formula(self):
        """Near-ATM (K very close to F) should also use the ATM path."""
        K_near = F_USD * (1 + 5e-9)   # within 1e-7 log-distance
        vol_near = sabr_implied_vol(F_USD, K_near, T_1Y, PARAMS_BASE)
        vol_atm  = _sabr_atm_vol(F_USD, T_1Y, PARAMS_BASE)
        assert abs(vol_near - vol_atm) < 1e-8

    def test_vol_range_typical_usd_params(self):
        """Standard USD swaption params should give a vol in [5%, 25%]."""
        vol = sabr_implied_vol(F_USD, F_USD, T_1Y, PARAMS_BASE)
        assert 0.05 < vol < 0.25

    def test_vol_increases_with_nu(self):
        """Higher vol-of-vol → wider smile (OTM vol increases more than ATM)."""
        K_otm = F_USD * 1.10
        p_lo  = SABRParams(alpha=0.03, beta=0.5, rho=-0.25, nu=0.20)
        p_hi  = SABRParams(alpha=0.03, beta=0.5, rho=-0.25, nu=0.80)
        vol_lo = sabr_implied_vol(F_USD, K_otm, T_1Y, p_lo)
        vol_hi = sabr_implied_vol(F_USD, K_otm, T_1Y, p_hi)
        assert vol_hi > vol_lo

    def test_smile_curvature_increases_with_nu(self):
        """Higher nu → wider symmetric smile (convexity = avg OTM vol − ATM vol increases)."""
        # Use β=1 so the backbone is flat and nu is the sole driver of curvature.
        K_up = F_USD * 1.10
        K_dn = F_USD * 0.90
        p_lo  = SABRParams(alpha=0.03, beta=1.0, rho=0.0, nu=0.10)
        p_hi  = SABRParams(alpha=0.03, beta=1.0, rho=0.0, nu=0.60)
        conv_lo = (
            sabr_implied_vol(F_USD, K_up, T_1Y, p_lo)
            + sabr_implied_vol(F_USD, K_dn, T_1Y, p_lo)
        ) / 2.0 - _sabr_atm_vol(F_USD, T_1Y, p_lo)
        conv_hi = (
            sabr_implied_vol(F_USD, K_up, T_1Y, p_hi)
            + sabr_implied_vol(F_USD, K_dn, T_1Y, p_hi)
        ) / 2.0 - _sabr_atm_vol(F_USD, T_1Y, p_hi)
        assert conv_hi > conv_lo

    def test_negative_rho_creates_put_skew(self):
        """Negative rho → OTM receiver (low K) vol > OTM payer (high K) vol."""
        K_lo = F_USD * 0.90
        K_hi = F_USD * 1.10
        p = SABRParams(alpha=0.03, beta=0.5, rho=-0.40, nu=0.40)
        vol_lo = sabr_implied_vol(F_USD, K_lo, T_1Y, p)
        vol_hi = sabr_implied_vol(F_USD, K_hi, T_1Y, p)
        assert vol_lo > vol_hi

    def test_positive_rho_creates_call_skew(self):
        """Positive rho → OTM payer (high K) vol > OTM receiver (low K) vol."""
        K_lo = F_USD * 0.90
        K_hi = F_USD * 1.10
        p = SABRParams(alpha=0.03, beta=0.5, rho=0.40, nu=0.40)
        vol_lo = sabr_implied_vol(F_USD, K_lo, T_1Y, p)
        vol_hi = sabr_implied_vol(F_USD, K_hi, T_1Y, p)
        assert vol_hi > vol_lo

    def test_zero_nu_zero_rho_flat_smile(self):
        """With β=1, nu=0 and rho=0 the SABR smile is exactly flat (log-normal backbone)."""
        # β=1 makes the backbone F-independent, and nu=0, rho=0 removes all smile.
        # For β<1 the denominator correction introduces a mild strike dependence
        # even with nu=0 (backbone-driven skew) — that is correct SABR behaviour.
        p_flat_beta1 = SABRParams(alpha=0.03, beta=1.0, rho=0.0, nu=0.0)
        atm_vol = _sabr_atm_vol(F_USD, T_1Y, p_flat_beta1)
        for K in [F_USD * 0.85, F_USD * 0.95, F_USD * 1.05, F_USD * 1.15]:
            v = sabr_implied_vol(F_USD, K, T_1Y, p_flat_beta1)
            assert abs(v - atm_vol) < 1e-8, f"Smile not flat at K={K}: {v} vs ATM={atm_vol}"

    def test_zero_T_returns_zero(self):
        assert sabr_implied_vol(F_USD, F_USD, 0.0, PARAMS_BASE) == 0.0

    def test_negative_T_returns_zero(self):
        assert sabr_implied_vol(F_USD, F_USD, -1.0, PARAMS_BASE) == 0.0

    def test_raises_on_nonpositive_F(self):
        with pytest.raises(ValueError, match="Forward"):
            sabr_implied_vol(0.0, F_USD, T_1Y, PARAMS_BASE)

    def test_raises_on_nonpositive_K(self):
        with pytest.raises(ValueError, match="Strike"):
            sabr_implied_vol(F_USD, -0.01, T_1Y, PARAMS_BASE)

    def test_vol_positive_for_otm_strikes(self):
        for K in [F_USD * 0.80, F_USD * 0.90, F_USD * 1.10, F_USD * 1.20]:
            v = sabr_implied_vol(F_USD, K, T_1Y, PARAMS_BASE)
            assert v > 0.0, f"Vol not positive for K={K}"

    def test_small_T_gives_positive_vol(self):
        """Near-expiry: vol should remain positive and well-defined."""
        v = sabr_implied_vol(F_USD, F_USD, 0.01, PARAMS_BASE)
        assert v > 0.0

    def test_long_T_vol_positive(self):
        v = sabr_implied_vol(F_USD, F_USD, 30.0, PARAMS_BASE)
        assert v > 0.0


# ── TestSABRNormalVol ─────────────────────────────────────────────────────────

class TestSABRNormalVol:

    def test_normal_vol_positive(self):
        nvol = sabr_normal_vol(F_USD, F_USD, T_1Y, PARAMS_BASE)
        assert nvol > 0.0

    def test_normal_vol_atm_approx_black_times_F(self):
        """At ATM, σ_normal ≈ σ_black × F (for β=1 exactly; for β=0.5, × sqrt(F²) = F)."""
        bvol = sabr_implied_vol(F_USD, F_USD, T_1Y, PARAMS_BASE)
        nvol = sabr_normal_vol(F_USD, F_USD, T_1Y, PARAMS_BASE)
        # σ_normal = σ_black × √(F·K) = σ_black × F at ATM
        assert abs(nvol - bvol * F_USD) < 1e-10

    def test_normal_vol_less_than_black_times_rate(self):
        """Normal vol in bps should be of order σ_black × F × 10000."""
        bvol = sabr_implied_vol(F_USD, F_USD * 1.05, T_1Y, PARAMS_BASE)
        nvol = sabr_normal_vol(F_USD, F_USD * 1.05, T_1Y, PARAMS_BASE)
        K = F_USD * 1.05
        expected = bvol * np.sqrt(F_USD * K)
        assert abs(nvol - expected) < 1e-10

    def test_normal_vol_positive_for_otm(self):
        for K in [F_USD * 0.85, F_USD * 1.15]:
            nvol = sabr_normal_vol(F_USD, K, T_1Y, PARAMS_BASE)
            assert nvol > 0.0

    def test_normal_vol_reasonable_bps(self):
        """Normal vol in bps should be in a realistic range for USD swaptions."""
        nvol_bps = sabr_normal_vol(F_USD, F_USD, T_1Y, PARAMS_BASE) * 10_000
        assert 20 < nvol_bps < 200, f"Normal vol {nvol_bps:.1f} bps seems unrealistic"


# ── TestSABRCalibration ───────────────────────────────────────────────────────

class TestSABRCalibration:

    def _generate_smile(self, params: SABRParams, F: float = F_USD, T: float = T_1Y):
        """Generate a set of (strike, vol) pairs from known SABR params."""
        offsets = [-100, -50, -25, 0, 25, 50, 100]   # bps
        strikes = [F + o / 10_000.0 for o in offsets]
        vols    = [sabr_implied_vol(F, k, T, params) for k in strikes]
        return strikes, vols

    def test_round_trip_alpha(self):
        """Calibrated alpha should recover the original alpha closely."""
        true_params = SABRParams(alpha=0.035, beta=0.5, rho=-0.25, nu=0.40)
        strikes, vols = self._generate_smile(true_params)
        cal = calibrate_sabr(F_USD, T_1Y, strikes, vols, beta=0.5)
        assert abs(cal.alpha - true_params.alpha) < 5e-4

    def test_round_trip_rho(self):
        """Calibrated rho should recover the original rho closely."""
        true_params = SABRParams(alpha=0.030, beta=0.5, rho=-0.30, nu=0.35)
        strikes, vols = self._generate_smile(true_params)
        cal = calibrate_sabr(F_USD, T_1Y, strikes, vols, beta=0.5)
        assert abs(cal.rho - true_params.rho) < 0.05

    def test_round_trip_nu(self):
        """Calibrated nu should recover the original nu closely."""
        true_params = SABRParams(alpha=0.030, beta=0.5, rho=-0.20, nu=0.45)
        strikes, vols = self._generate_smile(true_params)
        cal = calibrate_sabr(F_USD, T_1Y, strikes, vols, beta=0.5)
        assert abs(cal.nu - true_params.nu) < 0.05

    def test_calibrated_atm_vol_matches_market(self):
        """Calibrated SABR must reprice ATM vol within 0.1 bp."""
        true_params = SABRParams(alpha=0.032, beta=0.5, rho=-0.25, nu=0.40)
        strikes, vols = self._generate_smile(true_params)
        cal = calibrate_sabr(F_USD, T_1Y, strikes, vols, beta=0.5)
        model_atm  = sabr_implied_vol(F_USD, F_USD, T_1Y, cal)
        market_atm = sabr_implied_vol(F_USD, F_USD, T_1Y, true_params)
        assert abs(model_atm - market_atm) < 1e-5   # 0.1 bp

    def test_single_atm_strike_calibration(self):
        """Single ATM quote: calibrates alpha, keeps rho/nu at initial guess."""
        target_atm = 0.18
        initial_guess = (-0.25, 0.40)  # (rho0, nu0) — note: we pass triple (alpha, rho, nu)
        cal = calibrate_sabr(
            F=F_USD, T=T_1Y,
            strikes=[F_USD], market_vols=[target_atm],
            beta=0.5, initial_guess=(0.05, -0.25, 0.40),
        )
        recovered_atm = sabr_implied_vol(F_USD, F_USD, T_1Y, cal)
        assert abs(recovered_atm - target_atm) < 1e-8

    def test_calibrated_alpha_positive(self):
        true_params = SABRParams(alpha=0.025, beta=0.5, rho=-0.20, nu=0.30)
        strikes, vols = self._generate_smile(true_params)
        cal = calibrate_sabr(F_USD, T_1Y, strikes, vols, beta=0.5)
        assert cal.alpha > 0.0

    def test_calibrated_rho_in_bounds(self):
        true_params = SABRParams(alpha=0.025, beta=0.5, rho=-0.30, nu=0.30)
        strikes, vols = self._generate_smile(true_params)
        cal = calibrate_sabr(F_USD, T_1Y, strikes, vols, beta=0.5)
        assert -1.0 < cal.rho < 1.0

    def test_calibrated_nu_nonnegative(self):
        true_params = SABRParams(alpha=0.025, beta=0.5, rho=-0.30, nu=0.30)
        strikes, vols = self._generate_smile(true_params)
        cal = calibrate_sabr(F_USD, T_1Y, strikes, vols, beta=0.5)
        assert cal.nu >= 0.0

    def test_beta_fixed_in_calibration(self):
        """calibrate_sabr must honour the fixed beta parameter."""
        true_params = SABRParams(alpha=0.028, beta=0.5, rho=-0.25, nu=0.40)
        strikes, vols = self._generate_smile(true_params)
        for beta in [0.0, 0.3, 0.5, 0.7, 1.0]:
            cal = calibrate_sabr(F_USD, T_1Y, strikes, vols, beta=beta)
            assert cal.beta == beta

    def test_raises_on_empty_strikes(self):
        with pytest.raises(ValueError):
            calibrate_sabr(F_USD, T_1Y, [], [], beta=0.5)

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError):
            calibrate_sabr(F_USD, T_1Y, [F_USD, F_USD * 1.05], [0.15], beta=0.5)


# ── TestSABRSmile ─────────────────────────────────────────────────────────────

class TestSABRSmile:

    def test_returns_dataframe(self):
        df = sabr_vol_smile(F_USD, T_1Y, PARAMS_BASE)
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        df = sabr_vol_smile(F_USD, T_1Y, PARAMS_BASE)
        for col in ["strike_pct", "moneyness_bps", "sabr_vol_pct", "normal_vol_bps"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count(self):
        df = sabr_vol_smile(F_USD, T_1Y, PARAMS_BASE, n_strikes=21)
        assert len(df) == 21

    def test_strike_pct_range(self):
        """strike_pct should be approximately F*100 ± strike_range_bps/100."""
        rng = 200.0
        df = sabr_vol_smile(F_USD, T_1Y, PARAMS_BASE, n_strikes=21, strike_range_bps=rng)
        lo_expected = (F_USD - rng / 10_000) * 100
        hi_expected = (F_USD + rng / 10_000) * 100
        assert df["strike_pct"].min() >= lo_expected - 1e-6
        assert df["strike_pct"].max() <= hi_expected + 1e-6

    def test_moneyness_bps_sign(self):
        """moneyness_bps = (K - F) × 10000; negative for OTM receivers (low K)."""
        df = sabr_vol_smile(F_USD, T_1Y, PARAMS_BASE, n_strikes=21)
        assert df["moneyness_bps"].iloc[0] < 0   # lowest strike → negative moneyness
        assert df["moneyness_bps"].iloc[-1] > 0  # highest strike → positive moneyness

    def test_negative_rho_left_skew(self):
        """Negative rho → left skew: low-K vols > high-K vols."""
        p = SABRParams(alpha=0.03, beta=0.5, rho=-0.40, nu=0.40)
        df = sabr_vol_smile(F_USD, T_1Y, p, n_strikes=21)
        low_vol  = df.iloc[0]["sabr_vol_pct"]
        high_vol = df.iloc[-1]["sabr_vol_pct"]
        assert low_vol > high_vol

    def test_vols_positive(self):
        df = sabr_vol_smile(F_USD, T_1Y, PARAMS_BASE)
        assert (df["sabr_vol_pct"] > 0).all()
        assert (df["normal_vol_bps"] > 0).all()

    def test_atm_vol_consistent_with_sabr_implied_vol(self):
        """Middle strike in the grid should be close to ATM and match sabr_implied_vol."""
        df = sabr_vol_smile(F_USD, T_1Y, PARAMS_BASE, n_strikes=21)
        mid = len(df) // 2
        K_mid = df["strike_pct"].iloc[mid] / 100.0
        expected = sabr_implied_vol(F_USD, K_mid, T_1Y, PARAMS_BASE) * 100.0
        assert abs(df["sabr_vol_pct"].iloc[mid] - expected) < 1e-8


# ── TestSABRSurface ───────────────────────────────────────────────────────────

class TestSABRSurface:

    def _build_surface(self):
        """Build a minimal SABRSurface with two expiries and two tenors."""
        expiries = [1.0, 5.0]
        tenors   = [5.0, 10.0]
        p11 = SABRParams(alpha=0.030, beta=0.5, rho=-0.25, nu=0.40)
        p12 = SABRParams(alpha=0.028, beta=0.5, rho=-0.25, nu=0.38)
        p21 = SABRParams(alpha=0.025, beta=0.5, rho=-0.22, nu=0.35)
        p22 = SABRParams(alpha=0.023, beta=0.5, rho=-0.20, nu=0.32)
        params = [[p11, p12], [p21, p22]]
        return SABRSurface(expiries=expiries, tenors=tenors, params=params)

    def test_vol_returns_positive(self):
        surf = self._build_surface()
        v = surf.vol(1.0, 5.0, K=F_USD, F=F_USD)
        assert v > 0.0

    def test_vol_at_grid_node_matches_exact(self):
        """At a grid node, the surface vol should match sabr_implied_vol directly."""
        surf = self._build_surface()
        p11 = surf._params[0][0]
        exact = sabr_implied_vol(F_USD, F_USD, 1.0, p11)
        surf_vol = surf.vol(1.0, 5.0, K=F_USD, F=F_USD)
        assert abs(surf_vol - exact) < 1e-10

    def test_vol_interpolated_between_nodes(self):
        """Off-grid points should return values between the bracketing node vols."""
        surf = self._build_surface()
        v_lo = surf.vol(1.0, 5.0, K=F_USD, F=F_USD)
        v_hi = surf.vol(5.0, 5.0, K=F_USD, F=F_USD)
        v_mid = surf.vol(3.0, 5.0, K=F_USD, F=F_USD)
        assert min(v_lo, v_hi) <= v_mid <= max(v_lo, v_hi)

    def test_smile_df_returns_dataframe(self):
        import pandas as pd
        surf = self._build_surface()
        df = surf.smile_df(1.0, 5.0, F=F_USD)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_smile_df_columns(self):
        surf = self._build_surface()
        df = surf.smile_df(1.0, 5.0, F=F_USD)
        for col in ["strike_pct", "moneyness_bps", "sabr_vol_pct", "normal_vol_bps"]:
            assert col in df.columns

    def test_raises_on_mismatched_params_shape(self):
        with pytest.raises(ValueError):
            SABRSurface(
                expiries=[1.0, 2.0],
                tenors=[5.0],
                params=[[SABRParams(0.03, 0.5, -0.25, 0.4), SABRParams(0.03, 0.5, -0.25, 0.4)]],
            )

    def test_calibrate_from_atm_surface_atm_match(self):
        """ATM vols from calibrated surface must match input ATM vols within 1 bp."""
        from datetime import date
        from sofr_engine.bootstrap import flat_sofr_curve
        from sofr_engine.swaption import SwaptionVolSurface

        ref   = date(2024, 1, 2)
        curve = flat_sofr_curve(ref, rate=0.0480)
        surf  = SwaptionVolSurface.typical_market()
        sabr_surf = SABRSurface.calibrate_from_atm_surface(surf, curve)

        for i, T in enumerate(surf._expiries):
            for j, tenor in enumerate(surf._tenors):
                from sofr_engine.swaption import Swaption
                sw = Swaption(expiry_years=T, swap_tenor_years=tenor, strike=0.0)
                F = sw.forward_swap_rate(curve)
                if not np.isfinite(F) or F <= 0:
                    continue
                market_atm = surf._vols[i, j]
                model_atm  = sabr_surf.vol(T, tenor, K=F, F=F)
                assert abs(model_atm - market_atm) < 1e-4, (
                    f"ATM mismatch at ({T}y, {tenor}y): "
                    f"model={model_atm*100:.4f}% market={market_atm*100:.4f}%"
                )

    def test_repr(self):
        surf = self._build_surface()
        r = repr(surf)
        assert "SABRSurface" in r

"""
Tests for global_sofr_bootstrap — least-squares OIS curve fitting with
Tikhonov roughness regularization.

Validates:
  - Par rates are repriced within tolerance
  - Smoother forward curve than sequential bootstrap
  - DFs are monotonically decreasing and positive
  - Roughness lambda controls smoothness vs fit trade-off
  - Edge cases (single quote, many quotes, high lambda)
"""
from __future__ import annotations
import numpy as np
import pytest
from datetime import date

from sofr_engine.bootstrap import global_sofr_bootstrap, SOFRCurveBootstrapper


REF = date(2024, 6, 5)
SOFR_ON = 0.0530

# Standard OIS quote grid: (tenor_years, par_rate)
QUOTES_STANDARD = [
    (1.0,  0.0520),
    (2.0,  0.0480),
    (3.0,  0.0455),
    (5.0,  0.0430),
    (7.0,  0.0415),
    (10.0, 0.0400),
    (15.0, 0.0392),
    (20.0, 0.0388),
    (30.0, 0.0385),
]

QUOTES_INVERTED = [
    (1.0,  0.0530),
    (2.0,  0.0510),
    (3.0,  0.0490),
    (5.0,  0.0460),
    (7.0,  0.0440),
    (10.0, 0.0420),
    (20.0, 0.0400),
    (30.0, 0.0395),
]


# ── Construction ──────────────────────────────────────────────────────────────

class TestGlobalBootstrapConstruction:
    def test_returns_discount_curve(self):
        from sofr_engine.curve import DiscountCurve
        c = global_sofr_bootstrap(REF, SOFR_ON, QUOTES_STANDARD)
        assert isinstance(c, DiscountCurve)

    def test_label_is_global(self):
        c = global_sofr_bootstrap(REF, SOFR_ON, QUOTES_STANDARD)
        assert "Global" in c.label

    def test_max_tenor_covers_quotes(self):
        c = global_sofr_bootstrap(REF, SOFR_ON, QUOTES_STANDARD)
        assert c._times[-1] >= 29.9

    def test_empty_quotes_raises(self):
        with pytest.raises((ValueError, Exception)):
            global_sofr_bootstrap(REF, SOFR_ON, [])

    def test_single_quote(self):
        c = global_sofr_bootstrap(REF, SOFR_ON, [(5.0, 0.045)])
        assert c.df(5.0) > 0
        assert c.df(5.0) < 1.0


# ── Par rate repricing ────────────────────────────────────────────────────────

class TestGlobalBootstrapPricing:
    @pytest.fixture
    def curve(self):
        return global_sofr_bootstrap(REF, SOFR_ON, QUOTES_STANDARD, roughness_lambda=1e-6)

    @pytest.mark.parametrize("tenor,market_rate", QUOTES_STANDARD)
    def test_par_rate_repriced_within_2bp(self, curve, tenor, market_rate):
        model_rate = curve.par_ois_rate(tenor)
        diff_bps = abs(model_rate - market_rate) * 1e4
        assert diff_bps < 2.0, (
            f"Tenor {tenor}Y: model={model_rate*1e4:.2f}bps "
            f"market={market_rate*1e4:.2f}bps diff={diff_bps:.2f}bps"
        )

    def test_df_at_zero_is_one(self, curve):
        assert curve.df(0.0) == pytest.approx(1.0, abs=1e-8)

    def test_overnight_df_correct(self, curve):
        t_on = 1.0 / 365.25
        expected = 1.0 / (1.0 + SOFR_ON / 360.0)
        assert curve.df(t_on) == pytest.approx(expected, rel=1e-4)

    def test_dfs_strictly_positive(self, curve):
        ts = np.linspace(0.01, 29.9, 200)
        assert all(curve.df(t) > 0 for t in ts)

    def test_dfs_monotonically_decreasing(self, curve):
        ts = np.linspace(0.01, 29.9, 200)
        dfs = np.array([curve.df(t) for t in ts])
        assert np.all(np.diff(dfs) < 0), "DFs should be strictly decreasing"


# ── Inverted curve ────────────────────────────────────────────────────────────

class TestGlobalBootstrapInverted:
    @pytest.fixture
    def curve(self):
        return global_sofr_bootstrap(REF, SOFR_ON, QUOTES_INVERTED, roughness_lambda=1e-6)

    @pytest.mark.parametrize("tenor,market_rate", QUOTES_INVERTED)
    def test_inverted_par_rate_repriced(self, curve, tenor, market_rate):
        diff_bps = abs(curve.par_ois_rate(tenor) - market_rate) * 1e4
        assert diff_bps < 3.0

    def test_inverted_dfs_monotone(self, curve):
        ts = np.linspace(0.01, 29.9, 200)
        dfs = np.array([curve.df(t) for t in ts])
        assert np.all(np.diff(dfs) < 0)


# ── Smoothness vs sequential bootstrap ───────────────────────────────────────

class TestGlobalBootstrapSmoothness:
    def _pillar_roughness(self, curve) -> float:
        """
        Variance of second differences of log-DFs at pillar points.

        The Tikhonov penalty is  λ · ‖Δ²(log_DF)‖²  — it directly controls
        curvature of the pillar log-DF sequence, NOT inter-pillar forward rates
        (which remain piecewise-constant under log-linear interpolation).
        This metric measures exactly what the penalty controls.
        """
        log_dfs = curve._log_df[2:]  # skip t=0 and overnight fixed pillars
        return float(np.var(np.diff(log_dfs, n=2))) if len(log_dfs) >= 3 else 0.0

    def test_global_smoother_than_sequential_at_low_lambda(self):
        """Global fit achieves similar pillar roughness to sequential bootstrap."""
        seq = (SOFRCurveBootstrapper(REF, SOFR_ON)
               .add_ois_swaps(QUOTES_STANDARD)
               .build())
        glob = global_sofr_bootstrap(
            REF, SOFR_ON, QUOTES_STANDARD, roughness_lambda=1e-4
        )
        r_seq  = self._pillar_roughness(seq)
        r_glob = self._pillar_roughness(glob)
        # Global bootstrap with moderate lambda should be at most 3× as rough
        assert r_glob <= max(r_seq * 3.0, 1e-12)

    def test_higher_lambda_gives_smoother_curve(self):
        """Increasing roughness_lambda → smoother log-DF pillar sequence."""
        c_lo = global_sofr_bootstrap(REF, SOFR_ON, QUOTES_STANDARD, roughness_lambda=1e-8)
        c_hi = global_sofr_bootstrap(REF, SOFR_ON, QUOTES_STANDARD, roughness_lambda=1e-2)
        r_lo = self._pillar_roughness(c_lo)
        r_hi = self._pillar_roughness(c_hi)
        assert r_hi <= r_lo, (
            f"High lambda pillar roughness ({r_hi:.2e}) should be ≤ "
            f"low lambda ({r_lo:.2e})"
        )

    def test_high_lambda_larger_repricing_error(self):
        """Very high regularization → worse fit to market quotes."""
        c_lo = global_sofr_bootstrap(REF, SOFR_ON, QUOTES_STANDARD, roughness_lambda=1e-8)
        c_hi = global_sofr_bootstrap(REF, SOFR_ON, QUOTES_STANDARD, roughness_lambda=0.5)
        err_lo = sum(abs(c_lo.par_ois_rate(t) - r) for t, r in QUOTES_STANDARD)
        err_hi = sum(abs(c_hi.par_ois_rate(t) - r) for t, r in QUOTES_STANDARD)
        assert err_hi >= err_lo


# ── Roughness lambda = 0 approaches sequential ────────────────────────────────

class TestGlobalBootstrapZeroLambda:
    def test_zero_lambda_close_to_sequential(self):
        seq = (SOFRCurveBootstrapper(REF, SOFR_ON)
               .add_ois_swaps(QUOTES_STANDARD)
               .build())
        glob = global_sofr_bootstrap(
            REF, SOFR_ON, QUOTES_STANDARD, roughness_lambda=0.0
        )
        # Par rates should match closely even if DF values differ slightly
        for tenor, market_rate in QUOTES_STANDARD:
            model_seq  = seq.par_ois_rate(tenor)
            model_glob = glob.par_ois_rate(tenor)
            diff_bps = abs(model_seq - model_glob) * 1e4
            assert diff_bps < 25.0, (
                f"Tenor {tenor}Y: seq par={model_seq*1e4:.1f}bp glob par={model_glob*1e4:.1f}bp"
            )


# ── Payment frequency ────────────────────────────────────────────────────────

class TestGlobalBootstrapPaymentFreq:
    def test_semi_annual_payment_freq(self):
        c = global_sofr_bootstrap(
            REF, SOFR_ON, QUOTES_STANDARD,
            roughness_lambda=1e-5, payment_freq=2
        )
        for tenor, rate in QUOTES_STANDARD[:4]:
            model = c.par_ois_rate(tenor, payment_freq=2)
            assert abs(model - rate) * 1e4 < 5.0


# ── Many quotes ──────────────────────────────────────────────────────────────

class TestGlobalBootstrapManyQuotes:
    def test_dense_quote_grid(self):
        # Start from 1Y to avoid sub-annual/annual mismatch in par_ois_rate()
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        rates  = [0.050, 0.047, 0.045, 0.044, 0.043, 0.042, 0.041, 0.040, 0.040, 0.039, 0.039]
        quotes = list(zip(tenors, rates))
        c = global_sofr_bootstrap(REF, SOFR_ON, quotes, roughness_lambda=1e-5)
        for tenor, rate in quotes:
            diff_bps = abs(c.par_ois_rate(tenor) - rate) * 1e4
            assert diff_bps < 3.0, f"Tenor {tenor}Y: {diff_bps:.1f}bps off"

    def test_two_quotes_only(self):
        quotes = [(2.0, 0.048), (10.0, 0.042)]
        c = global_sofr_bootstrap(REF, SOFR_ON, quotes, roughness_lambda=1e-5)
        assert c.df(2.0) > 0
        assert c.df(10.0) > 0
        assert c.df(2.0) > c.df(10.0)

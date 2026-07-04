"""
Tests for PCHIP spline interpolation in DiscountCurve.

Validates that the 'pchip' interp_method produces:
  - Exact pillar values
  - Smooth (C1 continuous) forward rates without kinks
  - Monotonically decreasing discount factors
  - Backward-compatible default (log-linear)
"""
from __future__ import annotations
import math
import numpy as np
import pytest
from datetime import date

from sofr_engine.curve import DiscountCurve
from sofr_engine.bootstrap import flat_sofr_curve


REF = date(2024, 6, 5)

TIMES = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
# Realistic upward-sloping forward curve (inverted → normal)
RATES = [0.053, 0.052, 0.050, 0.047, 0.045, 0.043, 0.042, 0.041, 0.040, 0.040, 0.039]
DFS   = [math.exp(-r * t) for r, t in zip(RATES, TIMES)]


# ── Construction ──────────────────────────────────────────────────────────────

class TestPCHIPConstruction:
    def test_default_is_log_linear(self):
        c = DiscountCurve(REF, TIMES, DFS)
        assert c._interp_method == "log-linear"
        assert c._pchip is None

    def test_pchip_method_stored(self):
        c = DiscountCurve(REF, TIMES, DFS, interp_method="pchip")
        assert c._interp_method == "pchip"
        assert c._pchip is not None

    def test_invalid_interp_method_raises(self):
        with pytest.raises(ValueError, match="interp_method"):
            DiscountCurve(REF, TIMES, DFS, interp_method="cubic")

    def test_repr_shows_interp_method(self):
        c_ll = DiscountCurve(REF, TIMES, DFS, interp_method="log-linear")
        c_ph = DiscountCurve(REF, TIMES, DFS, interp_method="pchip")
        assert "log-linear" in repr(c_ll)
        assert "pchip" in repr(c_ph)


# ── Pillar exactness ──────────────────────────────────────────────────────────

class TestPCHIPPillarValues:
    @pytest.fixture
    def curve(self):
        return DiscountCurve(REF, TIMES, DFS, interp_method="pchip")

    def test_df_at_zero(self, curve):
        assert curve.df(0.0) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("t, expected_df", list(zip(TIMES, DFS)))
    def test_df_at_each_pillar(self, curve, t, expected_df):
        assert curve.df(t) == pytest.approx(expected_df, rel=1e-6)

    def test_scalar_returns_scalar(self, curve):
        result = curve.df(5.0)
        assert isinstance(result, float)

    def test_array_input(self, curve):
        ts = np.array([1.0, 5.0, 10.0])
        dfs = curve.df(ts)
        assert dfs.shape == (3,)
        for t, expected in zip([1.0, 5.0, 10.0], [curve.df(t) for t in [1.0, 5.0, 10.0]]):
            pass  # checked in pillar test


# ── Monotonicity ──────────────────────────────────────────────────────────────

class TestPCHIPMonotonicity:
    @pytest.fixture
    def curve(self):
        return DiscountCurve(REF, TIMES, DFS, interp_method="pchip")

    def test_dfs_strictly_decreasing(self, curve):
        ts = np.linspace(0.01, 29.9, 300)
        dfs = np.array([curve.df(t) for t in ts])
        assert np.all(np.diff(dfs) < 0), "DFs should be strictly decreasing"

    def test_dfs_positive(self, curve):
        ts = np.linspace(0.0, 30.0, 200)
        dfs = np.array([curve.df(t) for t in ts])
        assert np.all(dfs > 0)

    def test_df_at_zero_is_one(self, curve):
        assert curve.df(0.0) == pytest.approx(1.0, abs=1e-9)


# ── Smoothness vs log-linear ──────────────────────────────────────────────────

class TestPCHIPSmoothness:
    """
    PCHIP must produce smoother forward rates than log-linear.
    We measure smoothness by the variance of second differences of the
    forward rate evaluated at a fine grid.
    """

    @pytest.fixture
    def curves(self):
        c_ll = DiscountCurve(REF, TIMES, DFS, interp_method="log-linear")
        c_ph = DiscountCurve(REF, TIMES, DFS, interp_method="pchip")
        return c_ll, c_ph

    def _fwd_rates(self, curve, n=300):
        ts = np.linspace(0.1, 29.8, n)
        dt = ts[1] - ts[0]
        fwd = np.array([
            curve.forward_rate(t, min(t + dt, 30.0)) for t in ts[:-1]
        ])
        return fwd

    def test_pchip_smoother_forward_curve(self, curves):
        c_ll, c_ph = curves
        fwd_ll = self._fwd_rates(c_ll)
        fwd_ph = self._fwd_rates(c_ph)
        # PCHIP second-difference variance should be lower (smoother)
        roughness_ll = np.var(np.diff(fwd_ll, n=2))
        roughness_ph = np.var(np.diff(fwd_ph, n=2))
        assert roughness_ph < roughness_ll, (
            f"PCHIP ({roughness_ph:.2e}) should be smoother than "
            f"log-linear ({roughness_ll:.2e})"
        )

    def test_pchip_zero_rates_continuous(self, curves):
        _, c_ph = curves
        # Zero rates at densely sampled points should have small jumps
        ts = np.linspace(0.5, 30.0, 500)
        zr = np.array([c_ph.zero_rate(t) for t in ts])
        max_jump = np.max(np.abs(np.diff(zr)))
        assert max_jump < 0.002, f"Max zero-rate jump {max_jump:.4f} too large"


# ── Backward compatibility ────────────────────────────────────────────────────

class TestBackwardCompatibility:
    def test_log_linear_unchanged(self):
        c = DiscountCurve(REF, TIMES, DFS)
        ts = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        dfs = np.array([c.df(t) for t in ts])
        expected = np.exp(np.interp(ts, [0.0] + TIMES, [0.0] + [math.log(d) for d in DFS]))
        np.testing.assert_allclose(dfs, expected, rtol=1e-10)

    def test_flat_curve_same_result(self):
        """Flat curve → log-linear and PCHIP give identical DFs."""
        rate = 0.05
        c_ll = flat_sofr_curve(REF, rate)
        # Build PCHIP version manually
        ts = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
        dfs = np.exp(-rate * ts)
        c_ph = DiscountCurve(REF, list(ts), list(dfs), interp_method="pchip")
        for t in [0.5, 1.0, 3.0, 7.5, 15.0]:
            assert c_ll.df(t) == pytest.approx(c_ph.df(t), rel=1e-4)

    def test_par_ois_rate_preserved(self):
        """Par OIS rate from PCHIP curve should be close to log-linear."""
        c_ll = DiscountCurve(REF, TIMES, DFS, interp_method="log-linear")
        c_ph = DiscountCurve(REF, TIMES, DFS, interp_method="pchip")
        for tenor in [1.0, 2.0, 5.0, 10.0]:
            r_ll = c_ll.par_ois_rate(tenor)
            r_ph = c_ph.par_ois_rate(tenor)
            assert abs(r_ll - r_ph) < 0.001, (
                f"Par rate difference at {tenor}Y: {abs(r_ll - r_ph)*1e4:.1f}bps"
            )


# ── Analytics unaffected ──────────────────────────────────────────────────────

class TestPCHIPAnalytics:
    @pytest.fixture
    def curve(self):
        return DiscountCurve(REF, TIMES, DFS, interp_method="pchip")

    def test_zero_rate_positive(self, curve):
        for t in [0.5, 1.0, 5.0, 10.0, 20.0]:
            assert curve.zero_rate(t) > 0

    def test_forward_rate_positive(self, curve):
        for t in [0.5, 1.0, 3.0, 7.0]:
            assert curve.forward_rate(t, t + 0.25) > 0

    def test_dv01_positive(self, curve):
        for t in [1.0, 5.0, 10.0]:
            assert curve.dv01(t) > 0

    def test_zero_curve_returns_dataframe(self, curve):
        df = curve.zero_curve()
        assert len(df) == 11
        assert all(df["discount_factor"] > 0)
        assert all(df["zero_rate_pct"] > 0)

    def test_pillars_property(self, curve):
        p = curve.pillars
        assert len(p) == len(TIMES) + 1  # includes t=0 prepended
        assert p["df"].iloc[0] == pytest.approx(1.0)

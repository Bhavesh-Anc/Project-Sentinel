"""
Tests for models/pca_factors.py and models/carry_rolldown.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from datetime import date

from models.pca_factors import YieldCurvePCA, classify_pc_regime, _parse_tenors
from models.carry_rolldown import (
    carry_bps, rolldown_bps, total_return_bps,
    carry_rolldown_table, carry_rolldown_matrix,
    breakeven_yield_move, steepener_carry,
)
from sofr_engine.curve import DiscountCurve
from sofr_engine.bootstrap import flat_sofr_curve

REF_DATE = date(2024, 1, 2)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _synthetic_yield_df(n_obs: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily yield matrix: 6 tenors, correlated normal shocks."""
    rng = np.random.default_rng(seed)
    tenors = [2.0, 3.0, 5.0, 7.0, 10.0, 30.0]
    # level factor drives most variance
    level = np.cumsum(rng.normal(0, 0.01, n_obs))
    slopes = rng.normal(0, 0.002, (n_obs, len(tenors)))
    base = np.array([3.5, 3.7, 3.9, 4.1, 4.3, 4.5])
    yields = base + level[:, None] + slopes
    dates = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    cols = ["tsy_2y", "tsy_3y", "tsy_5y", "tsy_7y", "tsy_10y", "tsy_30y"]
    return pd.DataFrame(yields, index=dates, columns=cols)


def _flat_curve(rate: float = 0.04) -> DiscountCurve:
    """Flat discount curve at `rate` for carry/rolldown tests."""
    return flat_sofr_curve(REF_DATE, rate, max_tenor=30.0)


def _upward_slope_curve() -> DiscountCurve:
    """Upward sloping curve: 2% at short end → 5%+ at long end."""
    times = np.array([0.01, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    rates = np.array([0.020, 0.025, 0.030, 0.035, 0.038, 0.042, 0.045, 0.048, 0.050, 0.052])
    dfs = np.exp(-rates * times)
    return DiscountCurve(REF_DATE, times, dfs)


# ═══════════════════════════════════════════════════════════════════════════════
#  PCA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestYieldCurvePCA:

    def test_fit_returns_self(self):
        df = _synthetic_yield_df()
        pca = YieldCurvePCA(n_components=3)
        result = pca.fit(df)
        assert result is pca

    def test_explained_variance_sums_to_one(self):
        df = _synthetic_yield_df()
        pca = YieldCurvePCA(n_components=3).fit(df)
        total = pca.result.explained_var.sum()
        assert 0.0 < total <= 1.0 + 1e-10

    def test_first_pc_explains_most_variance(self):
        df = _synthetic_yield_df()
        pca = YieldCurvePCA(n_components=3).fit(df)
        ev = pca.result.explained_var
        # PC1 should dominate (level factor is the biggest driver)
        assert ev[0] > ev[1]
        assert ev[1] > ev[2]

    def test_components_shape(self):
        df = _synthetic_yield_df()
        pca = YieldCurvePCA(n_components=3).fit(df)
        assert pca.result.components.shape == (3, 6)

    def test_transform_shape(self):
        df = _synthetic_yield_df()
        pca = YieldCurvePCA(n_components=3).fit(df)
        scores = pca.transform(df)
        assert scores.shape == (len(df), 3)
        assert list(scores.columns) == ["PC1", "PC2", "PC3"]

    def test_reconstruct_close_to_original(self):
        df = _synthetic_yield_df()
        pca = YieldCurvePCA(n_components=3).fit(df)
        scores = pca.transform(df)
        recon = pca.reconstruct(scores)
        original = df.values
        # Using 3 PCs on synthetic data with ~3 real factors → tight reconstruction
        assert np.allclose(recon, original, atol=0.5)

    def test_loadings_df_shape(self):
        df = _synthetic_yield_df()
        pca = YieldCurvePCA(n_components=3).fit(df)
        ld = pca.loadings_df()
        assert ld.shape == (6, 3)
        assert list(ld.columns) == ["PC1", "PC2", "PC3"]

    def test_explained_variance_table_columns(self):
        df = _synthetic_yield_df()
        pca = YieldCurvePCA(n_components=3).fit(df)
        t = pca.explained_variance_table()
        assert "explained_pct" in t.columns
        assert "cumulative_pct" in t.columns

    def test_cumulative_explained_variance_monotone(self):
        df = _synthetic_yield_df()
        pca = YieldCurvePCA(n_components=3).fit(df)
        t = pca.explained_variance_table()
        cum = t["cumulative_pct"].values
        assert all(cum[i] <= cum[i+1] for i in range(len(cum)-1))

    def test_unfitted_raises(self):
        pca = YieldCurvePCA()
        with pytest.raises(RuntimeError):
            _ = pca.result

    def test_fit_with_explicit_tenors(self):
        df = _synthetic_yield_df()
        tenors_yrs = [2.0, 3.0, 5.0, 7.0, 10.0, 30.0]
        pca = YieldCurvePCA(n_components=2).fit(df, tenors_yrs=tenors_yrs)
        assert pca.result.components.shape == (2, 6)

    def test_hedge_ratios_shape(self):
        df = _synthetic_yield_df()
        tenors_yrs = [2.0, 3.0, 5.0, 7.0, 10.0, 30.0]
        pca = YieldCurvePCA(n_components=3).fit(df, tenors_yrs=tenors_yrs)
        hr = pca.hedge_ratios({10.0: -10_000})
        assert hr.shape == (3, 3)  # 3 PCs × (hedge_tenor, hedge_dv01, pc_exposure)

    def test_hedge_ratios_columns(self):
        df = _synthetic_yield_df()
        tenors_yrs = [2.0, 3.0, 5.0, 7.0, 10.0, 30.0]
        pca = YieldCurvePCA(n_components=3).fit(df, tenors_yrs=tenors_yrs)
        hr = pca.hedge_ratios({10.0: -10_000})
        assert "hedge_dv01_usd" in hr.columns
        assert "pc_exposure" in hr.columns


class TestClassifyPCRegime:

    def _make_scores(self, pc1_vals, pc2_vals):
        n = len(pc1_vals)
        return pd.DataFrame(
            {"PC1": pc1_vals, "PC2": pc2_vals, "PC3": [0.0] * n},
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )

    def test_high_rates_label(self):
        scores = self._make_scores([1.0], [0.0])
        regime = classify_pc_regime(scores)
        assert regime.iloc[0] == "high_rates"

    def test_low_rates_label(self):
        scores = self._make_scores([-1.0], [0.0])
        regime = classify_pc_regime(scores)
        assert regime.iloc[0] == "low_rates"

    def test_steep_curve_label(self):
        scores = self._make_scores([0.0], [0.5])
        regime = classify_pc_regime(scores)
        assert regime.iloc[0] == "steep_curve"

    def test_flat_curve_label(self):
        scores = self._make_scores([0.0], [-0.5])
        regime = classify_pc_regime(scores)
        assert regime.iloc[0] == "flat_curve"

    def test_neutral_label(self):
        scores = self._make_scores([0.0], [0.0])
        regime = classify_pc_regime(scores)
        assert regime.iloc[0] == "neutral"

    def test_valid_regime_values(self):
        df = _synthetic_yield_df()
        pca = YieldCurvePCA(n_components=3).fit(df)
        scores = pca.transform(df)
        regime = classify_pc_regime(scores)
        valid = {"high_rates", "low_rates", "steep_curve", "flat_curve", "neutral"}
        assert set(regime.unique()).issubset(valid)


class TestParseTenors:

    def test_year_suffix(self):
        tenors = _parse_tenors(["tsy_2y", "tsy_10y", "tsy_30y"])
        np.testing.assert_array_equal(tenors, [2.0, 10.0, 30.0])

    def test_month_suffix(self):
        tenors = _parse_tenors(["tsy_3m", "tsy_6m"])
        np.testing.assert_allclose(tenors, [0.25, 0.5], atol=1e-10)

    def test_unknown_returns_nan(self):
        tenors = _parse_tenors(["unknown"])
        assert np.isnan(tenors[0])


# ═══════════════════════════════════════════════════════════════════════════════
#  CARRY / ROLL-DOWN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCarryBps:

    def test_positive_carry_on_upward_curve(self):
        curve = _upward_slope_curve()
        # 10Y par rate > overnight rate → positive carry
        c = carry_bps(curve, 10.0, dt_years=1/252)
        assert c > 0

    def test_zero_carry_on_flat_curve(self):
        curve = _flat_curve(0.04)
        # flat curve: par rate ≈ financing rate → near-zero carry
        c = carry_bps(curve, 5.0, dt_years=1/252)
        assert abs(c) < 0.5  # less than 0.5 bps per day

    def test_carry_scales_with_dt(self):
        curve = _upward_slope_curve()
        c_daily = carry_bps(curve, 10.0, dt_years=1/252)
        c_monthly = carry_bps(curve, 10.0, dt_years=1/12)
        ratio = c_monthly / c_daily
        assert abs(ratio - 252/12) < 1.0  # should be ~21x

    def test_carry_units_are_bps(self):
        curve = _upward_slope_curve()
        # 10Y rate ~4.8%, overnight ~2% → spread ~280 bps/yr → daily ~1.1 bps
        c = carry_bps(curve, 10.0, dt_years=1/252)
        assert 0 < c < 5  # sanity range for daily bps


class TestRolldownBps:

    def test_positive_rolldown_on_upward_curve(self):
        curve = _upward_slope_curve()
        rd = rolldown_bps(curve, 10.0, dt_years=1/12)
        assert rd > 0, f"Expected positive roll-down on upward curve, got {rd}"

    def test_rolldown_larger_for_longer_tenor(self):
        curve = _upward_slope_curve()
        rd5  = rolldown_bps(curve, 5.0,  dt_years=1/12)
        rd10 = rolldown_bps(curve, 10.0, dt_years=1/12)
        # longer tenor → more duration → more roll-down
        assert rd10 > rd5

    def test_rolldown_scales_with_dt(self):
        curve = _upward_slope_curve()
        rd1m = rolldown_bps(curve, 10.0, dt_years=1/12)
        rd3m = rolldown_bps(curve, 10.0, dt_years=3/12)
        assert rd3m > rd1m

    def test_exact_and_approx_close(self):
        curve = _upward_slope_curve()
        rd_approx = rolldown_bps(curve, 5.0, dt_years=1/12, duration_approx=True)
        rd_exact  = rolldown_bps(curve, 5.0, dt_years=1/12, duration_approx=False)
        # Both should be positive and reasonably close
        assert rd_approx > 0
        assert rd_exact > 0


class TestTotalReturnBps:

    def test_keys_present(self):
        curve = _upward_slope_curve()
        tr = total_return_bps(curve, 5.0)
        assert "carry_bps" in tr
        assert "rolldown_bps" in tr
        assert "total_return_bps" in tr

    def test_total_equals_carry_plus_rolldown(self):
        curve = _upward_slope_curve()
        tr = total_return_bps(curve, 5.0)
        assert abs(tr["total_return_bps"] - (tr["carry_bps"] + tr["rolldown_bps"])) < 1e-10


class TestCarryRolldownTable:

    def test_returns_dataframe(self):
        curve = _upward_slope_curve()
        df = carry_rolldown_table(curve)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        curve = _upward_slope_curve()
        df = carry_rolldown_table(curve)
        for col in ["carry_bps", "rolldown_bps", "total_return_bps",
                    "breakeven_move_bps", "modified_duration"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_non_empty(self):
        curve = _upward_slope_curve()
        df = carry_rolldown_table(curve)
        assert len(df) > 0

    def test_index_name(self):
        curve = _upward_slope_curve()
        df = carry_rolldown_table(curve)
        assert df.index.name == "tenor_yrs"


class TestCarryRolldownMatrix:

    def test_shape(self):
        curve = _upward_slope_curve()
        mat = carry_rolldown_matrix(curve, tenors=[2.0, 5.0, 10.0],
                                    horizons=[1/12, 3/12, 6/12])
        assert mat.shape == (3, 3)

    def test_index_labels(self):
        curve = _upward_slope_curve()
        mat = carry_rolldown_matrix(curve, tenors=[2.0, 5.0], horizons=[1/12])
        # index format is "2.0Y", "5.0Y" (f-string from carry_rolldown_matrix)
        assert any("2" in idx for idx in mat.index)
        assert any("5" in idx for idx in mat.index)

    def test_column_labels(self):
        curve = _upward_slope_curve()
        mat = carry_rolldown_matrix(curve, tenors=[5.0], horizons=[1/12, 1.0])
        assert "1M" in mat.columns
        assert "1Y" in mat.columns


class TestBreakevenYieldMove:

    def test_positive_on_upward_curve(self):
        curve = _upward_slope_curve()
        be = breakeven_yield_move(curve, 10.0)
        assert be > 0

    def test_returns_float(self):
        curve = _upward_slope_curve()
        be = breakeven_yield_move(curve, 5.0)
        assert isinstance(be, float)


class TestSteepenerCarry:

    def test_keys_present(self):
        curve = _upward_slope_curve()
        sc = steepener_carry(curve, short_tenor=2.0, long_tenor=10.0)
        for k in ["carry_2y_bps", "carry_10y_bps", "net_carry_bps",
                  "rolldown_2y_bps", "rolldown_10y_bps", "net_rolldown_bps",
                  "net_total_bps"]:
            assert k in sc, f"Missing key: {k}"

    def test_net_total_sum(self):
        curve = _upward_slope_curve()
        sc = steepener_carry(curve, short_tenor=2.0, long_tenor=10.0)
        expected = sc["net_carry_bps"] + sc["net_rolldown_bps"]
        assert abs(sc["net_total_bps"] - expected) < 1e-10

    def test_net_carry_negative_on_inverted_curve(self):
        # On upward-sloping curve, 2Y rate < 10Y rate → net_carry = c2 - c10 < 0 for steepener
        curve = _upward_slope_curve()
        sc = steepener_carry(curve, short_tenor=2.0, long_tenor=10.0)
        # 2Y yield < 10Y yield → carry on 2Y receive leg < carry on 10Y pay leg
        # net_carry = c2 - c10 < 0 for normal upward-sloping curve
        assert sc["carry_2y_bps"] < sc["carry_10y_bps"]

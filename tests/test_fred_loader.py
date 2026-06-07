"""
Tests for data/fred_loader.py — all tests work without a FRED API key
by exercising the cache/synthetic fallback path.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from data.fred_loader import FREDLoader, _make_synthetic_rates

# Force synthetic fallback by using a known-bad key
LOADER = FREDLoader(api_key="bad_key_force_fallback")


# ── _make_synthetic_rates ────────────────────────────────────────────────────

class TestSyntheticRates:

    def test_returns_dataframe(self):
        df = _make_synthetic_rates("2022-01-01", "2023-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = _make_synthetic_rates()
        for col in ["tsy_2y", "tsy_10y", "sofr", "effr", "unemployment", "core_pce_yoy"]:
            assert col in df.columns

    def test_no_negative_values(self):
        df = _make_synthetic_rates("2020-01-01", "2024-01-01")
        assert (df >= 0).all().all()

    def test_reasonable_yield_range(self):
        df = _make_synthetic_rates()
        # Yields should stay in a plausible 0–15% range
        for col in ["tsy_2y", "tsy_5y", "tsy_10y", "tsy_30y"]:
            assert df[col].max() < 15.0
            assert df[col].min() >= 0.0

    def test_business_days_only(self):
        df = _make_synthetic_rates("2023-01-01", "2023-06-30")
        days_of_week = pd.Series(df.index).dt.dayofweek
        # dayofweek 5=Sat, 6=Sun — none should appear
        assert (days_of_week < 5).all()

    def test_reproducible_with_seed(self):
        df1 = _make_synthetic_rates("2022-01-01")
        df2 = _make_synthetic_rates("2022-01-01")
        pd.testing.assert_frame_equal(df1, df2)

    def test_date_range_respected(self):
        df = _make_synthetic_rates("2023-06-01", "2023-12-31")
        assert df.index.min() >= pd.Timestamp("2023-06-01")
        assert df.index.max() <= pd.Timestamp("2024-01-01")


# ── FREDLoader.treasury_yields ────────────────────────────────────────────────

class TestTreasuryYields:

    def test_returns_dataframe(self):
        df = LOADER.treasury_yields(start="2023-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = LOADER.treasury_yields(start="2023-01-01")
        for col in ["tsy_2y", "tsy_5y", "tsy_10y", "tsy_30y"]:
            assert col in df.columns

    def test_non_empty(self):
        df = LOADER.treasury_yields(start="2023-01-01")
        assert len(df) > 0

    def test_yields_positive(self):
        df = LOADER.treasury_yields(start="2023-01-01").dropna()
        # At least some values should be positive
        assert (df > 0).any().any()

    def test_monotone_term_structure_on_average(self):
        df = LOADER.treasury_yields(start="2023-01-01").dropna()
        if len(df) > 10:
            # On average 2Y < 10Y < 30Y (not always true for inverted periods)
            # Just check 2Y mean < 30Y mean within 200bps (structural check)
            assert df["tsy_2y"].mean() < df["tsy_30y"].mean() + 2.0


# ── FREDLoader.sofr_history ───────────────────────────────────────────────────

class TestSOFRHistory:

    def test_returns_series(self):
        s = LOADER.sofr_history(start="2023-01-01")
        assert isinstance(s, pd.Series)

    def test_series_name(self):
        s = LOADER.sofr_history(start="2023-01-01")
        assert s.name == "sofr"

    def test_non_empty(self):
        s = LOADER.sofr_history(start="2023-01-01")
        assert len(s) > 0

    def test_values_in_plausible_range(self):
        s = LOADER.sofr_history(start="2023-01-01").dropna()
        assert s.min() >= 0.0
        assert s.max() < 20.0


# ── FREDLoader.macro_snapshot ─────────────────────────────────────────────────

class TestMacroSnapshot:

    def test_returns_dict(self):
        snap = LOADER.macro_snapshot()
        assert isinstance(snap, dict)

    def test_required_keys_present(self):
        snap = LOADER.macro_snapshot()
        for key in ["tsy_2y", "tsy_10y", "sofr", "effr", "core_pce_yoy", "unemployment"]:
            assert key in snap, f"Missing key: {key}"

    def test_all_values_not_none(self):
        snap = LOADER.macro_snapshot()
        for key, val in snap.items():
            assert val is not None, f"Key {key} is None"

    def test_yields_positive(self):
        snap = LOADER.macro_snapshot()
        for key in ["tsy_2y", "tsy_10y", "sofr"]:
            assert snap[key] > 0

    def test_unemployment_plausible(self):
        snap = LOADER.macro_snapshot()
        assert 1.0 < snap["unemployment"] < 20.0


# ── FREDLoader.full_rates_dataset ─────────────────────────────────────────────

class TestFullRatesDataset:

    def test_returns_dataframe(self):
        df = LOADER.full_rates_dataset(start="2023-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = LOADER.full_rates_dataset(start="2023-01-01")
        for col in ["tsy_2y", "tsy_10y", "sofr", "core_pce_yoy", "unemployment"]:
            assert col in df.columns

    def test_non_empty(self):
        df = LOADER.full_rates_dataset(start="2023-01-01")
        assert len(df) > 0

    def test_index_is_datetime(self):
        df = LOADER.full_rates_dataset(start="2023-01-01")
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_no_all_nan_rows(self):
        df = LOADER.full_rates_dataset(start="2023-01-01")
        assert not df.isnull().all(axis=1).any()


# ── fetch_series (expects failure without valid key) ──────────────────────────

class TestFetchSeriesRaisesWithoutKey:

    def test_raises_without_key(self):
        loader = FREDLoader(api_key=None)
        # If there happens to be a real key in the environment this test
        # would call FRED — guard by passing a known-bad key
        bad_loader = FREDLoader(api_key="bad_key_force_failure_xyz")
        with pytest.raises(Exception):
            bad_loader.fetch_series("DGS10")

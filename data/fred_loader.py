"""
fred_loader.py — FRED data access layer for the US rates / SOFR pricing engine.

Fallback hierarchy for every method:
  1. Live FRED API  (requires FRED_API_KEY)
  2. Cached local CSVs  (data/ directory)
  3. Synthetic data  (_make_synthetic_rates)
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load .env from project root before anything reads env vars.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FRED series IDs
# ---------------------------------------------------------------------------
_TREASURY_SERIES: dict[str, str] = {
    "tsy_2y": "DGS2",
    "tsy_3y": "DGS3",
    "tsy_5y": "DGS5",
    "tsy_7y": "DGS7",
    "tsy_10y": "DGS10",
    "tsy_30y": "DGS30",
}

# ---------------------------------------------------------------------------
# Module-level synthetic data helper
# ---------------------------------------------------------------------------

def _make_synthetic_rates(
    start: str = "2018-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic rates DataFrame using correlated random walks
    anchored around plausible post-2018 levels.  Used only as last-resort fallback.
    """
    end_dt = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    idx = pd.bdate_range(start=start, end=end_dt)
    n = len(idx)
    rng = np.random.default_rng(seed=42)

    # Approximate 2018-2024 average levels
    anchors: dict[str, float] = {
        "tsy_2y": 2.50,
        "tsy_3y": 2.60,
        "tsy_5y": 2.75,
        "tsy_7y": 2.90,
        "tsy_10y": 3.00,
        "tsy_30y": 3.25,
        "sofr": 2.30,
        "effr": 2.35,
        "unemployment": 4.20,
        "core_pce_yoy": 2.50,
    }
    vols: dict[str, float] = {
        "tsy_2y": 0.007, "tsy_3y": 0.007, "tsy_5y": 0.006,
        "tsy_7y": 0.006, "tsy_10y": 0.005, "tsy_30y": 0.005,
        "sofr": 0.004, "effr": 0.003,
        "unemployment": 0.002, "core_pce_yoy": 0.002,
    }
    mean_rev = 0.005  # mild mean-reversion keeps paths near anchor

    df = pd.DataFrame(index=idx)
    for col, level in anchors.items():
        shocks = rng.normal(0, vols[col], n)
        path = np.empty(n)
        path[0] = level
        for i in range(1, n):
            path[i] = path[i - 1] + mean_rev * (level - path[i - 1]) + shocks[i]
        df[col] = np.maximum(path, 0.05)  # floor at 5 bp

    return df


# ---------------------------------------------------------------------------
# FREDLoader
# ---------------------------------------------------------------------------

class FREDLoader:
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("FRED_API_KEY")
        self._cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parent
        self._fred = None  # lazy-initialised on first use

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_fred(self):
        """Return a cached fredapi.Fred instance, or None if unavailable."""
        if self._fred is not None:
            return self._fred
        if not self._api_key:
            return None
        try:
            from fredapi import Fred  # type: ignore
            self._fred = Fred(api_key=self._api_key)
            return self._fred
        except Exception as exc:
            logger.warning("fredapi init failed: %s", exc)
            return None

    def _cache_path(self, filename: str) -> Path:
        return self._cache_dir / filename

    def _read_cache(self, filename: str) -> Optional[pd.DataFrame]:
        p = self._cache_path(filename)
        if not p.exists():
            return None
        return pd.read_csv(p, index_col=0, parse_dates=True)

    def _fetch_raw(
        self,
        series_id: str,
        start: str,
        end: Optional[str],
    ) -> pd.Series:
        """Call FRED and return a clean Series.  Raises on any failure."""
        fred = self._get_fred()
        if fred is None:
            raise RuntimeError(f"No FRED API key; cannot fetch '{series_id}'.")
        kwargs: dict = {"observation_start": start}
        if end:
            kwargs["observation_end"] = end
        s = fred.get_series(series_id, **kwargs)
        s.index = pd.to_datetime(s.index)
        s.name = series_id
        return s.sort_index()

    @staticmethod
    def _trim(df: pd.DataFrame, start: str, end: Optional[str]) -> pd.DataFrame:
        df = df.loc[df.index >= pd.Timestamp(start)]
        if end:
            df = df.loc[df.index <= pd.Timestamp(end)]
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_series(
        self,
        series_id: str,
        start: str = "2018-01-01",
        end: Optional[str] = None,
    ) -> pd.Series:
        """
        Fetch one FRED series by ID.  Raises if the API key is absent or the
        request fails — callers wanting graceful fallback should use the higher-
        level methods instead.
        """
        return self._fetch_raw(series_id, start, end)

    # ------------------------------------------------------------------

    def treasury_yields(
        self,
        start: str = "2018-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Daily Treasury CMT yields in percent.
        Columns: tsy_2y, tsy_3y, tsy_5y, tsy_7y, tsy_10y, tsy_30y
        """
        cols = list(_TREASURY_SERIES.keys())

        # 1. Live FRED
        fred = self._get_fred()
        if fred is not None:
            try:
                frames = {c: self._fetch_raw(sid, start, end) for c, sid in _TREASURY_SERIES.items()}
                df = pd.DataFrame(frames)
                df.index = pd.to_datetime(df.index)
                return df.sort_index().dropna(how="all")
            except Exception as exc:
                logger.warning("treasury_yields: FRED failed (%s); trying cache.", exc)

        # 2. Cached treasury_yields.csv
        try:
            df = self._read_cache("treasury_yields.csv")
            if df is not None:
                missing = [c for c in cols if c not in df.columns]
                if missing:
                    raise ValueError(f"Cache missing columns: {missing}")
                df = self._trim(df[cols], start, end)
                logger.warning("treasury_yields: using cached treasury_yields.csv.")
                return df.sort_index().dropna(how="all")
        except Exception as exc:
            logger.warning("treasury_yields: cache failed (%s); using synthetic.", exc)

        # 3. Synthetic
        logger.warning("treasury_yields: returning synthetic data.")
        synth = _make_synthetic_rates(start=start, end=end)
        return synth[[c for c in cols if c in synth.columns]]

    # ------------------------------------------------------------------

    def sofr_history(
        self,
        start: str = "2018-04-01",
        end: Optional[str] = None,
    ) -> pd.Series:
        """SOFR overnight rate in percent.  FRED series SOFR (starts April 2018)."""
        # 1. Live FRED
        fred = self._get_fred()
        if fred is not None:
            try:
                return self._fetch_raw("SOFR", start, end).rename("sofr")
            except Exception as exc:
                logger.warning("sofr_history: FRED failed (%s); trying cache.", exc)

        # 2. sofr_fred_data.csv
        try:
            df = self._read_cache("sofr_fred_data.csv")
            if df is not None and "sofr" in df.columns:
                s = self._trim(df[["sofr"]], start, end)["sofr"].dropna()
                logger.warning("sofr_history: using cached sofr_fred_data.csv.")
                return s.rename("sofr").sort_index()
        except Exception as exc:
            logger.warning("sofr_history: cache failed (%s); using synthetic.", exc)

        # 3. Synthetic
        logger.warning("sofr_history: returning synthetic data.")
        synth = _make_synthetic_rates(start=start, end=end)
        return synth["sofr"].rename("sofr")

    # ------------------------------------------------------------------

    def macro_snapshot(self) -> dict:
        """
        Latest available values for key macro indicators, all in percent:
        effr, sofr, core_pce_yoy, unemployment, tsy_2y, tsy_5y, tsy_10y, tsy_30y
        """
        def _last(s: pd.Series) -> Optional[float]:
            s = s.dropna()
            return float(s.iloc[-1]) if len(s) else None

        snapshot: dict = {}

        # Treasury yields (own method handles fallback)
        try:
            tsy = self.treasury_yields()
            for col in ("tsy_2y", "tsy_5y", "tsy_10y", "tsy_30y"):
                if col in tsy.columns:
                    snapshot[col] = _last(tsy[col])
        except Exception as exc:
            logger.warning("macro_snapshot: treasury yields unavailable (%s).", exc)

        # SOFR (own method handles fallback)
        try:
            snapshot["sofr"] = _last(self.sofr_history())
        except Exception as exc:
            logger.warning("macro_snapshot: SOFR unavailable (%s).", exc)

        # EFFR, core_pce_yoy, unemployment — try FRED live first
        fred = self._get_fred()
        if fred is not None:
            for key, sid, transform in (
                ("effr", "FEDFUNDS", None),
                ("unemployment", "UNRATE", None),
                # PCEPILFE is a price index; compute YoY from monthly series
                ("core_pce_yoy", "PCEPILFE", lambda s: s.pct_change(12) * 100),
            ):
                if key in snapshot:
                    continue
                try:
                    s = self._fetch_raw(sid, start="2016-01-01", end=None)
                    if transform:
                        s = transform(s)
                    snapshot[key] = _last(s)
                except Exception as exc:
                    logger.warning("macro_snapshot: %s unavailable from FRED (%s).", sid, exc)

        # Fill remaining keys from sofr_fred_data.csv cache
        macro_missing = [k for k in ("effr", "core_pce_yoy", "unemployment") if k not in snapshot]
        if macro_missing:
            try:
                df = self._read_cache("sofr_fred_data.csv")
                if df is not None:
                    col_map = {"fed_funds": "effr"}
                    df = df.rename(columns=col_map)
                    for key in macro_missing:
                        if key in df.columns:
                            snapshot[key] = _last(df[key])
            except Exception as exc:
                logger.warning("macro_snapshot: cache read failed (%s).", exc)

        # Final fallback: synthetic values for any still-missing keys
        synth = _make_synthetic_rates()
        synth_map = {
            "effr": "effr", "core_pce_yoy": "core_pce_yoy",
            "unemployment": "unemployment", "sofr": "sofr",
            "tsy_2y": "tsy_2y", "tsy_5y": "tsy_5y",
            "tsy_10y": "tsy_10y", "tsy_30y": "tsy_30y",
        }
        for key, col in synth_map.items():
            if key not in snapshot and col in synth.columns:
                snapshot.setdefault(key, _last(synth[col]))

        return snapshot

    # ------------------------------------------------------------------

    def full_rates_dataset(
        self,
        start: str = "2018-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Merged daily DataFrame with columns:
        tsy_2y, tsy_3y, tsy_5y, tsy_7y, tsy_10y, tsy_30y,
        sofr, effr, core_pce_yoy, unemployment

        Monthly macro series (core_pce_yoy, unemployment) are forward-filled
        to daily frequency so every business day has a value.
        """
        OUTPUT_COLS = [
            "tsy_2y", "tsy_3y", "tsy_5y", "tsy_7y", "tsy_10y", "tsy_30y",
            "sofr", "effr", "core_pce_yoy", "unemployment",
        ]
        FFILL_COLS = ("core_pce_yoy", "unemployment")

        # 1. Live FRED
        fred = self._get_fred()
        if fred is not None:
            try:
                return self._full_from_fred(start, end, OUTPUT_COLS, FFILL_COLS)
            except Exception as exc:
                logger.warning("full_rates_dataset: FRED failed (%s); trying cache.", exc)

        # 2. sofr_fred_data.csv (already has most of what we need)
        try:
            df = self._read_cache("sofr_fred_data.csv")
            if df is not None:
                df = df.rename(columns={"fed_funds": "effr"})
                df = self._trim(df, start, end)
                missing = [c for c in OUTPUT_COLS if c not in df.columns]
                if missing:
                    logger.warning(
                        "full_rates_dataset: cache missing %s; patching with synthetic.", missing
                    )
                    synth = _make_synthetic_rates(start=start, end=end)
                    for col in missing:
                        if col in synth.columns:
                            df[col] = synth[col]
                for col in FFILL_COLS:
                    if col in df.columns:
                        df[col] = df[col].ffill()
                logger.warning("full_rates_dataset: using cached sofr_fred_data.csv.")
                return df[[c for c in OUTPUT_COLS if c in df.columns]].sort_index().dropna(how="all")
        except Exception as exc:
            logger.warning("full_rates_dataset: cache failed (%s); using synthetic.", exc)

        # 3. Pure synthetic
        logger.warning("full_rates_dataset: returning synthetic data.")
        synth = _make_synthetic_rates(start=start, end=end)
        return synth[[c for c in OUTPUT_COLS if c in synth.columns]]

    # ------------------------------------------------------------------
    # Private: live FRED assembly
    # ------------------------------------------------------------------

    def _full_from_fred(
        self,
        start: str,
        end: Optional[str],
        output_cols: list[str],
        ffill_cols: tuple[str, ...],
    ) -> pd.DataFrame:
        series_map: dict[str, str] = {
            "tsy_2y": "DGS2", "tsy_3y": "DGS3", "tsy_5y": "DGS5",
            "tsy_7y": "DGS7", "tsy_10y": "DGS10", "tsy_30y": "DGS30",
            "sofr": "SOFR",
            "effr": "FEDFUNDS",
            # Fetched as a price index; YoY computed below
            "core_pce_yoy": "PCEPILFE",
            "unemployment": "UNRATE",
        }
        frames: dict[str, pd.Series] = {}
        for col, sid in series_map.items():
            try:
                frames[col] = self._fetch_raw(sid, start=start, end=end)
            except Exception as exc:
                logger.warning("_full_from_fred: could not fetch %s (%s).", sid, exc)

        if not frames:
            raise RuntimeError("No series fetched from FRED — all requests failed.")

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Convert PCEPILFE level → YoY percent (monthly series, then reindex to daily)
        if "core_pce_yoy" in df.columns:
            monthly_lvl = df["core_pce_yoy"].dropna().resample("ME").last()
            yoy = monthly_lvl.pct_change(12) * 100
            df["core_pce_yoy"] = yoy.reindex(df.index).ffill()

        for col in ffill_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        return df[[c for c in output_cols if c in df.columns]].dropna(how="all")

"""
Principal Component Analysis of the US Treasury yield curve.

The first three principal components explain >99% of historical yield
curve variation and correspond to intuitive economic factors:

  PC1 (~90-95% var.) : Level    — parallel shift of all rates
  PC2 (~4-7%  var.)  : Slope    — 2s10s tilt (long end vs short end)
  PC3 (~1-2%  var.)  : Curvature — belly vs wings (butterfly)

Classes
-------
YieldCurvePCA  : Fit and transform; hedge ratios; reconstruction
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class PCAResult:
    """Fitted PCA model for the yield curve."""
    components:       np.ndarray    # shape (n_components, n_tenors) — loadings
    explained_var:    np.ndarray    # shape (n_components,) — fraction of variance
    mean_yields:      np.ndarray    # shape (n_tenors,) — mean yield at each tenor
    tenors:           np.ndarray    # shape (n_tenors,) — in years
    singular_values:  np.ndarray    # shape (n_components,)


class YieldCurvePCA:
    """
    Fit PCA on historical daily yield curve data.

    Parameters
    ----------
    n_components : number of PCs to retain (default 3)
    """

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self._result: PCAResult | None = None

    def fit(
        self,
        yield_df: pd.DataFrame,
        tenor_cols: Sequence[str] | None = None,
        tenors_yrs: Sequence[float] | None = None,
    ) -> "YieldCurvePCA":
        """
        Fit PCA on a DataFrame of yields.

        Parameters
        ----------
        yield_df   : DataFrame where each column is a yield series (in %)
                     Rows = dates, Columns = tenor series
        tenor_cols : list of column names to use (default: all columns)
        tenors_yrs : maturity in years for each column
                     (inferred from column names if None, e.g. 'tsy_2y' → 2.0)
        """
        cols = tenor_cols if tenor_cols is not None else list(yield_df.columns)
        X    = yield_df[cols].dropna().values.astype(float)

        if tenors_yrs is not None:
            tenors = np.asarray(tenors_yrs, dtype=float)
        else:
            tenors = _parse_tenors(cols)

        # Demean
        mu   = X.mean(axis=0)
        Xc   = X - mu

        # SVD (numerically stable)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        total_var = (s ** 2).sum()
        n_keep    = min(self.n_components, len(s))

        self._result = PCAResult(
            components      = Vt[:n_keep],          # (n_comp, n_tenors)
            explained_var   = (s[:n_keep] ** 2) / total_var,
            mean_yields     = mu,
            tenors          = tenors,
            singular_values = s[:n_keep],
        )
        self._cols = cols
        return self

    @property
    def result(self) -> PCAResult:
        if self._result is None:
            raise RuntimeError("Call .fit() first")
        return self._result

    # ── Explained variance ────────────────────────────────────────────────────

    def explained_variance_table(self) -> pd.DataFrame:
        """Return a table of variance explained by each PC."""
        r   = self.result
        pct = r.explained_var * 100
        return pd.DataFrame({
            "PC":               [f"PC{i+1}" for i in range(len(pct))],
            "explained_pct":    pct.round(3),
            "cumulative_pct":   np.cumsum(pct).round(3),
            "singular_value":   r.singular_values.round(4),
        }).set_index("PC")

    # ── Transform: yields → PC scores ────────────────────────────────────────

    def transform(self, yield_df: pd.DataFrame) -> pd.DataFrame:
        """
        Project yield curve observations into PC score space.

        Returns DataFrame with columns PC1, PC2, ..., PCn.
        """
        r   = self.result
        X   = yield_df[self._cols].dropna().values - r.mean_yields
        scores = X @ r.components.T   # (n_obs, n_comp)
        cols   = [f"PC{i+1}" for i in range(scores.shape[1])]
        return pd.DataFrame(scores, index=yield_df[self._cols].dropna().index, columns=cols)

    # ── Reconstruct: PC scores → yields ──────────────────────────────────────

    def reconstruct(
        self, scores: np.ndarray | pd.DataFrame, n_components: int | None = None
    ) -> np.ndarray:
        """
        Reconstruct yield curve from PC scores.

        Parameters
        ----------
        scores       : array of shape (..., n_components) or DataFrame with PC columns
        n_components : use only first n_components (default: all fitted)
        """
        r   = self.result
        k   = n_components or self.n_components
        if isinstance(scores, pd.DataFrame):
            s = scores[[f"PC{i+1}" for i in range(k)]].values
        else:
            s = np.asarray(scores)[..., :k]
        return s @ r.components[:k] + r.mean_yields

    # ── Loadings ──────────────────────────────────────────────────────────────

    def loadings_df(self) -> pd.DataFrame:
        """Return the PC loadings (eigenvectors) as a DataFrame."""
        r    = self.result
        cols = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(
            r.components.T,
            index=[f"{t:.2f}Y" for t in r.tenors],
            columns=cols,
        )

    # ── Hedge ratios ──────────────────────────────────────────────────────────

    def hedge_ratios(
        self,
        position_dv01: dict[float, float],
        hedge_pcs: Sequence[int] = (1, 2, 3),
    ) -> pd.DataFrame:
        """
        Compute hedge ratios (in DV01 units) to neutralise exposure to
        the specified principal components.

        Parameters
        ----------
        position_dv01 : {tenor_years: dv01_usd} — existing position risk
                        e.g. {10.0: -10000}  = short 10Y $10k DV01
        hedge_pcs     : which PCs to hedge (default: all 3)

        Returns
        -------
        DataFrame showing the hedge DV01 needed at each tenor pillar
        to zero-out the PC exposures.
        """
        r        = self.result
        k        = len(hedge_pcs)
        tenors_t = r.tenors

        # Position DV01 vector across tenors (fill zeros for tenors not in position)
        dv01_vec = np.zeros(len(tenors_t))
        for tenor, dv01 in position_dv01.items():
            idx = np.argmin(np.abs(tenors_t - tenor))
            dv01_vec[idx] += dv01

        # PC exposure of position: e_i = Σ_j L_{ij} × DV01_j
        pc_exposure = r.components[np.array(hedge_pcs) - 1] @ dv01_vec  # (k,)

        # Build hedge: for each PC, choose a liquid instrument to offset
        # We use key-rate tenors corresponding to each PC's peak loading
        hedge_tenors_idx = []
        for i in range(k):
            loading = r.components[hedge_pcs[i] - 1]
            hedge_tenors_idx.append(int(np.argmax(np.abs(loading))))

        # Solve: L[:, hedge_tenors] @ h = -e  for h (hedge DV01 at each hedge tenor)
        L_hedge = r.components[np.array(hedge_pcs) - 1][:, hedge_tenors_idx]  # (k, k)
        try:
            h = np.linalg.solve(L_hedge, -pc_exposure)
        except np.linalg.LinAlgError:
            h = np.linalg.lstsq(L_hedge, -pc_exposure, rcond=None)[0]

        rows = []
        for i, (pc, idx, dv01) in enumerate(zip(hedge_pcs, hedge_tenors_idx, h)):
            rows.append({
                "PC":             f"PC{pc}",
                "hedge_tenor_y":  float(tenors_t[idx]),
                "hedge_dv01_usd": float(dv01),
                "pc_exposure":    float(pc_exposure[i]),
            })
        return pd.DataFrame(rows).set_index("PC")

    # ── Rolling PC scores ─────────────────────────────────────────────────────

    def rolling_scores(
        self,
        yield_df: pd.DataFrame,
        window: int = 252,
    ) -> pd.DataFrame:
        """
        Compute PC scores using a rolling re-fitted PCA window.
        Useful for out-of-sample / regime-stable analysis.

        Returns DataFrame with PC1, PC2, PC3 scores and date index.
        """
        cols  = self._cols
        df    = yield_df[cols].dropna()
        rows  = []
        for i in range(window, len(df) + 1):
            win = df.iloc[i - window: i]
            pca = YieldCurvePCA(self.n_components).fit(win, tenor_cols=cols,
                                                        tenors_yrs=list(self.result.tenors))
            obs    = df.iloc[i - 1].values - pca.result.mean_yields
            scores = obs @ pca.result.components.T
            row    = {f"PC{j+1}": float(scores[j]) for j in range(self.n_components)}
            row["date"] = df.index[i - 1]
            rows.append(row)
        return pd.DataFrame(rows).set_index("date")


# ── PC regime classifier ──────────────────────────────────────────────────────

def classify_pc_regime(
    scores_df: pd.DataFrame,
    pc1_threshold: float = 0.50,   # bps
    pc2_threshold: float = 0.25,
) -> pd.Series:
    """
    Classify each observation into a PC-based regime.

    Rules
    -----
    PC1 (level) > threshold  → high_rates
    PC1 (level) < -threshold → low_rates
    PC2 (slope) > threshold  → steep
    PC2 (slope) < -threshold → flat/inverted
    """
    regime = pd.Series("neutral", index=scores_df.index, name="pc_regime")
    p1 = scores_df["PC1"]
    p2 = scores_df["PC2"] if "PC2" in scores_df.columns else pd.Series(0, index=scores_df.index)
    regime[p1 >  pc1_threshold] = "high_rates"
    regime[p1 < -pc1_threshold] = "low_rates"
    regime[(p2 < -pc2_threshold) & (regime == "neutral")] = "flat_curve"
    regime[(p2 >  pc2_threshold) & (regime == "neutral")] = "steep_curve"
    return regime


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_tenors(col_names: list[str]) -> np.ndarray:
    """Infer tenor in years from column names like 'tsy_2y', 'tsy_30y', '1m', '3m'."""
    tenors = []
    for c in col_names:
        c_lower = c.lower()
        # Handle month suffixes: _1m, _3m, _6m
        import re
        m = re.search(r'(\d+(?:\.\d+)?)\s*m(?:on|o)?', c_lower)
        if m:
            tenors.append(float(m.group(1)) / 12.0)
            continue
        m = re.search(r'(\d+(?:\.\d+)?)\s*y(?:r|ear)?', c_lower)
        if m:
            tenors.append(float(m.group(1)))
            continue
        tenors.append(np.nan)
    return np.array(tenors)

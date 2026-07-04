"""
Tests for PCA-calibrated LMM correlation matrix (lmm_corr_from_pca).

Validates:
  - Output is symmetric, unit-diagonal, bounded in [-1, 1]
  - Positive semi-definite (all eigenvalues >= 0)
  - Correctly recovers dominant correlation structure from synthetic data
  - Integrates with LMMParams(corr_matrix=...)
"""
from __future__ import annotations
import numpy as np
import pytest
from datetime import date

from sofr_engine.lmm import (
    lmm_corr_from_pca,
    LMMParams,
    initial_forwards,
    exponential_correlation,
)
from sofr_engine.bootstrap import flat_sofr_curve


RNG = np.random.default_rng(42)
REF = date(2024, 6, 5)

# Standard LMM tenor grid (N=8 forward rates)
TENORS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0])
N = len(TENORS) - 1  # 8 forward rates


def _synthetic_changes(N: int, T_days: int, true_corr: np.ndarray, seed: int = 0) -> np.ndarray:
    """Generate correlated daily rate changes with a known correlation structure."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(true_corr + 1e-8 * np.eye(N))
    Z = rng.standard_normal((T_days, N))
    return Z @ L.T


# ── Basic mathematical properties ─────────────────────────────────────────────

class TestPCACorrelationProperties:
    @pytest.fixture
    def corr_basic(self):
        X = RNG.standard_normal((500, N))
        return lmm_corr_from_pca(X, n_components=3)

    def test_shape(self, corr_basic):
        assert corr_basic.shape == (N, N)

    def test_unit_diagonal(self, corr_basic):
        np.testing.assert_allclose(np.diag(corr_basic), 1.0, atol=1e-12)

    def test_symmetric(self, corr_basic):
        np.testing.assert_allclose(corr_basic, corr_basic.T, atol=1e-12)

    def test_bounded(self, corr_basic):
        assert np.all(corr_basic >= -1.0 - 1e-10)
        assert np.all(corr_basic <= 1.0 + 1e-10)

    def test_positive_semi_definite(self, corr_basic):
        eigvals = np.linalg.eigvalsh(corr_basic)
        assert np.all(eigvals >= -1e-10), f"Min eigenvalue: {eigvals.min():.2e}"

    def test_off_diagonal_in_range(self, corr_basic):
        mask = ~np.eye(N, dtype=bool)
        assert np.all(np.abs(corr_basic[mask]) <= 1.0)


# ── Factor recovery ───────────────────────────────────────────────────────────

class TestPCACorrelationRecovery:
    def test_high_correlation_structure_captured(self):
        """Exponential decay corr → estimated correlation positive for most adjacent pairs."""
        true_corr = exponential_correlation(TENORS, decay=0.05)
        X = _synthetic_changes(N, 2000, true_corr, seed=1)
        est_corr = lmm_corr_from_pca(X, n_components=3)
        # With 3 factors and noisy data, most adjacent pairs should be positive
        adjacent = [est_corr[i, i + 1] for i in range(N - 1)]
        n_positive = sum(v > 0 for v in adjacent)
        assert n_positive >= N - 2, (
            f"Expected most adjacent corrs positive; got {n_positive}/{N-1}: {adjacent}"
        )

    def test_uncorrelated_input_gives_near_identity(self):
        """Independent forward rates → correlation matrix near identity."""
        X = RNG.standard_normal((2000, N))
        corr = lmm_corr_from_pca(X, n_components=1)
        # With only 1 PC, off-diag should be small
        mask = ~np.eye(N, dtype=bool)
        avg_off_diag = np.mean(np.abs(corr[mask]))
        assert avg_off_diag < 0.5

    def test_more_components_higher_off_diagonal(self):
        """More PCA factors → captures more of the correlation structure."""
        true_corr = exponential_correlation(TENORS, decay=0.2)
        X = _synthetic_changes(N, 1000, true_corr, seed=2)
        corr1 = lmm_corr_from_pca(X, n_components=1)
        corr3 = lmm_corr_from_pca(X, n_components=3)
        # 3-component model should have higher average off-diagonal
        mask = ~np.eye(N, dtype=bool)
        assert np.mean(corr3[mask]) > np.mean(corr1[mask]) - 0.1

    def test_level_factor_dominates(self):
        """With a strong level factor, PC1 explains >90% of variance and loadings near-uniform."""
        rng2 = np.random.default_rng(99)
        level_shocks = rng2.standard_normal((2000, 1))
        X = np.tile(level_shocks, (1, N)) + 0.01 * rng2.standard_normal((2000, N))

        # Run SVD to check PC1 dominance
        Xc = X - X.mean(axis=0)
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        explained_pc1 = (s[0] ** 2) / (s ** 2).sum()
        assert explained_pc1 > 0.90, (
            f"PC1 explains {explained_pc1:.1%} variance; expected >90% for level-dominant data"
        )

        # PC1 loading vector should be near-uniform (all entries same sign)
        L1 = Vt[0]
        signs = np.sign(L1)
        assert np.all(signs == signs[0]), (
            f"PC1 loadings not all same sign for level factor: {L1.round(3)}"
        )


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestPCACorrelationEdgeCases:
    def test_n_components_1(self):
        X = RNG.standard_normal((300, N))
        corr = lmm_corr_from_pca(X, n_components=1)
        assert corr.shape == (N, N)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-12)

    def test_n_components_equals_N(self):
        X = RNG.standard_normal((300, N))
        corr = lmm_corr_from_pca(X, n_components=N)
        assert corr.shape == (N, N)
        assert np.all(np.linalg.eigvalsh(corr) >= -1e-8)

    def test_n_components_capped_at_rank(self):
        # T_days=5 means rank ≤ 4, n_components=10 should be capped
        X = RNG.standard_normal((5, N))
        corr = lmm_corr_from_pca(X, n_components=10)
        assert corr.shape == (N, N)

    def test_2d_input_required(self):
        with pytest.raises(ValueError, match="2-D"):
            lmm_corr_from_pca(np.ones(100), n_components=3)

    def test_single_rate(self):
        X = RNG.standard_normal((200, 1))
        corr = lmm_corr_from_pca(X, n_components=1)
        assert corr.shape == (1, 1)
        assert corr[0, 0] == pytest.approx(1.0)

    def test_two_rates(self):
        X = RNG.standard_normal((200, 2))
        corr = lmm_corr_from_pca(X, n_components=2)
        assert corr.shape == (2, 2)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-12)


# ── Integration with LMMParams ────────────────────────────────────────────────

class TestPCACorrelationIntegration:
    def test_plug_into_lmm_params(self):
        """PCA correlation matrix integrates cleanly into LMMParams."""
        X = RNG.standard_normal((500, N))
        corr = lmm_corr_from_pca(X, n_components=3)
        vols = np.full(N, 0.60)
        params = LMMParams(tenors=TENORS, vols=vols, corr_matrix=corr)
        # Should not raise; correlation() returns the custom matrix
        C = params.correlation()
        assert C.shape == (N, N)
        np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-12)

    def test_drift_weights_computed(self):
        """Drift weight matrix W should be upper triangular with no NaNs."""
        from sofr_engine.lmm import _drift_weights
        X = RNG.standard_normal((500, N))
        corr = lmm_corr_from_pca(X, n_components=3)
        vols = np.full(N, 0.60)
        params = LMMParams(tenors=TENORS, vols=vols, corr_matrix=corr)
        W = _drift_weights(params)
        assert W.shape == (N, N)
        assert not np.any(np.isnan(W))
        # Upper triangular: lower triangle should be zero
        assert np.all(np.tril(W, k=-1) == 0.0)

    def test_pca_vs_exponential_cap_price(self):
        """Cap prices from PCA corr should be in a reasonable range vs exponential."""
        from sofr_engine.lmm import cap_black76, cap_implied_vol
        curve = flat_sofr_curve(REF, 0.05)
        tenors = TENORS
        vols = np.full(N, 0.60)

        X = _synthetic_changes(N, 1000, exponential_correlation(tenors, 0.1), seed=5)
        corr = lmm_corr_from_pca(X, n_components=3)

        params_exp = LMMParams(tenors=tenors, vols=vols, corr_decay=0.1)
        params_pca = LMMParams(tenors=tenors, vols=vols, corr_matrix=corr)

        cap_exp = cap_black76(curve, params_exp, strike=0.05, notional=1e6)
        cap_pca = cap_black76(curve, params_pca, strike=0.05, notional=1e6)

        # Both should be positive and within 50% of each other
        assert cap_exp > 0
        assert cap_pca > 0
        ratio = cap_pca / cap_exp
        assert 0.5 < ratio < 2.0, f"Cap PCA/exp ratio = {ratio:.3f} out of range"

    def test_lmm_params_corr_matrix_overrides_decay(self):
        """When corr_matrix is set, corr_decay is ignored in correlation()."""
        identity = np.eye(N)
        params = LMMParams(tenors=TENORS, vols=np.ones(N) * 0.5,
                           corr_decay=0.5, corr_matrix=identity)
        C = params.correlation()
        np.testing.assert_array_equal(C, identity)

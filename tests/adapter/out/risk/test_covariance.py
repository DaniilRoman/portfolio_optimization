"""Tests for the Ledoit-Wolf covariance estimator (§4.1)."""
import numpy as np
import pandas as pd
import pytest

from src.adapter.out.risk.covariance import LedoitWolfCovarianceModel, compute_covariance


def _make_prices(
    n: int = 300,
    n_assets: int = 4,
    seed: int = 42,
    corr: float | None = None,
) -> pd.DataFrame:
    """Generate synthetic price data. If corr is set, assets 0 and 1 have that correlation."""
    rng = np.random.default_rng(seed)
    tickers = [f"A{i}" for i in range(n_assets)]

    if corr is not None and n_assets >= 2:
        base = rng.standard_normal(n)
        noise = rng.standard_normal((n, n_assets)) * 0.01
        returns = noise.copy()
        returns[:, 0] += base * 0.02
        returns[:, 1] += base * corr * 0.02 - base * (1 - corr) * 0.02 if corr >= 0 else -base * abs(corr) * 0.02
        for i in range(2, n_assets):
            returns[:, i] += rng.standard_normal(n) * 0.01
    else:
        returns = rng.normal(0.0003, 0.01, size=(n, n_assets))

    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame(prices, index=dates, columns=tickers)


class TestComputeCovariance:
    def test_output_shape(self):
        prices = _make_prices(n_assets=4)
        sigma = compute_covariance(prices)
        assert sigma.shape == (4, 4)

    def test_symmetric(self):
        sigma = compute_covariance(_make_prices())
        assert np.allclose(sigma, sigma.T, atol=1e-12)

    def test_positive_semi_definite(self):
        sigma = compute_covariance(_make_prices())
        eigenvalues = np.linalg.eigvalsh(sigma)
        assert np.all(eigenvalues >= -1e-9), f"Non-PSD: min eigenvalue = {eigenvalues.min():.2e}"

    def test_diagonal_entries_positive(self):
        sigma = compute_covariance(_make_prices())
        assert np.all(np.diag(sigma) > 0)

    def test_annualised_scale(self):
        # Diagonal entries should be on the order of (daily_vol * sqrt(252))^2 ~ annualised variance
        # For 1% daily vol: annualised var ≈ (0.01)^2 * 252 ≈ 0.0252
        prices = _make_prices()
        sigma = compute_covariance(prices)
        for var in np.diag(sigma):
            assert 1e-4 < var < 10.0, f"Annualised variance {var:.4f} is outside a sane range"

    def test_raises_on_single_asset(self):
        prices = _make_prices(n_assets=1)
        with pytest.raises(ValueError, match="2 assets"):
            compute_covariance(prices)

    def test_raises_on_single_observation(self):
        prices = _make_prices(n=1)
        with pytest.raises(ValueError, match="2 price observations"):
            compute_covariance(prices)

    def test_negatively_correlated_pair_has_negative_off_diagonal(self):
        # Build two perfectly negatively correlated price series.
        rng = np.random.default_rng(0)
        n = 500
        dates = pd.bdate_range("2020-01-02", periods=n)
        base = rng.standard_normal(n) * 0.015
        prices = pd.DataFrame(
            {
                "A": 100.0 * np.exp(np.cumsum(base + rng.standard_normal(n) * 0.001)),
                "B": 100.0 * np.exp(np.cumsum(-base + rng.standard_normal(n) * 0.001)),
            },
            index=dates,
        )
        sigma = compute_covariance(prices)
        assert sigma[0, 1] < 0, "Anti-correlated pair should have negative off-diagonal covariance"

    def test_quad_form_anti_correlated_lower_than_diagonal(self):
        """Adding a perfectly anti-correlated asset reduces portfolio variance vs diagonal proxy."""
        rng = np.random.default_rng(7)
        n = 500
        dates = pd.bdate_range("2020-01-02", periods=n)
        base = rng.standard_normal(n) * 0.02

        # Two assets: A and B with near-perfect negative correlation
        prices = pd.DataFrame(
            {
                "A": 100.0 * np.exp(np.cumsum(base + rng.standard_normal(n) * 0.001)),
                "B": 100.0 * np.exp(np.cumsum(-base + rng.standard_normal(n) * 0.001)),
            },
            index=dates,
        )
        sigma = compute_covariance(prices)

        # Equal-weight portfolio w = [0.5, 0.5]
        w = np.array([0.5, 0.5])

        # True portfolio variance using Ledoit-Wolf Σ
        var_lw = float(w @ sigma @ w)

        # Diagonal approximation: ignores off-diagonal correlations
        var_diag = float(w @ np.diag(np.diag(sigma)) @ w)

        # With negative correlation the LW variance must be strictly below the diagonal approximation
        assert var_lw < var_diag, (
            f"LW variance {var_lw:.6f} should be less than diagonal proxy {var_diag:.6f} "
            "for a negatively-correlated pair"
        )


class TestLedoitWolfCovarianceModel:
    def test_implements_risk_model_protocol(self):
        from src.domain.ports.risk_model import RiskModel
        model = LedoitWolfCovarianceModel()
        # Structural check: must have a .cov method matching the protocol
        assert hasattr(model, "cov")
        assert callable(model.cov)

    def test_cov_returns_ndarray(self):
        model = LedoitWolfCovarianceModel()
        prices = _make_prices(n_assets=3)
        result = model.cov(prices)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)

    def test_cov_matches_compute_covariance(self):
        prices = _make_prices(n_assets=3, seed=99)
        model = LedoitWolfCovarianceModel()
        assert np.allclose(model.cov(prices), compute_covariance(prices))

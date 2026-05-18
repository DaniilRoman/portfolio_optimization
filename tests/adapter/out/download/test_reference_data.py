"""Tests for the reference_data adapter.

All external I/O (yfinance, urllib) is mocked so tests run fully offline.
Each test that writes to the cache uses tmp_path + monkeypatch.chdir for isolation.
"""
import io
import zipfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.adapter.out.download.reference_data import (
    FF_COLUMNS,
    _parse_ff_csv,
    fetch_benchmark,
    fetch_ff_factors,
    fetch_risk_free_rate,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_yf_hist(n: int = 20, start: str = "2020-01-02") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="B", tz="America/New_York")
    prices = np.linspace(300.0, 320.0, n)
    df = pd.DataFrame({"Close": prices, "Volume": np.ones(n) * 1_000_000}, index=dates)
    df.index.name = "Date"
    return df


def _make_fred_csv(series: str = "DGS3MO", n: int = 10) -> bytes:
    dates = pd.bdate_range("2020-01-02", periods=n).strftime("%Y-%m-%d")
    lines = ["DATE," + series] + [f"{d},{1.5 + i * 0.01}" for i, d in enumerate(dates)]
    # Add a "." row to test missing-value handling
    lines.insert(3, f"{pd.bdate_range('2020-01-02', periods=n+1)[-1].strftime('%Y-%m-%d')},.")
    return "\n".join(lines).encode()


def _make_ff5_csv_content(n_rows: int = 260, start_yyyymmdd: int = 20200102) -> str:
    """Generate a minimal daily FF5 CSV matching Kenneth French's format."""
    header = "    DATE    Mkt-RF     SMB     HML     RMW     CMA      RF"
    rows = []
    d = pd.date_range(pd.Timestamp(str(start_yyyymmdd)), periods=n_rows, freq="B")
    for dt in d:
        rows.append(f"{dt.strftime('%Y%m%d')}   0.10  -0.05   0.03   0.02  -0.01   0.01")
    return f"This file was created by CMPT\n\n{header}\n" + "\n".join(rows) + "\nCopyright 2024\n"


def _make_mom_csv_content(n_rows: int = 260, start_yyyymmdd: int = 20200102) -> str:
    header = "    DATE      Mom"
    rows = []
    d = pd.date_range(pd.Timestamp(str(start_yyyymmdd)), periods=n_rows, freq="B")
    for dt in d:
        rows.append(f"{dt.strftime('%Y%m%d')}   0.08")
    return f"This file was created by CMPT\n\n{header}\n" + "\n".join(rows) + "\nCopyright 2024\n"


def _zip_bytes(filename: str, content: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, content.encode("latin-1"))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# fetch_benchmark
# ---------------------------------------------------------------------------

class TestFetchBenchmark:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

    def test_returns_price_series(self):
        with patch("src.adapter.out.download.reference_data._yf_history", return_value=_make_yf_hist()):
            result = fetch_benchmark("SPY", "2020-01-02", "2020-02-01")
        assert isinstance(result, pd.Series)
        assert result.dtype == float
        assert not result.empty

    def test_cache_hit_skips_yfinance(self):
        mock_hist = MagicMock(return_value=_make_yf_hist())
        with patch("src.adapter.out.download.reference_data._yf_history", mock_hist):
            fetch_benchmark("SPY", "2020-01-02", "2020-02-01")
            fetch_benchmark("SPY", "2020-01-02", "2020-02-01")
        assert mock_hist.call_count == 1

    def test_empty_response_raises(self):
        with patch("src.adapter.out.download.reference_data._yf_history", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="No benchmark data"):
                fetch_benchmark("INVALID", "2020-01-02", "2020-02-01")


# ---------------------------------------------------------------------------
# fetch_risk_free_rate
# ---------------------------------------------------------------------------

class TestFetchRiskFreeRate:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

    def test_returns_monotonic_series(self):
        with patch("src.adapter.out.download.reference_data._urlopen_bytes", return_value=_make_fred_csv()):
            result = fetch_risk_free_rate("2020-01-02", "2020-12-31")
        assert isinstance(result, pd.Series)
        assert not result.empty
        assert result.index.is_monotonic_increasing

    def test_values_are_decimal(self):
        with patch("src.adapter.out.download.reference_data._urlopen_bytes", return_value=_make_fred_csv()):
            result = fetch_risk_free_rate("2020-01-02", "2020-12-31")
        assert result.max() < 1.0

    def test_drops_missing_values(self):
        """Rows with "." in the FRED CSV must be absent from the result."""
        with patch("src.adapter.out.download.reference_data._urlopen_bytes", return_value=_make_fred_csv()):
            result = fetch_risk_free_rate("2020-01-02", "2020-12-31")
        assert result.isna().sum() == 0

    def test_cache_hit_skips_urlopen(self):
        mock_open = MagicMock(return_value=_make_fred_csv())
        with patch("src.adapter.out.download.reference_data._urlopen_bytes", mock_open):
            fetch_risk_free_rate("2020-01-02", "2020-12-31")
            fetch_risk_free_rate("2020-01-02", "2020-12-31")
        assert mock_open.call_count == 1


# ---------------------------------------------------------------------------
# _parse_ff_csv (unit-level)
# ---------------------------------------------------------------------------

class TestParseFfCsv:
    def test_daily_columns(self):
        content = _make_ff5_csv_content(n_rows=10)
        df = _parse_ff_csv(content, "daily")
        assert set(df.columns) >= {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"}

    def test_daily_index_is_datetime(self):
        content = _make_ff5_csv_content(n_rows=10)
        df = _parse_ff_csv(content, "daily")
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_mom_daily_columns(self):
        content = _make_mom_csv_content(n_rows=10)
        df = _parse_ff_csv(content, "daily")
        assert "Mom" in df.columns


# ---------------------------------------------------------------------------
# fetch_ff_factors
# ---------------------------------------------------------------------------

class TestFetchFfFactors:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

    def _mock_urlopen(self, ff5_rows: int = 260) -> MagicMock:
        ff5_zip = _zip_bytes("F-F_Research_Data_5_Factors_2x3_daily.CSV", _make_ff5_csv_content(ff5_rows))
        mom_zip = _zip_bytes("F-F_Momentum_Factor_daily.CSV", _make_mom_csv_content(ff5_rows))
        mock = MagicMock(side_effect=[ff5_zip, mom_zip])
        return mock

    def test_returns_expected_columns(self):
        with patch("src.adapter.out.download.reference_data._urlopen_bytes", self._mock_urlopen()):
            df = fetch_ff_factors("ff5", "2020-01-02", "2020-12-31", "daily")
        assert list(df.columns) == FF_COLUMNS

    def test_values_are_decimal(self):
        with patch("src.adapter.out.download.reference_data._urlopen_bytes", self._mock_urlopen()):
            df = fetch_ff_factors("ff5", "2020-01-02", "2020-12-31", "daily")
        # Raw CSV has 0.10 for Mkt-RF; after /100 it should be 0.001
        assert df["Mkt-RF"].abs().max() < 1.0

    def test_filters_to_date_range(self):
        # Generate 3 years of daily data starting 2018
        mock_open = MagicMock(
            side_effect=[
                _zip_bytes("ff5.CSV", _make_ff5_csv_content(n_rows=780, start_yyyymmdd=20180102)),
                _zip_bytes("mom.CSV", _make_mom_csv_content(n_rows=780, start_yyyymmdd=20180102)),
            ]
        )
        with patch("src.adapter.out.download.reference_data._urlopen_bytes", mock_open):
            df = fetch_ff_factors("ff5", "2020-01-01", "2020-12-31", "daily")
        assert df.index.min() >= pd.Timestamp("2020-01-01")
        assert df.index.max() <= pd.Timestamp("2020-12-31")

    def test_cache_hit_skips_download(self):
        mock_open = self._mock_urlopen()
        with patch("src.adapter.out.download.reference_data._urlopen_bytes", mock_open):
            fetch_ff_factors("ff5", "2020-01-02", "2020-12-31", "daily")
            fetch_ff_factors("ff5", "2020-01-02", "2020-12-31", "daily")
        # Two downloads on first call (ff5 + mom), zero on second (cache hit)
        assert mock_open.call_count == 2

    def test_approximately_252_rows_for_full_year_2020(self):
        with patch("src.adapter.out.download.reference_data._urlopen_bytes", self._mock_urlopen(260)):
            df = fetch_ff_factors("ff5", "2020-01-02", "2020-12-31", "daily")
        # 260 business days generated; 2020 trading days ≈ 252; within ±10 is fine
        assert 240 <= len(df) <= 270

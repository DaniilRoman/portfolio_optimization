"""Tests for config.universe.resolve_universe."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from config.universe import _to_date, resolve_universe  # noqa: F401

# ---------------------------------------------------------------------------
# Test 1 (acceptance criteria): as_of="2010-01-01" includes ≥10 tickers
# no longer in the current sp500 index.
# ---------------------------------------------------------------------------


def test_sp500_historical_has_at_least_10_delisted_tickers():
    """resolve_universe('sp500', as_of='2010-01-01') must include ≥10 former members."""
    historical = resolve_universe("sp500", as_of="2010-01-01")
    current = set(resolve_universe("sp500"))
    delisted_in_historical = [t for t in historical if t not in current]
    assert len(delisted_in_historical) >= 10, (
        f"Expected ≥10 former S&P 500 members in 2010-01-01 universe, got {len(delisted_in_historical)}: "
        f"{delisted_in_historical}"
    )


# ---------------------------------------------------------------------------
# Test 2: current universe (as_of=None) excludes all delisted tickers.
# ---------------------------------------------------------------------------


def test_current_universe_excludes_delisted():
    current = resolve_universe("sp500")
    # All 15 historical tickers added to sp500.csv must be absent
    delisted_tickers = ["YHOO", "SHLD", "EK", "S", "GENZ", "Q", "MFE", "NVLS", "GR", "DELL"]
    for t in delisted_tickers:
        assert t not in current, f"Delisted ticker {t!r} should not appear in current sp500 universe"


# ---------------------------------------------------------------------------
# Test 3: tickers listed after as_of are excluded.
# ---------------------------------------------------------------------------


def test_future_listings_excluded(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_universe.csv"
    csv_path.write_text("ticker,listed_at,delisted_at\nOLD,,\nNEW,2025-01-01,\n")
    with patch("config.universe._HISTORY_DIR", tmp_path):
        result = resolve_universe("test_universe", as_of="2020-01-01")
    assert "OLD" in result
    assert "NEW" not in result


# ---------------------------------------------------------------------------
# Test 4: tickers delisted before as_of are excluded.
# ---------------------------------------------------------------------------


def test_delisted_before_as_of_excluded(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_universe.csv"
    csv_path.write_text("ticker,listed_at,delisted_at\nACTIVE,,\nGONE,,2019-06-01\n")
    with patch("config.universe._HISTORY_DIR", tmp_path):
        result = resolve_universe("test_universe", as_of="2020-01-01")
    assert "ACTIVE" in result
    assert "GONE" not in result


# ---------------------------------------------------------------------------
# Test 5: tickers delisted on exactly as_of are excluded (delisted_at is exclusive).
# ---------------------------------------------------------------------------


def test_delisted_on_as_of_date_excluded(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_universe.csv"
    csv_path.write_text("ticker,listed_at,delisted_at\nXYZ,,2020-01-01\n")
    with patch("config.universe._HISTORY_DIR", tmp_path):
        result = resolve_universe("test_universe", as_of="2020-01-01")
    assert "XYZ" not in result


# ---------------------------------------------------------------------------
# Test 6: missing universe file raises FileNotFoundError.
# ---------------------------------------------------------------------------


def test_missing_universe_raises(tmp_path: Path) -> None:
    with patch("config.universe._HISTORY_DIR", tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_universe("nonexistent")


# ---------------------------------------------------------------------------
# Test 7: all registered universes can be resolved without error.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["sp500", "etf", "de_etf", "ishares", "vanguard"])
def test_all_universes_resolvable(name: str) -> None:
    result = resolve_universe(name)
    assert len(result) > 0, f"Universe {name!r} resolved to an empty list"


# ---------------------------------------------------------------------------
# Test 8: _to_date helper handles edge cases.
# ---------------------------------------------------------------------------


def test_to_date_handles_empty_string():
    assert _to_date("") is None


def test_to_date_handles_none():
    assert _to_date(None) is None


def test_to_date_parses_iso_string():
    from datetime import date

    assert _to_date("2020-06-15") == date(2020, 6, 15)

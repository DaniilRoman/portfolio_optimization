"""Point-in-time universe resolution from universe_history CSVs."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Union

_HISTORY_DIR = Path(__file__).parent / "universe_history"

DateLike = Union[str, date, datetime, None]


def resolve_universe(name: str, as_of: DateLike = None) -> list[str]:
    """Return tickers alive in *name* universe on *as_of* date.

    A ticker is considered alive when:
      - listed_at is empty OR listed_at <= as_of
      - delisted_at is empty OR delisted_at > as_of

    When *as_of* is None, returns only tickers with no delisted_at (current members).
    """
    csv_path = _HISTORY_DIR / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No universe history for {name!r}: {csv_path}")

    as_of_date: date | None = _to_date(as_of)

    tickers: list[str] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            listed = _to_date(row.get("listed_at", ""))
            delisted = _to_date(row.get("delisted_at", ""))

            if as_of_date is None:
                if delisted is None:
                    tickers.append(row["ticker"])
            else:
                if listed is not None and listed > as_of_date:
                    continue
                if delisted is not None and delisted <= as_of_date:
                    continue
                tickers.append(row["ticker"])

    return tickers


def _to_date(value: DateLike) -> date | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None

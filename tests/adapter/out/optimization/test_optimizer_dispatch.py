"""Contract tests: every registered optimizer backend must be reachable via the dispatcher."""

from unittest.mock import patch

import pytest

from config.configuration import settings
from src.adapter.out.optimization import optimizer_dispatcher
from src.domain.data.data import AnalysisReport, OptimizationResult
from tests.factories import make_optimization_result, make_stock_data

_BACKEND_TARGET: dict[str, str] = {
    "ga": "src.adapter.out.optimization.optimizer.optimize",
    "cvxpy": "src.adapter.out.optimization.optimizer_cvxpy.optimize",
}


def _make_stocks() -> list[AnalysisReport]:
    return [make_stock_data() for _ in range(3)]


@pytest.mark.parametrize("backend", optimizer_dispatcher._OPTIMIZERS.keys())
def test_dispatcher_routes_to_every_optimizer(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each registered optimizer key must be reachable through the dispatcher and return OptimizationResult."""
    monkeypatch.setattr(settings, "OPTIMIZER", backend)
    expected = make_optimization_result()
    with patch(_BACKEND_TARGET[backend], return_value=expected):
        result = optimizer_dispatcher.optimize(_make_stocks(), budget=100)
    assert isinstance(result, OptimizationResult)
    assert result is expected

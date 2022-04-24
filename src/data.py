import uuid
from enum import Enum
from typing import List

from dataclasses import dataclass, field
from dacite import from_dict, Config
import jsons


class OptimizationJobStatus(Enum):
    CREATED = "CREATED"
    STARTED = "STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    PREPARING_DATA = "PREPARING_DATA"
    OPTIMIZATION = "OPTIMIZATION"
    CANCELED = "CANCELED"
    FINISHED = "FINISHED"


@dataclass
class StockOptimizationJob:
    # init data
    stock_names: List[str]
    max_stock_count_list: List[int]
    budget: int
    predict_period_days: int
    is_backtest: bool = False

    # enriched data
    current_prices: List[float] = field(default_factory=list)
    predicted_prices: List[float] = field(default_factory=list)
    best_set: List[int] = field(default_factory=list)
    real_prices: List[int] = field(default_factory=list)

    # support data
    _id: str = str(uuid.uuid4())
    status: OptimizationJobStatus = OptimizationJobStatus.CREATED

    @staticmethod
    def from_dict(job_dict):
        return from_dict(data_class=StockOptimizationJob, data=job_dict, config=Config(type_hooks={OptimizationJobStatus: OptimizationJobStatus}))

    def to_dict(self):
        return jsons.dump(self)

@dataclass
class OptimizationResult:
    _id: str
    stock_names: List[str]
    max_stock_count_list: List[int]
    budget: int
    predict_period_days: int

    total_cost: float
    total_predicted_cost: float
    total_real_cost: float

    profit: float
    profit_percent: float
    real_profit: float
    real_profit_percent: float

    current_prices: List[float] = field(default_factory=list)
    predicted_prices: List[float] = field(default_factory=list)
    real_prices: List[float] = field(default_factory=list)
    best_set: List[int] = field(default_factory=list)

    by_stock_cost: List[float] = field(default_factory=list)
    by_stock_predicted_cost: List[float] = field(default_factory=list)
    by_stock_real_cost: List[float] = field(default_factory=list)

    @staticmethod
    def from_dict(job_dict):
        return from_dict(data_class=OptimizationResult, data=job_dict)

    def to_dict(self):
        return jsons.dump(self)

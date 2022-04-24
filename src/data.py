import uuid
from enum import Enum
from typing import List

from dataclasses import dataclass


class OptimizationJobStatus(Enum):
    CREATED = "CREATED"
    STARTED = "STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    CANCELED = "CANCELED"
    FINISHED = "FINISHED"


@dataclass
class StockOptimizationJob:
    # init data
    stock_names: List[str]
    max_stock_count_list: List[int]
    budget: int
    predict_period_days: int

    # enriched data
    current_prices: List[float]
    predicted_data: List[float]
    best_set: List[int]

    # support data
    id: str = str(uuid.uuid4())
    status: OptimizationJobStatus = OptimizationJobStatus.STARTED
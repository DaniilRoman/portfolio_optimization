import logging
import os
from typing import Literal

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Read APP_ENV before Settings is instantiated so the profile file path can be
# computed at class-definition time (SettingsConfigDict is evaluated once).
_APP_ENV = os.getenv("APP_ENV", "prod")
_PROFILE_FILE = f"config/profiles/{_APP_ENV}.env"


class GASettings(BaseModel):
    population: int = 120
    generations: int = 350
    tournament_size: int = 5
    mutation_rate: float = 0.55
    crossover_rate: float = 0.35
    mutation_indpb: float = 0.4
    mate_indpb: float = 0.1
    max_sector_concentration: float = 0.40


class CVXPYSettings(BaseModel):
    solvers: list[str] = ["ECOS_BB", "OSQP"]
    company_max_exposure: float = 0.10
    risk_gamma: float = 0.01


class HRPSettings(BaseModel):
    linkage_method: Literal["single", "complete", "average", "ward"] = "single"
    risk_measure: Literal["variance", "cvar"] = "variance"


class BacktestSettings(BaseModel):
    train_window_days: int = 252
    rebalance_days: int = 21
    transaction_cost: float = 0.0005
    slippage: float = 0.0002


class Settings(BaseSettings):
    # Credentials — required in prod; optional (default "") in dev/backtest.
    TELEGRAM_TO: str = ""
    TELEGRAM_TOKEN: str = ""
    GET_AND_INCREMENT_COUNTER_URL: str = ""
    APP_SCRIPT_ID: str = ""

    APP_ENV: Literal["dev", "prod", "backtest"] = "prod"
    TELEGRAM_ENABLED: bool = True
    PREDICTER: Literal["garch", "prophet"] = "garch"
    OPTIMIZER: Literal["ga", "cvxpy", "hrp"] = "ga"
    EXPECTED_RETURNS: Literal["mean", "bl"] = "mean"
    FACTOR_SCREEN_PERCENTILE: float = 0.0
    UNIVERSE: Literal["sp500", "etf", "de_etf", "ishares", "vanguard"] = "etf"
    BENCHMARK: dict[str, str] = {
        "sp500": "SPY",
        "etf": "ACWI",
        "de_etf": "EXSA.DE",
        "ishares": "ACWI",
        "vanguard": "ACWI",
    }
    RISK_FREE_SERIES: str = "DGS3MO"
    ga: GASettings = GASettings()
    cvxpy: CVXPYSettings = CVXPYSettings()
    hrp: HRPSettings = HRPSettings()
    backtest: BacktestSettings = BacktestSettings()

    # Profile file is loaded after .env; env vars in the shell win over both.
    model_config = SettingsConfigDict(
        env_file=[".env", _PROFILE_FILE],
        env_nested_delimiter="__",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _require_telegram_credentials(self) -> "Settings":
        if self.TELEGRAM_ENABLED:
            missing = [f for f in ("TELEGRAM_TOKEN", "TELEGRAM_TO") if not getattr(self, f)]
            if missing:
                raise ValueError(f"Required when TELEGRAM_ENABLED=true: {missing}")
        return self


settings = Settings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

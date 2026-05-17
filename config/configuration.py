import logging
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class Settings(BaseSettings):
    TELEGRAM_TO: str
    TELEGRAM_TOKEN: str
    GET_AND_INCREMENT_COUNTER_URL: str
    APP_SCRIPT_ID: str
    PREDICTER: Literal["garch", "prophet"] = "garch"
    OPTIMIZER: Literal["ga", "cvxpy"] = "ga"
    ga: GASettings = GASettings()
    cvxpy: CVXPYSettings = CVXPYSettings()

    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")


settings = Settings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ADR-001: pydantic-settings over dynaconf / decouple

**Date:** 2026-05  
**Status:** Accepted

## Context

The codebase originally used `python-decouple` for environment variable loading — scattered `config(...)` calls with no central validation and no type safety. Running with a missing `TELEGRAM_TOKEN` would fail deep inside a Telegram call rather than at startup.

## Decision

Replace all `decouple.config(...)` calls with a single `Settings(BaseSettings)` class in `config/configuration.py`. Load order: `.env` → profile overlay (`config/profiles/{APP_ENV}.env`) → shell env vars (highest priority).

## Rationale

- **pydantic-settings** is already a pydantic-ecosystem dependency (pydantic v2), so it adds no new package.  
- Typed fields catch misconfigured values at startup — before any network call happens.  
- Nested `GASettings` / `CVXPYSettings` models group algorithm hyperparameters cleanly.  
- `model_validator` enforces credential requirements only when `TELEGRAM_ENABLED=true`, keeping `dev` and `backtest` profiles credential-free.

## Alternatives considered

- **dynaconf**: Feature-rich but adds a heavy dependency and its own config-file format.  
- **python-decouple** (status quo): No validation, no types, no nested structures.

## Consequences

- Single import: `from config.configuration import settings`.  
- Tests override settings via `monkeypatch.setattr(settings, "PREDICTER", "prophet")`.  
- Adding a new setting requires one field in `Settings`; it is automatically documented by `docs/configuration.md`.

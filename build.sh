#!/usr/bin/env bash
# Single source of truth for the verification pipeline.
# Run locally before pushing; CI runs this same script.

set -euo pipefail

: "${TELEGRAM_TO:=dummy}"
: "${TELEGRAM_TOKEN:=dummy}"
: "${GET_AND_INCREMENT_COUNTER_URL:=http://localhost}"
: "${APP_SCRIPT_ID:=dummy}"
export TELEGRAM_TO TELEGRAM_TOKEN GET_AND_INCREMENT_COUNTER_URL APP_SCRIPT_ID

step() {
  printf '\n==> %s\n' "$1"
}

step "poetry install"
poetry install

step "ruff check"
poetry run ruff check .

step "ruff format --check"
poetry run ruff format --check .

step "mypy"
poetry run mypy src tests

step "pytest"
poetry run pytest

printf '\nbuild.sh: OK\n'

# CLAUDE.md

Python 3.12 / Poetry project for portfolio optimization and stock-pick prediction.

## Verification

Before ending any coding session, run the build script from the repo root and ensure it exits 0:

```bash
bash build.sh
```

`build.sh` is the single source of truth for what CI runs. It executes `poetry install`, `ruff check`, `ruff format --check`, `mypy src tests`, and `pytest` (with dummy env vars for the integrations).

If `build.sh` fails, fix the underlying issue. Do not skip checks, pass `--no-verify`, or comment out failing tests to get a green run.

When adding a new check (lint rule, test category, type-check scope), edit `build.sh` — do not add the step inline to `.github/workflows/ci.yml`.

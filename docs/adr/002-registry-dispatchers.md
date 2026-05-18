# ADR-002: Registry dict + importlib over if/elif dispatchers

**Date:** 2026-05  
**Status:** Accepted

## Context

The predicter and optimizer dispatchers originally contained `if/elif` chains to select backends:

```python
if backend == "prophet":
    from .predicter_prophet import predict as _predict
else:
    from .predicter_garch import predict as _predict
```

Adding a new algorithm required editing the dispatcher source — easy to forget and not discoverable.

## Decision

Replace the `if/elif` chains with a `_BACKENDS: dict[str, str]` registry mapping name → relative module path, dispatched via `importlib.import_module`:

```python
_BACKENDS: dict[str, str] = {"garch": ".predicter_garch", "prophet": ".predicter_prophet"}

module = importlib.import_module(_BACKENDS[backend], package=__package__)
return module.predict(data, predict_period=predict_period, **kwargs)
```

## Rationale

- **One-file change**: adding a new backend requires only one entry in `_BACKENDS`/`_OPTIMIZERS`.  
- **Lazy imports preserved**: `importlib.import_module` is called at dispatch time, not at module load, so expensive deps (prophet, cvxpy) are not imported unless the backend is actually selected.  
- **Free parametrized tests**: `pytest.mark.parametrize` over `_BACKENDS.keys()` gives contract coverage for every registered backend automatically.

## Alternatives considered

- **Class-based registry** (`BACKENDS: dict[str, type[Predictor]]`): cleaner typing but requires wrapping the existing module-level functions in classes — unnecessary indirection for two backends.  
- **Entry points / plugin system**: overkill for an internal registry with two entries.

## Consequences

- `mypy` cannot infer the return type of `module.predict(...)` without a cast; `# type: ignore[no-any-return]` is used at the call site.  
- The `_BACKENDS` / `_OPTIMIZERS` dicts serve as the source of truth for valid setting values (the `Literal` in `Settings` must be kept in sync manually).

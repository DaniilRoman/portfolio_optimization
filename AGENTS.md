# Portfolio Optimization - Agent Instructions

## Project Overview

Python project managed by Poetry for portfolio optimization and stock analysis.

## Project Structure

```
portfolio_optimization/
├── main.py           # Application entry point
├── config/           # Configuration files
├── src/              # Source code (main package)
│   ├── adapter/      # External interfaces (download, predict, notify)
│   ├── infrastructure/
│   └── logic/        # Business logic
└── tests/            # Test files (mirrors src structure)
```

## Essential Commands

### Setup
```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Running the Application
```bash
# Run main application
poetry run python main.py
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/adapter/out/optimization/test_optimizer.py

# Run with verbose output
poetry run pytest -v

# Run with print statements visible
poetry run pytest -s
```

### Dependency Management
```bash
# Add a new dependency
poetry add <package-name>

# Add a dev dependency
poetry add --group dev <package-name>

# Update dependencies
poetry update
```

## Important Notes

- All packages (`src/`, `config/`, `tests/`) have `__init__.py` files
- Tests are configured via `pyproject.toml` with `pythonpath = ["."]`
- Use `poetry run` prefix for all commands to ensure correct environment

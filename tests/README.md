# Tests

This directory contains unit tests for the `grasp_pose_generation` package.

## Running Tests

### Run all tests:
```bash
pytest
```

### Run specific test file:
```bash
pytest tests/test_models.py
pytest tests/test_evaluation.py
pytest tests/test_data.py
```

### Run with verbose output:
```bash
pytest -v
```

### Run with coverage report:
```bash
pytest --cov=src/grasp_pose_generation --cov-report=html
```

## Test Structure

- `test_models.py` - Tests for CVAE model architectures
- `test_evaluation.py` - Tests for evaluation metrics (MSE, MAE)
- `test_data.py` - Tests for data loading and preprocessing utilities
- `conftest.py` - Pytest configuration and shared fixtures

## Adding New Tests

When adding new functionality, create corresponding test files following the naming convention `test_*.py`. Use pytest fixtures for shared test data and follow the existing test patterns.


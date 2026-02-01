"""
Pytest configuration file for portfolio optimization tests.
Sets up environment variables for testing.
"""
import os
from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def mock_environment_variables():
    """Mock environment variables for all tests."""
    env_vars = {
        'TELEGRAM_TO': 'test_telegram_to',
        'TELEGRAM_TOKEN': 'test_telegram_token',
        'GET_AND_INCREMENT_COUNTER_URL': 'http://test.url',
        'APP_SCRIPT_ID': 'test_app_script_id'
    }
    
    # Patch environment variables for the entire test session
    with patch.dict(os.environ, env_vars):
        yield
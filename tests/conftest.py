"""
Pytest configuration file for portfolio optimization tests.
Sets up environment variables before any tests are collected.
"""
import os


def pytest_configure(config):
    """Configure pytest - runs before test collection."""
    # Set environment variables for testing
    # This runs before any tests are collected, so imports will use these values
    os.environ.update({
        'TELEGRAM_TO': 'test_telegram_to',
        'TELEGRAM_TOKEN': 'test_telegram_token',
        'GET_AND_INCREMENT_COUNTER_URL': 'http://test.url',
        'APP_SCRIPT_ID': 'test_app_script_id'
    })
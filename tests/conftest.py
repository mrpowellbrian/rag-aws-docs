"""Pytest configuration and shared fixtures."""

import os

# Ensure the config singleton reads from environment only — no .env file
# present in CI. Set a dummy API key so the Settings validator doesn't warn.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

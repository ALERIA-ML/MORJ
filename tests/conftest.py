import os

import pytest


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Tests use explicit settings, never the operator's MORJ_* environment."""
    for name in list(os.environ):
        if name.startswith('MORJ_'):
            monkeypatch.delenv(name)

import os

import pytest


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Tests use explicit settings, never the operator's JUDGE_* (or old MORJ_*) environment."""
    for name in list(os.environ):
        if name.startswith(('JUDGE_', 'MORJ_')):
            monkeypatch.delenv(name)

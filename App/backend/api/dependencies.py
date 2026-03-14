from __future__ import annotations

from functools import lru_cache

from App.backend.orchestration.pipeline import BackendRuntime


@lru_cache(maxsize=1)
def get_runtime() -> BackendRuntime:
    return BackendRuntime()

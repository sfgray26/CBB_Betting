from __future__ import annotations

import os


_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY_VALUES


def deployment_role() -> str:
    return (os.getenv("DEPLOYMENT_ROLE") or "primary").strip().lower()


def main_scheduler_enabled() -> bool:
    return env_flag("ENABLE_MAIN_SCHEDULER", True)


def startup_catchup_enabled() -> bool:
    raw = os.getenv("ENABLE_STARTUP_CATCHUP")
    if raw is None:
        return main_scheduler_enabled()
    return raw.strip().lower() in _TRUTHY_VALUES
import os
import logging

logger = logging.getLogger(__name__)

def get_float_env(key: str, default: str) -> float:
    """Robustly parse float from environment variable, handling leading spaces or equals signs."""
    val = os.getenv(key, default).strip()
    if val.startswith("="):
        val = val[1:].strip()
    try:
        # Handle cases like "1.15 " or " =1.15"
        return float(val)
    except ValueError:
        logger.warning("Failed to parse env var %s='%s', using default %s", key, val, default)
        try:
            return float(default)
        except ValueError:
            return 0.0

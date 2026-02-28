"""Core mathematics and configuration for the CBB Edge betting framework.

This package contains pure, sport-agnostic building blocks:

- ``odds_math``    — vig removal, probability conversion, dynamic SD
- ``kelly``        — Kelly criterion sizing and portfolio divisor
- ``sport_config`` — per-sport constants (D1 averages, home advantage, etc.)
- ``sim_interface``— ABCs and DTOs for swappable pricing engines

Nothing in this package imports from ``backend.services`` or ``backend.models``.
All modules are side-effect-free and unit-testable in isolation.
"""

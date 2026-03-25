"""
OpenClaw v4.0 — Autonomous Model Operations

Phase 1: Performance Monitor + Pattern Detector (Guardian-compliant, read-only)

This module provides autonomous monitoring and pattern detection for the CBB Edge
betting model and future MLB model. It operates in read-only mode until Apr 7, 2026,
after which self-improvement features can be activated.

Components:
- PerformanceMonitor: CLV decay detection, win rate tracking
- PatternDetector: CBB and MLB-specific vulnerability detection
- Database: Time-series metrics and vulnerability storage

Usage:
    from backend.services.openclaw import PerformanceMonitor, PatternDetector
    
    monitor = PerformanceMonitor()
    decay_report = monitor.check_clv_decay()
    
    detector = PatternDetector(sport='cbb')
    vulnerabilities = detector.analyze(game_context)
"""

from .performance_monitor import PerformanceMonitor, DecayReport
from .pattern_detector import PatternDetector, Vulnerability
from .database import OpenClawDB

__all__ = [
    'PerformanceMonitor',
    'PatternDetector',
    'DecayReport',
    'Vulnerability',
    'OpenClawDB',
]

__version__ = '4.0.0'

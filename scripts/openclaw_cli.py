#!/usr/bin/env python3
"""
OpenClaw CLI - Manual operations and diagnostics

Usage:
    python scripts/openclaw_cli.py check-performance --sport cbb
    python scripts/openclaw_cli.py run-sweep --sport cbb --days 30
    python scripts/openclaw_cli.py health-summary --sport cbb
    python scripts/openclaw_cli.py status
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.openclaw import PerformanceMonitor, PatternDetector, OpenClawDB


def cmd_check_performance(args):
    """Run performance monitor check."""
    print(f"Running performance check for {args.sport}...")
    
    db = OpenClawDB()
    monitor = PerformanceMonitor(sport=args.sport, db=db)
    
    # Check CLV decay
    report = monitor.check_clv_decay(window_hours=args.window)
    
    print(f"\n=== CLV Decay Report ({args.sport.upper()}) ===")
    print(f"Decay Rate: {report.current_decay_pct:.2f}%")
    print(f"Severity: {report.severity.value.upper()}")
    print(f"Sample Size: {report.sample_size}")
    print(f"Trend: {report.trend}")
    print(f"Confidence: {report.confidence*100:.0f}%")
    
    if report.details:
        print(f"Details: {json.dumps(report.details, indent=2, default=str)}")
    
    # Check win rate
    wr = monitor.check_win_rate(days=args.days)
    print(f"\n=== Win Rate Report ({args.sport.upper()}) ===")
    print(f"Actual: {wr.actual_win_rate*100:.1f}%")
    print(f"Expected: {wr.expected_win_rate*100:.1f}%")
    print(f"Variance: {wr.variance_from_expected*100:+.1f}%")
    print(f"Sample Size: {wr.sample_size}")
    print(f"Significant: {wr.is_significant}")
    
    return 0 if report.severity.value in ('normal', 'elevated') else 1


def cmd_run_sweep(args):
    """Run pattern detection sweep."""
    print(f"Running pattern sweep for {args.sport} (last {args.days} days)...")
    
    db = OpenClawDB()
    detector = PatternDetector(sport=args.sport, db=db)
    
    report = detector.run_sweep(days=args.days)
    
    print(f"\n=== Pattern Sweep Report ({args.sport.upper()}) ===")
    print(f"Games Analyzed: {report.games_analyzed}")
    print(f"Days Analyzed: {report.days_analyzed}")
    print(f"Patterns Checked: {report.patterns_checked}")
    print(f"Vulnerabilities Found: {len(report.vulnerabilities)}")
    
    if report.vulnerabilities:
        by_sev = report.by_severity()
        print(f"\nBreakdown:")
        for sev in ['CRITICAL', 'WARNING', 'INFO']:
            count = len(by_sev.get(sev, []))
            if count > 0:
                print(f"  {sev}: {count}")
        
        print(f"\nDetails:")
        for vuln in report.vulnerabilities[:10]:  # Show first 10
            print(f"\n  [{vuln.severity}] {vuln.pattern_type.value}")
            print(f"    Confidence: {vuln.confidence*100:.0f}%")
            print(f"    Affected Games: {vuln.affected_games}")
            print(f"    Description: {vuln.description}")
            print(f"    Recommendation: {vuln.recommended_action}")
    
    return 1 if report.has_critical() else 0


def cmd_health_summary(args):
    """Get health summary."""
    print(f"Generating health summary for {args.sport}...")
    
    db = OpenClawDB()
    monitor = PerformanceMonitor(sport=args.sport, db=db)
    
    summary = monitor.get_health_summary()
    
    print(f"\n=== Health Summary ({args.sport.upper()}) ===")
    print(f"Status: {summary['status'].upper()}")
    print(f"Checked At: {summary['checked_at']}")
    
    print(f"\nMetrics:")
    for metric, value in summary['metrics'].items():
        if isinstance(value, dict):
            print(f"  {metric}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {metric}: {value}")
    
    if summary['alerts']:
        print(f"\nActive Alerts:")
        for alert in summary['alerts']:
            print(f"  [{alert['level']}] {alert['message']}")
    else:
        print(f"\nNo active alerts")
    
    return 0 if summary['status'] == 'healthy' else 1


def cmd_status(args):
    """Show OpenClaw system status."""
    db = OpenClawDB()
    
    print("=== OpenClaw System Status ===")
    print(f"Version: 4.0.0 (Phase 1)")
    print(f"Guardian Freeze: {'ACTIVE' if db.is_guardian_active() else 'LIFTED'}")
    print(f"Guardian Lift Date: {db.GUARDIAN_LIFT_DATE.date()}")
    print(f"Current Date: {datetime.now().date()}")
    
    print(f"\nPhase 1 Components:")
    print(f"  [OK] Performance Monitor")
    print(f"  [OK] Pattern Detector")
    print(f"  [OK] Database Layer")
    print(f"  [OK] Scheduler Integration")
    
    print(f"\nSupported Sports:")
    for sport in ['cbb', 'mlb']:
        detector = PatternDetector(sport=sport)
        patterns = len(detector._pattern_weights)
        print(f"  {sport.upper()}: {patterns} pattern types configured")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='OpenClaw CLI - Model monitoring and diagnostics'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # check-performance
    check_parser = subparsers.add_parser(
        'check-performance',
        help='Run performance monitor check'
    )
    check_parser.add_argument(
        '--sport', '-s',
        choices=['cbb', 'mlb'],
        default='cbb',
        help='Sport to check (default: cbb)'
    )
    check_parser.add_argument(
        '--window', '-w',
        type=int,
        default=48,
        help='CLV analysis window in hours (default: 48)'
    )
    check_parser.add_argument(
        '--days', '-d',
        type=int,
        default=14,
        help='Win rate analysis period in days (default: 14)'
    )
    check_parser.set_defaults(func=cmd_check_performance)
    
    # run-sweep
    sweep_parser = subparsers.add_parser(
        'run-sweep',
        help='Run pattern detection sweep'
    )
    sweep_parser.add_argument(
        '--sport', '-s',
        choices=['cbb', 'mlb'],
        default='cbb',
        help='Sport to analyze (default: cbb)'
    )
    sweep_parser.add_argument(
        '--days', '-d',
        type=int,
        default=30,
        help='Lookback period in days (default: 30)'
    )
    sweep_parser.set_defaults(func=cmd_run_sweep)
    
    # health-summary
    health_parser = subparsers.add_parser(
        'health-summary',
        help='Get health summary'
    )
    health_parser.add_argument(
        '--sport', '-s',
        choices=['cbb', 'mlb'],
        default='cbb',
        help='Sport to check (default: cbb)'
    )
    health_parser.set_defaults(func=cmd_health_summary)
    
    # status
    status_parser = subparsers.add_parser(
        'status',
        help='Show OpenClaw system status'
    )
    status_parser.set_defaults(func=cmd_status)
    
    args = parser.parse_args()

    # PAUSED (2026-04-21): OpenClaw CLI is on hold until the baseball module is
    # fully implemented.
    if not args.command:
        parser.print_help()
    print("\nOpenClaw CLI is paused. Re-enable when baseball module is complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

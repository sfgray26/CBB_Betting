"""
OpenClaw HANDOFF.md Integration Module

Provides safe, structured updates to HANDOFF.md from OpenClaw processes.
Called by analysis.py after integrity sweep, and by baseline script after execution.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

HANDOFF_PATH = Path("HANDOFF.md")


def update_openclaw_status(
    last_sweep: Optional[Dict] = None,
    alerts: Optional[List[str]] = None,
    circuit_state: Optional[str] = None,
    baseline_complete: bool = False,
    baseline_stats: Optional[Dict] = None
) -> bool:
    """
    Update OpenClaw status section in HANDOFF.md.
    Called after integrity sweep or baseline completion.
    
    Args:
        last_sweep: Dict with 'timestamp', 'games_checked', 'verdict_counts'
        alerts: List of alert strings to display
        circuit_state: Circuit breaker state (CLOSED/OPEN/HALF_OPEN)
        baseline_complete: Whether O-8 baseline is complete
        baseline_stats: Dict with 'teams_analyzed', 'high_risk_count', etc.
    
    Returns:
        True if update successful, False otherwise
    """
    if not HANDOFF_PATH.exists():
        logger.warning("HANDOFF.md not found, skipping update")
        return False
    
    try:
        content = HANDOFF_PATH.read_text(encoding='utf-8')
        
        # Build the new OpenClaw status section
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        update_lines = [f"### OpenClaw Status (Auto-Updated {now})", ""]
        
        # Status table
        update_lines.extend([
            "| Component | Status | Detail |",
            "|-----------|--------|--------|"
        ])
        
        # Circuit breaker status
        cb_status = circuit_state or "CLOSED"
        cb_emoji = "✅" if cb_status == "CLOSED" else "⚠️"
        update_lines.append(f"| {cb_emoji} Circuit Breaker | {cb_status} | Auto-reset after 60s if OPEN |")
        
        # Last sweep info
        if last_sweep:
            ts = last_sweep.get('timestamp', 'N/A')
            games = last_sweep.get('games_checked', 0)
            update_lines.append(f"| 🔄 Last Integrity Sweep | {ts} | {games} games checked |")
        else:
            update_lines.append(f"| 🔄 Last Integrity Sweep | No recent sweep | — |")
        
        # O-8 Baseline status
        if baseline_complete and baseline_stats:
            teams = baseline_stats.get('teams_analyzed', 0)
            high_risk = baseline_stats.get('high_risk_count', 0)
            update_lines.append(f"| 📊 O-8 Baseline | ✅ COMPLETE | {teams} teams, {high_risk} high-risk |")
        else:
            update_lines.append(f"| 📊 O-8 Baseline | ⏳ PENDING | Execute March 16 ~9 PM ET |")
        
        update_lines.append("")
        
        # Alerts section
        update_lines.append("**Active Alerts:**")
        if alerts:
            for alert in alerts:
                update_lines.append(f"- {alert}")
        else:
            update_lines.append("- None")
        
        update_lines.append("")
        
        # Find and replace existing OpenClaw section, or insert after Section 1 header
        section_content = "\n".join(update_lines)
        
        # Pattern to match existing OpenClaw status section
        pattern = r'### OpenClaw Status \(Auto-Updated.*?(?=\n## |\n### |\Z)'
        
        if re.search(pattern, content, re.DOTALL):
            # Replace existing section
            new_content = re.sub(pattern, section_content.rstrip(), content, flags=re.DOTALL)
        else:
            # Insert after "## 1. SYSTEM STATUS" section
            # Find the end of Section 1 (before Section 2 or first mission)
            section_1_end = re.search(r'(## 1\.[^{}]*?)(?=\n## [2-9]|\n## [A-Z]|\Z)', content, re.DOTALL)
            if section_1_end:
                insert_pos = section_1_end.end()
                new_content = content[:insert_pos] + "\n\n" + section_content + content[insert_pos:]
            else:
                # Fallback: prepend to file
                new_content = section_content + "\n\n" + content
        
        HANDOFF_PATH.write_text(new_content, encoding='utf-8')
        logger.info("HANDOFF.md updated with OpenClaw status")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update HANDOFF.md: {e}")
        return False


def log_integrity_sweep_results(
    slate_date: str,
    games_checked: int,
    verdict_counts: Dict[str, int],
    latency_ms: float,
    circuit_state: str
) -> bool:
    """
    Log integrity sweep results to HANDOFF.md and JSON state file.
    Called by analysis.py after _integrity_sweep() completes.
    """
    # Build alerts based on results
    alerts = []
    
    volatile_count = verdict_counts.get('VOLATILE', 0)
    abort_count = verdict_counts.get('ABORT', 0) + verdict_counts.get('RED FLAG', 0)
    
    if games_checked > 0:
        volatile_pct = (volatile_count / games_checked) * 100
        if volatile_pct > 20:
            alerts.append(f"🚨 SYSTEM_RISK_ELEVATED: {volatile_pct:.0f}% of games VOLATILE")
        if abort_count > 0:
            alerts.append(f"🛑 {abort_count} ABORT/RED FLAG verdict(s) — review immediately")
    
    if circuit_state != "CLOSED":
        alerts.append(f"⚠️ Circuit breaker {circuit_state} — using fallback")
    
    # Update HANDOFF.md
    last_sweep = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "games_checked": games_checked,
        "verdict_counts": verdict_counts,
        "latency_ms": latency_ms
    }
    
    success = update_openclaw_status(
        last_sweep=last_sweep,
        alerts=alerts if alerts else None,
        circuit_state=circuit_state
    )
    
    # Also write to JSON state file for persistence
    try:
        state_file = Path(".openclaw/operational-state.json")
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        
        if "sweeps" not in state:
            state["sweeps"] = []
        
        state["sweeps"].append({
            "date": slate_date,
            **last_sweep
        })
        
        # Keep only last 30 sweeps
        state["sweeps"] = state["sweeps"][-30:]
        state["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        state_file.write_text(json.dumps(state, indent=2))
    except Exception as e:
        logger.warning(f"Failed to write state file: {e}")
    
    return success


def log_baseline_completion(stats: Dict) -> bool:
    """
    Log O-8 baseline completion to HANDOFF.md.
    Called by openclaw_baseline.py after execution.
    """
    alerts = []
    
    high_risk = stats.get('high_risk_count', 0)
    if high_risk > 0:
        alerts.append(f"🔴 {high_risk} teams classified HIGH/CRITICAL risk")
    
    riskiest = stats.get('riskiest_region')
    if riskiest:
        alerts.append(f"📍 {riskiest} Region has highest average risk")
    
    return update_openclaw_status(
        baseline_complete=True,
        baseline_stats=stats,
        alerts=alerts if alerts else ["✅ All teams within acceptable risk parameters"]
    )


# Backward compatibility
update_handoff_section = update_openclaw_status

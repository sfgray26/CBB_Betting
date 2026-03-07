"""
OpenClaw Coordinator v2.0
Intelligent task routing between local LLM (qwen2.5:3b) and remote engines.

Usage:
    from .openclaw.coordinator import OpenClawCoordinator
    
    coordinator = OpenClawCoordinator()
    result = await coordinator.route_task(
        task_type="integrity_check",
        context={"home_team": "Duke", "away_team": "UNC", "recommended_units": 1.0},
        prompt="..."
    )
"""

import asyncio
import json
import logging
import os
import time
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import requests

logger = logging.getLogger("openclaw")


class Engine(Enum):
    LOCAL = "local"
    KIMI = "kimi"
    CLAUDE = "claude"


class TaskType(Enum):
    INTEGRITY_CHECK = "integrity_check"
    SCOUTING_REPORT = "scouting_report"
    MORNING_BRIEFING = "morning_briefing"
    INJURY_ANALYSIS = "injury_analysis"
    HEALTH_NARRATIVE = "health_narrative"


@dataclass
class TaskContext:
    """Context for routing decisions."""
    game_key: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    recommended_units: float = 0.0
    edge_conservative: float = 0.0
    is_neutral: bool = False
    tournament_round: Optional[int] = None
    integrity_verdict: Optional[str] = None
    sources_disagree: bool = False
    star_player_injury: bool = False
    
    # Convenience properties for routing rules
    @property
    def is_elite_eight_or_later(self) -> bool:
        return self.tournament_round is not None and self.tournament_round >= 4
    
    @property
    def is_high_stakes(self) -> bool:
        return self.recommended_units >= 1.5 or self.is_elite_eight_or_later
    
    @property
    def is_volatile(self) -> bool:
        return self.integrity_verdict is not None and "VOLATILE" in self.integrity_verdict.upper()


@dataclass
class TaskResult:
    """Result of a task execution."""
    success: bool
    output: str
    engine_used: Engine
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: float = 0.0
    error: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker pattern for local LLM."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0
        self.lock = asyncio.Lock()
    
    async def can_execute(self) -> bool:
        async with self.lock:
            if self.state == "CLOSED":
                return True
            elif self.state == "OPEN":
                if time.time() - (self.last_failure_time or 0) > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.half_open_calls = 0
                    return True
                return False
            elif self.state == "HALF_OPEN":
                return self.half_open_calls < 2
        return False
    
    async def record_success(self):
        async with self.lock:
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
            else:
                self.failures = max(0, self.failures - 1)
    
    async def record_failure(self):
        async with self.lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.state == "HALF_OPEN":
                self.state = "OPEN"
            elif self.failures >= self.failure_threshold:
                self.state = "OPEN"


class OpenClawCoordinator:
    """
    Central coordinator for OpenClaw local LLM operations.
    Routes tasks between local and remote engines based on stakes and complexity.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or ".openclaw/config.yaml"
        self.config = self._load_config()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.get("local", {}).get("circuit_breaker", {}).get("failure_threshold", 5),
            recovery_timeout=self.config.get("local", {}).get("circuit_breaker", {}).get("recovery_timeout_seconds", 60)
        )
        self.usage_log: List[Dict] = []
        self.daily_cost_usd = 0.0
        self.last_budget_reset = datetime.now().date()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration if file not found."""
        return {
            "routing": {"default_engine": "local", "fallback_on_timeout": True},
            "local": {
                "url": "http://localhost:11434/api/generate",
                "model": "qwen2.5:3b",
                "limits": {"max_concurrent": 8, "timeout_seconds": 10}
            },
            "tracking": {"enabled": True, "budgets": {"kimi_daily_usd": 5.0}}
        }
    
    def _check_budget(self) -> bool:
        """Check if we're within daily budget."""
        today = datetime.now().date()
        if today != self.last_budget_reset:
            self.daily_cost_usd = 0.0
            self.last_budget_reset = today
        
        budget = self.config.get("tracking", {}).get("budgets", {}).get("kimi_daily_usd", 5.0)
        return self.daily_cost_usd < budget * 0.8  # Stay under 80%
    
    def _select_engine(self, task_type: TaskType, context: TaskContext) -> Engine:
        """
        Select appropriate engine based on routing rules.
        Rules are evaluated in order - first match wins.
        """
        rules = self.config.get("routing", {}).get("rules", [])
        
        for rule in rules:
            condition = rule.get("condition", "")
            engine_name = rule.get("engine", "local")
            
            # Evaluate condition
            if self._evaluate_condition(condition, context):
                # Check if we can use the preferred engine
                if engine_name == "local":
                    return Engine.LOCAL
                elif engine_name == "kimi":
                    # Check budget before routing to Kimi
                    if self._check_budget():
                        return Engine.KIMI
                    else:
                        logger.warning("Budget limit approaching, using local")
                        return Engine.LOCAL
                elif engine_name == "claude":
                    return Engine.CLAUDE
        
        return Engine.LOCAL  # Default fallback
    
    def _evaluate_condition(self, condition: str, context: TaskContext) -> bool:
        """Evaluate a routing condition against context."""
        # Simple condition evaluation - could be made more sophisticated
        condition_map = {
            "elite_eight_or_later": context.is_elite_eight_or_later,
            "recommended_units >= 1.5": context.recommended_units >= 1.5,
            "integrity_verdict contains VOLATILE": context.is_volatile,
            "sources_disagree AND edge_conservative > 0.05": context.sources_disagree and context.edge_conservative > 0.05,
            "integrity_check AND bet_tier": True,  # Default for integrity
            "injury_impact AND star_player": context.star_player_injury,
            "scouting_report": True,
            "health_summary": True,
            "briefing_narrative": True,
        }
        return condition_map.get(condition, False)
    
    async def _call_local(self, prompt: str, task_type: TaskType) -> TaskResult:
        """Call local Ollama instance."""
        if not await self.circuit_breaker.can_execute():
            return TaskResult(
                success=False,
                output="",
                engine_used=Engine.LOCAL,
                latency_ms=0,
                error="Circuit breaker OPEN"
            )
        
        local_config = self.config.get("local", {})
        url = local_config.get("url", "http://localhost:11434/api/generate")
        model = local_config.get("model", "qwen2.5:3b")
        timeout = local_config.get("limits", {}).get("timeout_seconds", 10)
        
        # Get task-specific parameters
        params = local_config.get("parameters", {}).get(task_type.value, {})
        if not params:
            params = local_config.get("parameters", {}).get("default", {})
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": params.get("num_predict", 128),
                "temperature": params.get("temperature", 0.3),
                "top_p": params.get("top_p", 0.9),
            }
        }
        
        if params.get("format") == "json":
            payload["format"] = "json"
        
        start_time = time.time()
        try:
            # Run sync request in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, json=payload, timeout=timeout)
            )
            response.raise_for_status()
            result = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            output = result.get("response", "").strip()
            
            await self.circuit_breaker.record_success()
            
            return TaskResult(
                success=True,
                output=output,
                engine_used=Engine.LOCAL,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            await self.circuit_breaker.record_failure()
            logger.warning(f"Local LLM call failed: {e}")
            return TaskResult(
                success=False,
                output="",
                engine_used=Engine.LOCAL,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _call_kimi(self, prompt: str, task_type: TaskType) -> TaskResult:
        """
        Call Kimi CLI for high-stakes tasks.
        In practice, this returns a signal that the caller should route to Kimi.
        """
        # Since Kimi is the coordinator, this is a no-op that signals escalation
        return TaskResult(
            success=False,
            output="ESCALATE_TO_KIMI",
            engine_used=Engine.KIMI,
            latency_ms=0,
            error="Task escalated to Kimi CLI for high-stakes analysis"
        )
    
    async def route_task(
        self,
        task_type: TaskType,
        context: TaskContext,
        prompt: str,
        allow_escalation: bool = True
    ) -> TaskResult:
        """
        Route a task to the appropriate engine.
        
        Args:
            task_type: Type of task being performed
            context: Context for routing decisions
            prompt: The prompt to send to the LLM
            allow_escalation: Whether to allow escalation to remote engines
            
        Returns:
            TaskResult with output and metadata
        """
        # Select engine
        engine = self._select_engine(task_type, context)
        
        # Try primary engine
        if engine == Engine.LOCAL:
            result = await self._call_local(prompt, task_type)
            
            # Handle escalation
            if not result.success and allow_escalation:
                if self.config.get("routing", {}).get("fallback_on_timeout", True):
                    logger.info(f"Local failed for {task_type.value}, escalating to Kimi")
                    return await self._call_kimi(prompt, task_type)
            
            return result
            
        elif engine == Engine.KIMI:
            return await self._call_kimi(prompt, task_type)
        
        else:
            return TaskResult(
                success=False,
                output="",
                engine_used=engine,
                latency_ms=0,
                error=f"Unknown engine: {engine}"
            )
    
    def log_usage(self, result: TaskResult, task_type: TaskType, context: TaskContext):
        """Log task usage for tracking."""
        if not self.config.get("tracking", {}).get("enabled", True):
            return
        
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "task_type": task_type.value,
            "engine": result.engine_used.value,
            "game_key": context.game_key,
            "success": result.success,
            "latency_ms": result.latency_ms,
            "cost_usd": result.estimated_cost_usd
        }
        
        self.usage_log.append(entry)
        self.daily_cost_usd += result.estimated_cost_usd
        
        # Write to file periodically (every 10 entries)
        if len(self.usage_log) >= 10:
            self._flush_usage_log()
    
    def _send_notification(self, event_name: str, context: Dict):
        """
        Send Discord notification for configured events.
        Uses discord_notifier.py if available, otherwise logs only.
        """
        notifications = self.config.get("notifications", {})
        events = notifications.get("events", [])
        
        # Find matching event config
        event_config = None
        for e in events:
            if e.get("name") == event_name:
                event_config = e
                break
        
        if not event_config:
            return
        
        # Build message
        message_template = event_config.get("message", "")
        try:
            message = message_template.format(**context)
        except KeyError:
            message = message_template
        
        # Try to send via discord_notifier
        try:
            # Import here to avoid circular dependency
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from backend.services.discord_notifier import _post
            
            payload = {
                "content": message,
                "flags": 0  # No special flags
            }
            
            success = _post(payload)
            if success:
                logger.info(f"Notification sent: {event_name}")
            else:
                # Discord not configured, log to file instead
                self._log_notification_fallback(event_name, message)
                
        except Exception as e:
            logger.debug(f"Discord notification failed (expected if not configured): {e}")
            self._log_notification_fallback(event_name, message)
    
    def _log_notification_fallback(self, event_name: str, message: str):
        """Log notification to file when Discord unavailable."""
        log_dir = Path(".openclaw/notifications")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        timestamp = datetime.utcnow().isoformat()
        
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {event_name}: {message}\n")
    
    def _flush_usage_log(self):
        """Write usage log to disk."""
        log_file = self.config.get("tracking", {}).get("log_file", ".openclaw/token-usage.jsonl")
        try:
            with open(log_file, 'a') as f:
                for entry in self.usage_log:
                    f.write(json.dumps(entry) + '\n')
            self.usage_log = []
        except Exception as e:
            logger.warning(f"Failed to write usage log: {e}")
    
    def get_stats(self) -> Dict:
        """Get coordinator statistics."""
        return {
            "circuit_breaker_state": self.circuit_breaker.state,
            "daily_cost_usd": self.daily_cost_usd,
            "pending_logs": len(self.usage_log),
            "budget_remaining_pct": max(0, 100 - (self.daily_cost_usd / 5.0 * 100))
        }


# Singleton instance
_coordinator: Optional[OpenClawCoordinator] = None


def get_coordinator() -> OpenClawCoordinator:
    """Get or create singleton coordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = OpenClawCoordinator()
    return _coordinator


# Convenience functions for common tasks

async def check_integrity(
    home_team: str,
    away_team: str,
    verdict: str,
    search_results: str,
    context: Optional[TaskContext] = None
) -> str:
    """
    Perform integrity check with intelligent routing.
    
    Usage:
        result = await check_integrity("Duke", "UNC", "Bet 1.0u Duke -4", search_text)
    """
    coordinator = get_coordinator()
    
    ctx = context or TaskContext()
    ctx.home_team = home_team
    ctx.away_team = away_team
    
    prompt = f"""You are a College Basketball Betting Integrity Officer. 

Matchup: {away_team} @ {home_team}
Model Verdict: {verdict}

Real-Time News/Injuries:
{search_results}

Return EXACTLY ONE of: CONFIRMED, CAUTION, VOLATILE, ABORT, RED FLAG
Follow with a brief 1-sentence reason.

Verdict:"""
    
    result = await coordinator.route_task(
        task_type=TaskType.INTEGRITY_CHECK,
        context=ctx,
        prompt=prompt
    )
    
    coordinator.log_usage(result, TaskType.INTEGRITY_CHECK, ctx)
    
    if result.output == "ESCALATE_TO_KIMI":
        # Signal to caller that Kimi should handle this
        coordinator._send_notification("high_stakes_escalation", {
            "home_team": home_team,
            "away_team": away_team,
            "recommended_units": getattr(ctx, 'recommended_units', 0) if ctx else 0,
            "game": f"{away_team} @ {home_team}"
        })
        return "KIMI_ESCALATION"
    
    # Check for VOLATILE verdict
    if result.success and result.output and "VOLATILE" in result.output:
        coordinator._send_notification("integrity_volatile", {
            "home_team": home_team,
            "away_team": away_team,
            "integrity_verdict": result.output[:100],
            "game": f"{away_team} @ {home_team}"
        })
    
    return result.output if result.success else "Sanity check unavailable"


async def generate_scouting_insight(
    home_team: str,
    away_team: str,
    matchup_notes: List[str],
    verdict: str,
    edge: float
) -> str:
    """Generate scouting report with local LLM (always local - low stakes)."""
    coordinator = get_coordinator()
    
    notes_str = "\n".join(matchup_notes) if matchup_notes else "Efficiency ratings favor this side."
    
    prompt = f"""Summarize this betting edge in one punchy sentence (max 20 words).

Game: {away_team} @ {home_team}
Edge: {edge:.1%}
Factors: {notes_str}

Insight:"""
    
    ctx = TaskContext(home_team=home_team, away_team=away_team)
    
    result = await coordinator.route_task(
        task_type=TaskType.SCOUTING_REPORT,
        context=ctx,
        prompt=prompt,
        allow_escalation=False  # Never escalate scouting reports
    )
    
    coordinator.log_usage(result, TaskType.SCOUTING_REPORT, ctx)
    
    return result.output if result.success else "Model identifies matchup advantages"


# Backward compatibility wrapper
async def perform_sanity_check(
    home_team: str,
    away_team: str,
    verdict: str,
    search_results: str
) -> str:
    """Backward-compatible wrapper that uses new routing logic."""
    return await check_integrity(home_team, away_team, verdict, search_results)

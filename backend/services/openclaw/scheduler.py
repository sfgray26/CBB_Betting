"""
OpenClaw Scheduler Integration

Integrates Performance Monitor and Pattern Detector into the
DailyIngestionOrchestrator scheduling system.

Schedule:
- Performance Monitor: Every 2 hours during season
- Pattern Detector: Daily at 6 AM (full sweep)
- Health Summary: Daily at 7 AM (Discord brief)

All monitoring is read-only during Guardian freeze (until Apr 7).
"""

from datetime import datetime
from typing import Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


class OpenClawScheduler:
    """
    Scheduler adapter for OpenClaw monitoring agents.
    
    Integrates with DailyIngestionOrchestrator's APScheduler instance.
    
    Usage:
        from backend.services.daily_ingestion import DailyIngestionOrchestrator
        
        orchestrator = DailyIngestionOrchestrator()
        openclaw = OpenClawScheduler(orchestrator.scheduler)
        openclaw.start_monitoring()
    """
    
    # Job IDs for management
    JOB_PERFORMANCE_MONITOR = 'openclaw_performance_monitor'
    JOB_PATTERN_SWEEP = 'openclaw_pattern_sweep'
    JOB_HEALTH_SUMMARY = 'openclaw_health_summary'
    
    def __init__(
        self,
        scheduler,
        sport: str = 'cbb',
        discord_hook: Optional[Callable] = None
    ):
        """
        Initialize OpenClaw scheduler.
        
        Args:
            scheduler: APScheduler instance from DailyIngestionOrchestrator
            sport: 'cbb' or 'mlb'
            discord_hook: Callable for Discord alerts (optional)
        """
        self.scheduler = scheduler
        self.sport = sport
        self.discord_hook = discord_hook
        self._monitor = None
        self._detector = None
        
    def start_monitoring(self):
        """Schedule all OpenClaw monitoring jobs."""
        from .performance_monitor import PerformanceMonitor, DecaySeverity
        from .pattern_detector import PatternDetector
        
        self._monitor = PerformanceMonitor(sport=self.sport)
        self._detector = PatternDetector(sport=self.sport)
        
        # 1. Performance Monitor: Every 2 hours
        self.scheduler.add_job(
            self._run_performance_check,
            trigger='interval',
            hours=2,
            id=self.JOB_PERFORMANCE_MONITOR,
            name=f'OpenClaw: Performance Monitor ({self.sport})',
            replace_existing=True,
            next_run_time=datetime.now()  # Run immediately on start
        )
        
        # 2. Pattern Detector: Daily at 6 AM
        self.scheduler.add_job(
            self._run_pattern_sweep,
            trigger='cron',
            hour=6,
            minute=0,
            id=self.JOB_PATTERN_SWEEP,
            name=f'OpenClaw: Pattern Sweep ({self.sport})',
            replace_existing=True
        )
        
        # 3. Health Summary: Daily at 7 AM (after pattern sweep)
        self.scheduler.add_job(
            self._send_health_summary,
            trigger='cron',
            hour=7,
            minute=0,
            id=self.JOB_HEALTH_SUMMARY,
            name=f'OpenClaw: Health Summary ({self.sport})',
            replace_existing=True
        )
        
        logger.info(
            f"OpenClaw monitoring scheduled for {self.sport}: "
            f"performance every 2h, sweep daily at 6 AM, summary at 7 AM"
        )
    
    def stop_monitoring(self):
        """Remove all OpenClaw scheduled jobs."""
        for job_id in [
            self.JOB_PERFORMANCE_MONITOR,
            self.JOB_PATTERN_SWEEP,
            self.JOB_HEALTH_SUMMARY
        ]:
            try:
                self.scheduler.remove_job(job_id)
                logger.debug(f"Removed job: {job_id}")
            except Exception:
                pass  # Job may not exist
        
        logger.info("OpenClaw monitoring stopped")
    
    def _run_performance_check(self):
        """Execute performance monitor check."""
        if not self._monitor:
            return
        
        try:
            logger.debug("Running performance monitor check...")
            
            # Check CLV decay
            report = self._monitor.check_clv_decay(window_hours=48)
            
            # Alert on WARNING or CRITICAL
            if report.severity in (DecaySeverity.WARNING, DecaySeverity.CRITICAL):
                logger.warning(
                    f"CLV {report.severity.value}: {report.current_decay_pct:.1f}% decay"
                )
                
                if self.discord_hook:
                    self.discord_hook(report.to_discord_embed())
            
            # Check win rates
            wr_report = self._monitor.check_win_rate(days=14)
            if wr_report.is_significant and wr_report.variance_from_expected < -0.05:
                logger.warning(
                    f"Win rate underperforming: {wr_report.variance_from_expected*100:.1f}%"
                )
                
        except Exception as e:
            logger.error(f"Performance monitor error: {e}", exc_info=True)
    
    def _run_pattern_sweep(self):
        """Execute full pattern detection sweep."""
        if not self._detector:
            return
        
        try:
            logger.info("Running pattern detection sweep...")
            
            report = self._detector.run_sweep(days=30)
            
            # Alert on findings
            if report.vulnerabilities:
                by_sev = report.by_severity()
                
                logger.warning(
                    f"Pattern sweep found {len(report.vulnerabilities)} vulnerabilities: "
                    f"{len(by_sev.get('CRITICAL', []))} critical, "
                    f"{len(by_sev.get('WARNING', []))} warning"
                )
                
                # Send Discord alerts for WARNING and CRITICAL
                if self.discord_hook:
                    for vuln in report.vulnerabilities:
                        if vuln.severity in ('CRITICAL', 'WARNING'):
                            self.discord_hook(vuln.to_discord_embed())
            else:
                logger.info("Pattern sweep complete: no vulnerabilities detected")
                
        except Exception as e:
            logger.error(f"Pattern sweep error: {e}", exc_info=True)
    
    def _send_health_summary(self):
        """Send daily health summary to Discord."""
        if not self._monitor or not self.discord_hook:
            return
        
        try:
            summary = self._monitor.get_health_summary()
            
            # Build summary embed
            embed = {
                'title': f'📊 OpenClaw Health Summary: {self.sport.upper()}',
                'description': f'Daily monitoring status ({datetime.now().strftime("%Y-%m-%d")})',
                'color': 0x00FF00 if summary['status'] == 'healthy' else 
                        0xFFA500 if summary['status'] == 'warning' else 0xFF0000,
                'fields': [
                    {
                        'name': 'Status',
                        'value': summary['status'].upper(),
                        'inline': True
                    },
                    {
                        'name': 'CLV Decay',
                        'value': summary['metrics'].get('clv_decay', {}).get('value', 'N/A'),
                        'inline': True
                    },
                    {
                        'name': 'Win Rate',
                        'value': summary['metrics'].get('win_rate', {}).get('actual', 'N/A'),
                        'inline': True
                    }
                ],
                'timestamp': datetime.now().isoformat()
            }
            
            # Add alerts if any
            if summary['alerts']:
                alert_text = '\n'.join(
                    f"• [{a['level']}] {a['message']}" 
                    for a in summary['alerts'][:5]  # Limit to 5
                )
                embed['fields'].append({
                    'name': 'Active Alerts',
                    'value': alert_text or 'None',
                    'inline': False
                })
            
            self.discord_hook(embed)
            logger.info("Health summary sent")
            
        except Exception as e:
            logger.error(f"Health summary error: {e}", exc_info=True)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        jobs = []
        for job_id in [
            self.JOB_PERFORMANCE_MONITOR,
            self.JOB_PATTERN_SWEEP,
            self.JOB_HEALTH_SUMMARY
        ]:
            try:
                job = self.scheduler.get_job(job_id)
                if job:
                    jobs.append({
                        'id': job.id,
                        'name': job.name,
                        'next_run': job.next_run_time.isoformat() if job.next_run_time else None
                    })
            except Exception:
                pass
        
        return {
            'sport': self.sport,
            'jobs_scheduled': len(jobs),
            'jobs': jobs,
            'monitor_initialized': self._monitor is not None,
            'detector_initialized': self._detector is not None
        }

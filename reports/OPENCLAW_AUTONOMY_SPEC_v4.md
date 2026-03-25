# OpenClaw Autonomy Specification v4.0

> **Workstream Owner:** Kimi CLI (Deep Intelligence Unit)  
> **Status:** DESIGN COMPLETE — Phase 1 implementation starts NOW (March 24)  
> **Target:** Autonomous model monitoring, alpha decay detection, self-improvement  
> **Guardian Compliance:** 
> - ✅ **Phase 1 (Monitoring):** Can implement NOW — read-only, doesn't touch frozen CBB model files
> - ⏸️ **Phase 4 (Auto-Implementation):** Wait until Apr 7 — requires modifying betting_model.py
> 
> **See:** `reports/OPENCLAW_AUTONOMY_SPEC_v4_PHASE1_NOW.md` for corrected timeline

---

## Executive Summary

OpenClaw v4.0 transforms from a **heuristic integrity checker** (v3.0) into an **autonomous model operations system** that monitors performance, detects degradation, and proposes/implement improvements without human intervention.

### Why This Matters

| Current State (v3.0) | Target State (v4.0) | Value |
|---------------------|---------------------|-------|
| Manual performance reviews | 24/7 automated monitoring | Catch edge degradation in <4 hours vs days/weeks |
| Gut-feel pattern detection | Statistical vulnerability analysis | Systematic identification of model weaknesses |
| Static improvement roadmap | Living, auto-prioritized backlog | Always working on highest-ROI improvements |
| Human-in-loop for all changes | Autonomous implementation (with safety) | 10x faster iteration, 80% less oversight |

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         OPENCLAW AUTONOMOUS SYSTEM v4.0                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         SCHEDULER LAYER                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │   │
│  │  │ Every 2h    │  │ Daily 4am   │  │ Weekly Mon  │  │ On-Demand       │ │   │
│  │  │ Performance │  │ Pattern     │  │ Roadmap     │  │ Alert-Driven    │ │   │
│  │  │ Monitor     │  │ Analysis    │  │ Update      │  │ Checks          │ │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │   │
│  │         └─────────────────┴─────────────────┴──────────────────┘         │   │
│  │                         │                                                │   │
│  │                         ▼                                                │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │                  ORCHESTRATOR (openclaw_orchestrator.py)         │    │   │
│  │  │  • Manages agent lifecycle                                        │    │   │
│  │  │  • Handles concurrency and resource limits                        │    │   │
│  │  │  • Routes findings to appropriate outputs                         │    │   │
│  │  └────────────────────────┬────────────────────────────────────────┘    │   │
│  └───────────────────────────┼─────────────────────────────────────────────┘   │
│                              │                                                  │
│                              ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         AGENT LAYER                                      │   │
│  │                                                                          │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐  │   │
│  │  │  PERFORMANCE    │    │    PATTERN      │    │      LEARNING       │  │   │
│  │  │    MONITOR      │◄──►│   DETECTOR      │◄──►│       ENGINE        │  │   │
│  │  │                 │    │                 │    │                     │  │   │
│  │  │ • CLV tracking  │    │ • Loss clustering│   │ • Historical analysis│  │   │
│  │  │ • Win rate trends│   │ • Feature drift │    │ • A/B test design    │  │   │
│  │  │ • Decay alerts  │    │ • Anomaly detection│ │ • Backtest runner    │  │   │
│  │  └────────┬────────┘    └────────┬────────┘    └──────────┬──────────┘  │   │
│  │           │                      │                       │             │   │
│  │           └──────────────────────┼───────────────────────┘             │   │
│  │                                  │                                     │   │
│  │                                  ▼                                     │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐  │   │
│  │  │    ROADMAP      │    │   SELF-IMPROVE  │    │     NOTIFIER        │  │   │
│  │  │   MAINTAINER    │◄──►│     AGENT       │    │      AGENT          │  │   │
│  │  │                 │    │  (Post-Apr 7)   │    │                     │  │   │
│  │  │ • Auto-prioritize│   │ • Recalibration │    │ • Discord alerts    │  │   │
│  │  │ • Update ROADMAP│    │ • Weight adjust │    │ • Health summaries  │  │   │
│  │  │ • Queue tasks   │    │ • Safe rollback │    │ • Escalation queue  │  │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────────┘  │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                  │
│                              ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         OUTPUT LAYER                                     │   │
│  │                                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │   │
│  │  │   Discord    │  │    DB        │  │  ROADMAP.md  │  │   Alerts    │  │   │
│  │  │   #bets      │  │  Telemetry   │  │   Updates    │  │   Dashboard │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Agents Specification

### 2.1 Performance Monitor Agent

**Purpose:** Detect alpha decay in real-time by comparing predictions vs outcomes and tracking CLV.

**Triggers:** Every 2 hours (configurable)

**Inputs:**
- `predictions` table (our projected spreads)
- `closing_lines` table (final market lines)
- `bet_log` table (actual bets placed)

**Outputs:**
- Performance metrics JSON
- Decay alerts (if thresholds breached)
- CLV trend analysis

**Key Metrics Tracked:**

| Metric | Definition | Decay Threshold | Alert Priority |
|--------|-----------|-----------------|----------------|
| **CLV Capture Rate** | % of bets where our line beat close | <55% (from baseline) | HIGH |
| **Win Rate (7d)** | Rolling 7-day win percentage | <45% (from baseline) | HIGH |
| **MAE Drift** | Mean absolute error vs predictions | >3 pts sustained | MEDIUM |
| **Sharpe Decay** | Risk-adjusted return trend | Declining 3 periods | MEDIUM |
| **Kelly Efficiency** | Actual vs theoretical Kelly growth | <0.8 efficiency | LOW |

**Algorithm: CLV Decay Detection**

```python
class PerformanceMonitor:
    """
    SOUL.md "Alpha Decay Detection" implementation.
    """
    
    def check_clv_decay(self, window_days: int = 7) -> DecayReport:
        """
        Compare recent CLV vs historical baseline.
        
        Returns DecayReport with:
        - baseline_clv: Historical average (90-day)
        - recent_clv: Last window_days average
        - decay_pct: (recent - baseline) / baseline
        - is_significant: bool (decay_pct > threshold)
        - recommended_action: str
        """
        # Query recent bets
        recent = self.db.query(
            """SELECT AVG(clv) FROM bet_log 
               WHERE placed_at > NOW() - INTERVAL '{window} days'
               AND status = 'settled'"""
        )
        
        # Query baseline
        baseline = self.db.query(
            """SELECT AVG(clv) FROM bet_log 
               WHERE placed_at BETWEEN NOW() - INTERVAL '90 days' 
               AND NOW() - INTERVAL '{window} days'"""
        )
        
        decay_pct = (recent - baseline) / baseline
        
        if decay_pct < -0.15:  # 15% degradation
            return DecayReport(
                severity="CRITICAL",
                message=f"CLV decay: {decay_pct:.1%}. Model edge degraded.",
                recommended_action="TRIGGER_RECALIBRATION"
            )
        elif decay_pct < -0.08:  # 8% degradation
            return DecayReport(
                severity="WARNING",
                message=f"CLV trend down: {decay_pct:.1%}",
                recommended_action="INVESTIGATE"
            )
```

**Database Schema:**

```sql
-- New table for time-series performance metrics
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_date TIMESTAMP NOT NULL,
    metric_type VARCHAR(50) NOT NULL,  -- 'clv', 'win_rate', 'mae', 'sharpe'
    window_days INTEGER NOT NULL,       -- 1, 7, 30, 90
    value FLOAT NOT NULL,
    baseline_value FLOAT,
    decay_pct FLOAT,
    is_significant BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_perf_metrics_date ON performance_metrics(metric_date DESC);
CREATE INDEX idx_perf_metrics_type ON performance_metrics(metric_type, window_days);
```

---

### 2.2 Pattern Detector Agent

**Purpose:** SOUL.md "Structural Vulnerability Assessment" — find patterns in losses that indicate model weaknesses.

**Triggers:** Daily at 4 AM ET (after bet settlement)

**Inputs:**
- `bet_log` (outcomes)
- `predictions` (model inputs/outputs)
- `games` (context: conference, total, seed, etc.)

**Outputs:**
- Vulnerability reports
- Clustering analysis
- Feature drift detection

**Analysis Dimensions:**

| Dimension | Analysis | Example Finding |
|-----------|----------|-----------------|
| **Conference** | Loss rate by conference | "SEC losses 15% above baseline" |
| **Game Total** | Performance by over/under | "Struggle with totals >155" |
| **Seed** | Tournament seed performance | "Underperform as 5-seed fav" |
| **HCA** | Home court advantage errors | "HCA undervalued in Big East" |
| **Back-to-Back** | Fatigue impact | "0-3 on B2B spot bets" |
| **Rest Days** | Rest advantage | "Poor read on 4+ rest teams" |
| **Line Movement** | Opening vs closing | "Fade when line moves >3pts" |

**Algorithm: Loss Clustering**

```python
class PatternDetector:
    """
    SOUL.md "Structural Vulnerability Assessment"
    """
    
    def analyze_loss_patterns(self, days: int = 7) -> List[Vulnerability]:
        """
        Cluster recent losses to find patterns.
        """
        losses = self.get_recent_losses(days)
        
        vulnerabilities = []
        
        # Check each dimension
        for dimension in ['conference', 'total', 'seed', 'hca', 'rest']:
            distribution = self.analyze_by_dimension(losses, dimension)
            
            # Statistical test: is loss rate significantly higher?
            baseline_rate = self.get_baseline_loss_rate(dimension)
            
            for category, rate in distribution.items():
                if rate > baseline_rate * 1.5:  # 50% worse than baseline
                    vulnerabilities.append(Vulnerability(
                        dimension=dimension,
                        category=category,
                        loss_rate=rate,
                        baseline_rate=baseline_rate,
                        severity="HIGH" if rate > baseline_rate * 2 else "MEDIUM",
                        sample_size=len(losses)
                    ))
        
        return vulnerabilities
    
    def detect_feature_drift(self) -> DriftReport:
        """
        Detect if input features are drifting from training distribution.
        """
        # Compare recent game features vs historical
        recent_features = self.get_feature_distribution(days=30)
        historical_features = self.get_feature_distribution(days=365)
        
        drift_scores = {}
        for feature in ['kenpom_adj_eff', 'barttorvik_adj_eff', 'pace']:
            drift = self.calculate_kl_divergence(
                recent_features[feature],
                historical_features[feature]
            )
            drift_scores[feature] = drift
        
        significant_drift = {
            f: score for f, score in drift_scores.items() 
            if score > 0.1  # KL divergence threshold
        }
        
        if significant_drift:
            return DriftReport(
                features=list(significant_drift.keys()),
                drift_scores=significant_drift,
                recommended_action="RETRAIN_MODEL"
            )
```

---

### 2.3 Learning Engine

**Purpose:** SOUL.md "Roadmap Evolution" — generate concrete improvement proposals based on findings.

**Triggers:**
- Weekly (comprehensive analysis)
- On-demand (when Performance Monitor or Pattern Detector flags issues)

**Inputs:**
- Performance reports
- Pattern analysis
- Historical backtest results

**Outputs:**
- Ranked improvement proposals
- Expected ROI estimates
- Implementation plans

**Proposal Generation Logic:**

```python
class LearningEngine:
    """
    SOUL.md "Roadmap Evolution"
    """
    
    def generate_proposals(
        self, 
        performance_report: DecayReport,
        vulnerabilities: List[Vulnerability]
    ) -> List[ImprovementProposal]:
        """
        Generate concrete improvement proposals.
        """
        proposals = []
        
        # Proposal 1: If CLV decay detected
        if performance_report.severity == "CRITICAL":
            proposals.append(ImprovementProposal(
                title="Emergency Recalibration",
                description="Full model recalibration due to CLV decay",
                expected_roi="+15-25% CLV recovery",
                effort="HIGH",
                confidence=0.9,
                auto_implement=True,  # Post-Apr 7
                rollback_plan="Restore previous model weights"
            ))
        
        # Proposal 2: If conference-specific vulnerability
        for vuln in vulnerabilities:
            if vuln.dimension == "conference":
                proposals.append(ImprovementProposal(
                    title=f"Conference Weight Adjustment: {vuln.category}",
                    description=f"Reduce weight on {vuln.category} ratings by 20%",
                    expected_roi=f"+{vuln.loss_rate - vuln.baseline_rate:.1%} win rate",
                    effort="LOW",
                    confidence=0.7,
                    auto_implement=False,  # Requires review
                    rollback_plan="Revert weight to previous value"
                ))
        
        # Proposal 3: If feature drift detected
        if feature_drift := self.check_feature_drift():
            proposals.append(ImprovementProposal(
                title="Model Retraining — Feature Drift",
                description=f"Drift detected in: {', '.join(feature_drift.features)}",
                expected_roi="+10-20% accuracy recovery",
                effort="HIGH",
                confidence=0.8,
                auto_implement=False,
                rollback_plan="Keep previous model version"
            ))
        
        # Rank by expected ROI / effort ratio
        return sorted(proposals, key=lambda p: p.roi_per_effort, reverse=True)
    
    def design_ab_test(self, proposal: ImprovementProposal) -> ABTestSpec:
        """
        Design A/B test for proposed improvement.
        """
        return ABTestSpec(
            name=f"test_{proposal.title.lower().replace(' ', '_')}",
            control_group_pct=50,
            treatment_group_pct=50,
            duration_days=14,
            success_metric="clv_capture_rate",
            min_detectable_effect=0.05,
            stopping_criteria=[
                "statistical_significance_p < 0.05",
                "treatment_clv > control_clv + 2%"
            ]
        )
```

---

### 2.4 Roadmap Maintainer

**Purpose:** Maintain a living, auto-prioritized improvement roadmap.

**Triggers:** Weekly (Monday 6 AM)

**Inputs:**
- Learning Engine proposals
- A/B test results
- Historical proposal outcomes

**Outputs:**
- Updated `ROADMAP.md`
- Prioritized task queue
- Implementation schedule

**Auto-Update Logic:**

```python
class RoadmapMaintainer:
    """
    Auto-maintains ROADMAP.md with findings and priorities.
    """
    
    def update_roadmap(self, proposals: List[ImprovementProposal]):
        """
        Append new proposals and re-prioritize.
        """
        current_roadmap = self.load_roadmap()
        
        # Add new proposals
        for proposal in proposals:
            current_roadmap.add_proposal(proposal)
        
        # Update priorities based on:
        # 1. Performance decay severity
        # 2. Expected ROI
        # 3. Implementation effort
        # 4. Confidence level
        current_roadmap.prioritize(
            weights={
                'urgency': 0.4,
                'expected_roi': 0.3,
                'confidence': 0.2,
                'effort': 0.1  # Lower effort = higher priority
            }
        )
        
        # Write updated roadmap
        self.save_roadmap(current_roadmap)
        
        # Notify Discord
        self.notify_top_priorities(current_roadmap.top_n(5))
    
    def format_roadmark_md(self, roadmap: Roadmap) -> str:
        """
        Generate markdown for ROADMAP.md
        """
        return f"""# OpenClaw Auto-Generated Roadmap
> Last Updated: {datetime.now().isoformat()}  
> Next Review: {(datetime.now() + timedelta(days=7)).isoformat()}

## Current Priorities (Auto-Ranked)

{self._format_priority_list(roadmap.prioritized_items)}

## Recently Implemented

{self._format_completed(roadmap.completed_last_30d)}

## A/B Tests In Progress

{self._format_ab_tests(roadmap.active_tests)}

---
*Generated by OpenClaw Roadmap Maintainer Agent*
"""
```

---

### 2.5 Self-Improvement Agent (Post-Apr 7)

**Purpose:** SOUL.md "Autonomous Implementation" — safely apply approved improvements without human intervention.

**Guardian Compliance:**
- **Until Apr 7, 2026:** DISABLED — proposals only
- **Post-Apr 7, 2026:** ENABLED with safety constraints

**Safety Architecture:**

```python
class SelfImprovementAgent:
    """
    SOUL.md "Autonomous Implementation"
    Only active post-Guardian (Apr 7, 2026).
    """
    
    SAFETY_CONSTRAINTS = {
        'max_daily_changes': 3,
        'max_risk_exposure': 0.05,  # 5% of bankroll
        'require_ab_test': True,    # All changes need A/B validation
        'auto_rollback': True,      # Auto-revert if degradation detected
        'human_approval_for_major': True  # >20% weight changes need approval
    }
    
    def implement_proposal(self, proposal: ImprovementProposal) -> ImplementationResult:
        """
        Safely implement an approved proposal.
        """
        # Safety check 1: Daily change limit
        if self.daily_changes >= self.SAFETY_CONSTRAINTS['max_daily_changes']:
            return ImplementationResult(
                status="DEFERRED",
                reason="Daily change limit reached"
            )
        
        # Safety check 2: A/B test first (if required)
        if self.SAFETY_CONSTRAINTS['require_ab_test']:
            ab_test = self.learning_engine.design_ab_test(proposal)
            self.start_ab_test(ab_test)
            return ImplementationResult(
                status="AB_TEST_STARTED",
                test_id=ab_test.id,
                estimated_completion=ab_test.end_date
            )
        
        # Safety check 3: Create rollback point
        rollback_id = self.create_rollback_point()
        
        try:
            # Apply change
            result = self.apply_change(proposal)
            
            # Verify no immediate degradation
            if self.detect_immediate_issues():
                self.rollback(rollback_id)
                return ImplementationResult(
                    status="ROLLED_BACK",
                    reason="Immediate degradation detected"
                )
            
            self.daily_changes += 1
            return ImplementationResult(
                status="IMPLEMENTED",
                rollback_id=rollback_id,
                monitoring_duration_hours=48
            )
            
        except Exception as e:
            self.rollback(rollback_id)
            return ImplementationResult(
                status="FAILED",
                error=str(e)
            )
    
    def monitor_implementation(self, impl_id: str):
        """
        Monitor recently implemented changes for degradation.
        """
        impl = self.get_implementation(impl_id)
        
        # Check performance at 6h, 24h, 48h
        for check_hour in [6, 24, 48]:
            performance = self.get_performance_since(impl.timestamp)
            
            if performance.clv_decay > 0.10:  # 10% worse
                self.rollback(impl.rollback_id)
                self.alert_rollback(impl, performance)
                return
```

---

## 3. Database Schema

### 3.1 Performance Metrics (Time-Series)

```sql
-- Time-series performance tracking
CREATE TABLE openclaw_performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_timestamp TIMESTAMP NOT NULL,
    metric_type VARCHAR(50) NOT NULL CHECK (metric_type IN (
        'clv_capture_rate',
        'win_rate', 
        'mae',
        'sharpe_ratio',
        'kelly_efficiency',
        'prediction_accuracy'
    )),
    window_hours INTEGER NOT NULL,  -- 48, 168 (7d), 720 (30d), 2160 (90d)
    value FLOAT NOT NULL,
    baseline_value FLOAT,
    decay_pct FLOAT,
    is_significant BOOLEAN DEFAULT FALSE,
    metadata JSONB,  -- Additional context
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_openclaw_perf_time ON openclaw_performance_metrics(metric_timestamp DESC);
CREATE INDEX idx_openclaw_perf_type ON openclaw_performance_metrics(metric_type, window_hours);
```

### 3.2 Vulnerability Detections

```sql
-- Pattern detector findings
CREATE TABLE openclaw_vulnerabilities (
    id SERIAL PRIMARY KEY,
    detected_at TIMESTAMP DEFAULT NOW(),
    dimension VARCHAR(50) NOT NULL,  -- 'conference', 'total', 'seed', etc.
    category VARCHAR(100) NOT NULL,  -- 'SEC', 'high_total', '5_seed', etc.
    loss_rate FLOAT NOT NULL,
    baseline_loss_rate FLOAT NOT NULL,
    severity VARCHAR(20) CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    sample_size INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'MITIGATED', 'FALSE_POSITIVE')),
    mitigation_proposal_id INTEGER,
    resolved_at TIMESTAMP,
    resolution_notes TEXT
);

CREATE INDEX idx_openclaw_vuln_status ON openclaw_vulnerabilities(status);
CREATE INDEX idx_openclaw_vuln_severity ON openclaw_vulnerabilities(severity);
```

### 3.3 Improvement Proposals

```sql
-- Learning Engine proposals
CREATE TABLE openclaw_proposals (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    expected_roi_pct FLOAT,
    effort_level VARCHAR(20) CHECK (effort_level IN ('LOW', 'MEDIUM', 'HIGH')),
    confidence_score FLOAT CHECK (confidence_score BETWEEN 0 AND 1),
    triggered_by VARCHAR(50),  -- 'performance_decay', 'pattern_detection', 'feature_drift'
    auto_implement_eligible BOOLEAN DEFAULT FALSE,
    status VARCHAR(20) DEFAULT 'PROPOSED' CHECK (status IN (
        'PROPOSED', 'APPROVED', 'REJECTED', 'IN_AB_TEST', 
        'IMPLEMENTED', 'ROLLED_BACK', 'COMPLETED'
    )),
    ab_test_id INTEGER,
    implemented_at TIMESTAMP,
    implemented_by VARCHAR(50),  -- 'openclaw_auto' or human username
    rollback_id VARCHAR(100),
    actual_roi_pct FLOAT,
    outcome_notes TEXT
);

CREATE INDEX idx_openclaw_proposals_status ON openclaw_proposals(status);
CREATE INDEX idx_openclaw_proposals_priority ON openclaw_proposals(
    expected_roi_pct DESC, 
    confidence_score DESC
);
```

### 3.4 A/B Tests

```sql
-- A/B test tracking
CREATE TABLE openclaw_ab_tests (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    proposal_id INTEGER REFERENCES openclaw_proposals(id),
    started_at TIMESTAMP DEFAULT NOW(),
    ended_at TIMESTAMP,
    control_group_pct INTEGER DEFAULT 50,
    treatment_group_pct INTEGER DEFAULT 50,
    success_metric VARCHAR(100) NOT NULL,
    min_detectable_effect FLOAT,
    control_clv FLOAT,
    treatment_clv FLOAT,
    p_value FLOAT,
    is_significant BOOLEAN DEFAULT FALSE,
    winner VARCHAR(20) CHECK (winner IN ('CONTROL', 'TREATMENT', 'INCONCLUSIVE')),
    status VARCHAR(20) DEFAULT 'RUNNING' CHECK (status IN ('RUNNING', 'COMPLETED', 'STOPPED'))
);
```

---

## 4. Orchestrator Integration

### 4.1 APScheduler Jobs

```python
# backend/services/openclaw_orchestrator.py

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

class OpenClawOrchestrator:
    """
    Central coordinator for all OpenClaw autonomous agents.
    Integrates with DailyIngestionOrchestrator (EPIC-2).
    """
    
    def register_jobs(self, scheduler: AsyncIOScheduler):
        """
        Register all OpenClaw jobs with scheduler.
        """
        # Performance monitoring — every 2 hours
        scheduler.add_job(
            self.run_performance_monitor,
            trigger=CronTrigger(minute=0, hour='*/2'),  :00, 2:00, 4:00, etc.
            id='openclaw_performance_monitor',
            name='OpenClaw: Performance Monitor',
            replace_existing=True
        )
        
        # Pattern analysis — daily at 4 AM
        scheduler.add_job(
            self.run_pattern_detector,
            trigger=CronTrigger(hour=4, minute=0),
            id='openclaw_pattern_detector',
            name='OpenClaw: Pattern Analysis',
            replace_existing=True
        )
        
        # Roadmap update — weekly Monday 6 AM
        scheduler.add_job(
            self.run_roadmap_maintainer,
            trigger=CronTrigger(day_of_week='mon', hour=6, minute=0),
            id='openclaw_roadmap_update',
            name='OpenClaw: Roadmap Update',
            replace_existing=True
        )
        
        # Self-improvement (Post-Apr 7) — daily at 3 AM
        if datetime.now() > GUARDIAN_LIFT_DATE:
            scheduler.add_job(
                self.run_self_improvement,
                trigger=CronTrigger(hour=3, minute=0),
                id='openclaw_self_improvement',
                name='OpenClaw: Self-Improvement',
                replace_existing=True
            )
```

### 4.2 Advisory Lock IDs (ADR-001)

```python
LOCK_IDS = {
    'openclaw_performance': 200_001,
    'openclaw_pattern': 200_002,
    'openclaw_roadmap': 200_003,
    'openclaw_learning': 200_004,
    'openclaw_improvement': 200_005,  # Post-Apr 7
}
```

---

## 5. Discord Integration

### 5.1 Alert Types

| Alert | Trigger | Channel | Priority |
|-------|---------|---------|----------|
| **Alpha Decay** | CLV drop >15% | #openclaw-escalations | 5 (Critical) |
| **Pattern Found** | New vulnerability | #openclaw-briefs | 3 |
| **Proposal Ready** | Top 5 roadmap update | #openclaw-briefs | 2 |
| **A/B Test Complete** | Statistical significance | #openclaw-briefs | 2 |
| **Auto-Implemented** | Change applied | #openclaw-health | 1 |
| **Rollback** | Degradation detected | #openclaw-escalations | 5 |

### 5.2 Daily Health Summary

```markdown
🤖 **OpenClaw Daily Health — {date}**

📊 Performance (7d):
• CLV Capture: {clv_7d:.1%} ({clv_delta:+.1%} vs baseline)
• Win Rate: {win_rate:.1%}
• Sharpe: {sharpe:.2f}

🔍 Patterns Detected: {vulnerability_count}
{top_vulnerability}

📝 Proposals Generated: {proposal_count}
• Top: {top_proposal_title} (ROI: {top_proposal_roi})

✅ A/B Tests Active: {active_tests}
• {test_name}: {test_status}

Status: {overall_status}
```

---

## 6. Implementation Phases

### Phase 1: Foundation (Waiver Wire Parallel)
**Duration:** 1 week  
**Owner:** Kimi CLI (design) → Claude Code (implementation)

- [ ] Database schema (performance_metrics, vulnerabilities, proposals, ab_tests)
- [ ] PerformanceMonitor agent core
- [ ] PatternDetector agent core  
- [ ] Basic Discord alerting
- [ ] Unit tests (80% coverage)

### Phase 2: Intelligence (Post-Waiver)
**Duration:** 2 weeks

- [ ] LearningEngine proposal generation
- [ ] RoadmapMaintainer auto-updates
- [ ] A/B test framework
- [ ] Feature drift detection
- [ ] Historical pattern analysis

### Phase 3: Autonomy (Apr 1-7)
**Duration:** 1 week
- [ ] SelfImprovementAgent (disabled mode)
- [ ] Rollback mechanism
- [ ] Safety constraint validation
- [ ] Guardian lift preparation

### Phase 4: Activation (Post-Apr 7)
**Duration:** Ongoing
- [ ] Enable self-improvement
- [ ] Full A/B test automation
- [ ] Continuous learning loop
- [ ] Weekly autonomy reports

---

## 7. Success Metrics & KPIs

| Metric | Baseline (v3.0) | Target (v4.0) | Measurement |
|--------|-----------------|---------------|-------------|
| **Alpha Detection Latency** | Manual (days) | <4 hours | Time from degradation → alert |
| **Pattern Discovery Speed** | Manual (weekly) | Automated (daily) | Vulnerabilities found per week |
| **Improvement Proposal Quality** | N/A | >70% accepted | Ratio approved vs rejected |
| **Implementation Cycle Time** | Manual (weeks) | Automated (days) | Proposal → deployment time |
| **False Positive Rate** | N/A | <20% | Rolled back changes / total changes |
| **Human Oversight Reduction** | 100% manual | 80% automated | % of changes requiring review |
| **CLV Recovery Speed** | Days/weeks | <24 hours | Time to recover from edge decay |

---

## 8. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **False pattern detection** | Medium | Medium | Require statistical significance (p < 0.05), min sample size 30 |
| **Overfitting to recent data** | Medium | High | Use multiple time windows, require trend confirmation |
| **Autonomous change causes losses** | Low | Critical | Circuit breakers, auto-rollback, max daily change limits |
| **Alert fatigue** | Medium | Medium | Tiered alerting, consolidate related issues, digest mode |
| **Database bloat** | Low | Low | 90-day retention policy for metrics, archive old data |

---

## 9. Handoff to Claude Code

### Immediate Implementation Tasks (Claude)

1. **Database Migration** (`scripts/migrate_openclaw_v4.py`)
   - Create 4 new tables
   - Add indexes
   - Backfill initial data

2. **PerformanceMonitor** (`backend/services/openclaw/performance_monitor.py`)
   - CLV decay detection
   - Win rate tracking
   - Database integration

3. **PatternDetector** (`backend/services/openclaw/pattern_detector.py`)
   - Loss clustering
   - Conference/total/seed analysis
   - Vulnerability scoring

4. **Orchestrator Integration** (`backend/services/openclaw/__init__.py`)
   - Job registration
   - Advisory locks
   - Error handling

### Next Operator (Kimi CLI)

After Claude completes Phase 1:
- Review implementation for architecture alignment
- Design LearningEngine and RoadmapMaintainer
- Specify A/B test framework
- Prepare Phase 2 specification

---

**Document Version:** OAS-v4.0-DRAFT  
**Last Updated:** March 24, 2026  
**Status:** Design Complete — Ready for Phase 1 Implementation  
**Next Review:** After Phase 1 completion

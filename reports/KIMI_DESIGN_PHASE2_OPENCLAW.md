# Kimi CLI Design: OpenClaw Phase 2 — Learning Engine & Roadmap Maintainer

> **Owner:** Kimi CLI (Deep Intelligence Unit)  
> **Status:** DESIGN PHASE — Ready for implementation after Claude completes Phase 1  
> **Prerequisite:** Phase 1 (Performance Monitor, Pattern Detector) operational  
> **Timeline:** Implementation Mar 31-Apr 7

---

## 1. Learning Engine — Deep Design

### 1.1 Core Purpose

Transform raw performance metrics and detected patterns into **actionable improvement proposals** with:
- Expected ROI quantification
- Confidence scoring
- Implementation effort estimation
- A/B test design

### 1.2 Input/Output Contract

```python
@dataclass
class LearningEngineInput:
    """Data consumed by Learning Engine."""
    performance_metrics: List[PerformanceMetric]  # From Phase 1
    vulnerabilities: List[Vulnerability]          # From Phase 1 Pattern Detector
    historical_outcomes: List[BetOutcome]         # From bet_log table
    ab_test_results: List[ABTestResult]          # From completed experiments
    model_params: Dict[str, float]               # Current weights/configs

@dataclass  
class ImprovementProposal:
    """Output: Actionable improvement recommendation."""
    id: str
    title: str
    description: str
    category: str  # 'weight_adjustment', 'recalibration', 'feature_engineering', etc.
    
    # Impact estimation
    expected_roi_bps: int  # Basis points (0.01 = +1% win rate)
    confidence_score: float  # 0.0-1.0
    effort_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    
    # Implementation
    implementation_plan: str
    rollback_plan: str
    affected_files: List[str]
    
    # A/B testing
    ab_test_spec: Optional[ABTestSpec]
    
    # Metadata
    triggered_by: str  # Which agent found this
    created_at: datetime
    expires_at: datetime  # Proposals stale after 30 days
```

### 1.3 Proposal Categories

| Category | Trigger | Example | Auto-Implement? |
|----------|---------|---------|-----------------|
| **Weight Adjustment** | Pattern showing systematic bias | "Reduce KenPom weight 10% for SEC games" | Post-Apr 7 |
| **Recalibration** | CLV decay >15% | "Full model recalibration" | Post-Apr 7 |
| **Feature Engineering** | Feature drift detected | "Add rest-days feature" | Manual review |
| **Threshold Tuning** | Edge detection suboptimal | "Lower MIN_EDGE from 0.035 to 0.030" | Post-Apr 7 |
| **New Data Source** | Systematic info gap | "Incorporate bullpen ERA" | Manual review |

### 1.4 ROI Estimation Algorithm

```python
def estimate_roi(self, proposal: ProposalConcept) -> ROIEstimate:
    """
    Estimate expected return on improvement.
    Uses historical data and simulation.
    """
    
    if proposal.category == 'weight_adjustment':
        # Simulate with adjusted weights on historical data
        backtest = self.simulate_weight_change(
            param=proposal.target_param,
            new_value=proposal.proposed_value,
            window_days=90
        )
        return ROIEstimate(
            expected_clv_change=backtest.clv_delta,
            win_rate_change=backtest.win_rate_delta,
            confidence=self.calculate_confidence(backtest),
            sample_size=backtest.n_games
        )
    
    elif proposal.category == 'recalibration':
        # Compare to last recalibration impact
        last_recal = self.get_last_recalibration_outcome()
        return ROIEstimate(
            expected_clv_change=last_recal.clv_recovery,
            win_rate_change=last_recal.win_rate_recovery,
            confidence=0.9 if self.clv_decay > 0.15 else 0.6,
            sample_size=None  # Full retrain
        )
```

### 1.5 Confidence Scoring

```python
def calculate_confidence(self, evidence: Evidence) -> float:
    """
    Multi-factor confidence score.
    """
    factors = {
        'sample_size': min(evidence.n_samples / 100, 1.0),  # Need 100+ samples
        'statistical_significance': 1.0 - evidence.p_value,
        'historical_precedent': 0.8 if evidence.similar_fix_worked else 0.5,
        'time_consistency': 1.0 if evidence.trend_persistent else 0.6,
        'cross_validation': evidence.cv_score
    }
    
    # Weighted average
    weights = {
        'sample_size': 0.25,
        'statistical_significance': 0.30,
        'historical_precedent': 0.20,
        'time_consistency': 0.15,
        'cross_validation': 0.10
    }
    
    confidence = sum(factors[k] * weights[k] for k in factors)
    return round(confidence, 2)
```

### 1.6 Proposal Generation Triggers

**Scheduled (Weekly):**
- Comprehensive analysis every Monday 6 AM
- Review all metrics, patterns, A/B tests
- Generate comprehensive proposal set

**Event-Driven (Real-time):**
| Event | Response Time | Action |
|-------|---------------|--------|
| CLV decay >15% | <1 hour | Emergency recalibration proposal |
| Pattern detected | <4 hours | Targeted fix proposal |
| A/B test completes | <1 hour | Implementation/rollback recommendation |
| New vulnerability | <24 hours | Mitigation proposal |

---

## 2. Roadmap Maintainer — Deep Design

### 2.1 Core Purpose

Maintain a **living, auto-prioritized improvement roadmap** that:
- Auto-updates with new proposals
- Tracks implementation status
- Prioritizes by ROI/effort ratio
- Archives completed items
- Identifies stale proposals

### 2.2 Auto-Update Workflow

```python
class RoadmapMaintainer:
    """
    Automated roadmap maintenance.
    """
    
    def weekly_update(self):
        """
        Monday 6 AM job.
        """
        # 1. Fetch new proposals from Learning Engine
        new_proposals = self.learning_engine.generate_proposals()
        
        # 2. Load current roadmap
        roadmap = self.load_roadmap()
        
        # 3. Add new proposals
        for proposal in new_proposals:
            roadmap.add(proposal)
        
        # 4. Update existing proposal statuses
        self.sync_implementation_status(roadmap)
        
        # 5. Re-prioritize
        roadmap.prioritize(strategy='roi_per_effort')
        
        # 6. Mark stale proposals
        self.archive_stale(roadmap, max_age_days=30)
        
        # 7. Write updated roadmap
        self.save_roadmap(roadmap)
        
        # 8. Generate summary
        summary = self.generate_summary(roadmap)
        self.discord.notify_roadmap_update(summary)
    
    def sync_implementation_status(self, roadmap: Roadmap):
        """
        Check which proposals have been implemented.
        """
        for item in roadmap.in_progress:
            # Check if code was modified
            if self.check_implementation(item):
                item.status = 'COMPLETED'
                item.completed_at = datetime.now()
                
                # Measure actual vs expected ROI
                actual_roi = self.measure_actual_roi(item)
                item.actual_roi = actual_roi
                item.roi_accuracy = actual_roi / item.expected_roi
    ```

### 2.3 Prioritization Strategies

```python
class PrioritizationStrategy(Enum):
    ROI_ONLY = 'roi_only'  # Maximize expected return
    EFFORT_ONLY = 'effort_only'  # Quick wins first
    ROI_PER_EFFORT = 'roi_per_effort'  # Bang for buck (default)
    RISK_REDUCTION = 'risk_reduction'  # Fix vulnerabilities first
    CONFIDENCE = 'confidence'  # High confidence first

def prioritize(self, roadmap: Roadmap, strategy: PrioritizationStrategy):
    """
    Re-rank proposals by strategy.
    """
    if strategy == PrioritizationStrategy.ROI_PER_EFFORT:
        # Calculate ROI per unit effort
        for item in roadmap.items:
            effort_map = {'LOW': 1, 'MEDIUM': 3, 'HIGH': 9}
            effort_units = effort_map[item.effort_level]
            item.priority_score = item.expected_roi_bps / effort_units
        
        roadmap.items.sort(key=lambda x: x.priority_score, reverse=True)
    
    elif strategy == PrioritizationStrategy.RISK_REDUCTION:
        # Prioritize vulnerabilities
        vulnerabilities = [i for i in roadmap.items if i.category == 'vulnerability_fix']
        vulnerabilities.sort(key=lambda x: x.severity_score, reverse=True)
        roadmap.items = vulnerabilities + [i for i in roadmap.items if i.category != 'vulnerability_fix']
```

### 2.4 ROADMAP.md Auto-Format

```markdown
<!-- Auto-generated by OpenClaw Roadmap Maintainer -->
<!-- Last Updated: 2026-03-31 06:00:00 UTC -->
<!-- Next Update: 2026-04-07 06:00:00 UTC -->

# OpenClaw Auto-Generated Roadmap

## Current Priorities (Top 5)

### #1: Reduce KenPom Weight for SEC Games
- **Expected ROI:** +150 bps (+1.5% win rate)
- **Effort:** LOW (1 day)
- **Confidence:** 0.87
- **Category:** Weight Adjustment
- **Triggered by:** Pattern Detector (SEC losses 18% above baseline)
- **Status:** PROPOSED
- **A/B Test:** Recommended

[Details...]

### #2: Emergency Recalibration
- **Expected ROI:** +250 bps (+2.5% win rate)
- **Effort:** HIGH (3 days)
- **Confidence:** 0.92
- **Category:** Recalibration
- **Triggered by:** Performance Monitor (CLV decay 17%)
- **Status:** PROPOSED
- **A/B Test:** Not applicable

[Details...]

## Recently Completed

### ✅ Lower MIN_EDGE Threshold (Mar 28)
- **Expected ROI:** +80 bps
- **Actual ROI:** +95 bps
- **ROI Accuracy:** 119% (better than expected)

## A/B Tests In Progress

| Test | Started | Expected Completion | Status |
|------|---------|---------------------|--------|
| Adjust HCA for neutral sites | Mar 25 | Apr 8 | RUNNING |

## Stale Proposals (Auto-Archived)

- "Add March Madness experience factor" (Created Feb 15, archived Mar 20)

---
*This roadmap is automatically maintained by OpenClaw v4.0*
*Manual edits will be overwritten*
```

### 2.5 Integration with GitHub Issues (Optional)

For transparency, Roadmap Maintainer can sync to GitHub:

```python
def sync_to_github(self, roadmap: Roadmap):
    """
    Create GitHub issues for top proposals.
    """
    for item in roadmap.top_n(10):
        if not item.github_issue_id:
            issue = self.github.create_issue(
                title=f"[OpenClaw] {item.title}",
                body=self.format_issue_body(item),
                labels=['openclaw', item.category, item.effort_level.lower()],
                assignee=self.recommend_assignee(item)
            )
            item.github_issue_id = issue.number
```

---

## 3. A/B Test Framework Design

### 3.1 Test Lifecycle

```
PROPOSED → DESIGN → RUNNING → ANALYZING → [COMPLETED | ROLLED_BACK]
                ↓
            STOPPED (early termination)
```

### 3.2 Early Stopping Rules

```python
class ABTestFramework:
    """
    Manages A/B test lifecycle.
    """
    
    def check_early_stopping(self, test: ABTest) -> Optional[StopDecision]:
        """
        Check if test should stop early.
        """
        # Rule 1: Statistical significance achieved
        if test.p_value < 0.01 and abs(test.effect_size) > test.min_detectable_effect:
            return StopDecision(
                reason="Statistical significance achieved",
                winner='TREATMENT' if test.effect_size > 0 else 'CONTROL',
                confidence=1 - test.p_value
            )
        
        # Rule 2: Harm detected (safety)
        if test.treatment_clv < test.control_clv * 0.9:  # 10% worse
            return StopDecision(
                reason="Harm detected - treatment performing 10% worse",
                winner='CONTROL',
                confidence=0.95
            )
        
        # Rule 3: Futility (no chance of significance)
        if test.days_running > 7 and test.p_value > 0.3:
            return StopDecision(
                reason="Futility - no significant effect after 7 days",
                winner='INCONCLUSIVE',
                confidence=0.8
            )
        
        return None  # Continue test
```

### 3.3 Test Types

| Type | Description | Duration | Sample Size |
|------|-------------|----------|-------------|
| **Weight Adjustment** | Test +/- 10% weight change | 7 days | ~50 games |
| **Threshold Tuning** | Test edge threshold changes | 5 days | ~40 games |
| **Feature Addition** | Test new feature impact | 14 days | ~100 games |
| **Full Recalibration** | Test retrained model | 21 days | ~150 games |

---

## 4. Implementation Sequence

### Week 1 (Mar 31-Apr 4): Learning Engine
- Day 1: Core data structures (Proposal, ROIEstimate)
- Day 2: ROI estimation algorithms
- Day 3: Confidence scoring
- Day 4: Proposal generation triggers
- Day 5: Integration testing

### Week 2 (Apr 5-7): Roadmap Maintainer + A/B Framework
- Day 1: Roadmap data model
- Day 2: Prioritization algorithms
- Day 3: ROADMAP.md auto-generation
- Day 4: A/B test framework core
- Day 5: Early stopping + integration
- Day 6-7: End-to-end testing

---

## 5. Success Criteria

| Component | Metric | Target |
|-----------|--------|--------|
| Learning Engine | Proposal quality | >70% accepted by human review |
| Learning Engine | ROI accuracy | Predicted vs actual within 20% |
| Roadmap Maintainer | Update latency | <1 hour after Learning Engine |
| Roadmap Maintainer | Priority stability | Top 3 items stable for 3+ days |
| A/B Framework | Test completion rate | >90% reach conclusion |
| A/B Framework | Early stopping accuracy | <5% false positives |

---

## 6. Handoff to Claude

### Phase 2 Implementation Tasks

1. **Learning Engine** (`backend/services/openclaw/learning_engine.py`)
   - Proposal generation logic
   - ROI estimation
   - Confidence scoring

2. **Roadmap Maintainer** (`backend/services/openclaw/roadmap_maintainer.py`)
   - Weekly update job
   - Prioritization algorithms
   - ROADMAP.md generation

3. **A/B Test Framework** (`backend/services/openclaw/ab_test_framework.py`)
   - Test lifecycle management
   - Early stopping rules
   - Results analysis

4. **Database** (migration)
   - `ab_tests` table
   - Update `proposals` table with new fields

5. **Scheduler Integration**
   - Weekly Monday 6 AM job
   - Event-driven triggers

---

**Document Version:** KIMI-PHASE2-v1  
**Status:** Design Complete — Ready for Implementation  
**Prerequisite:** Phase 1 (Performance Monitor, Pattern Detector) operational

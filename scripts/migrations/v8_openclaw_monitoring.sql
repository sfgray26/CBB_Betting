--
-- OpenClaw v4.0 Monitoring Schema
-- Migration: Add time-series tables for autonomous model operations
--
-- Phase 1: Performance Monitor + Pattern Detector (read-only during Guardian freeze)
-- Phase 4: Self-improvement activation (post-Apr 7, 2026)
--
-- Tables:
-- - model_performance_metrics: CLV, accuracy, win rate time-series
-- - vulnerability_reports: Detected pattern vulnerabilities
-- - learning_journal: Experimental features and results
-- - roadmap_state: Prioritized improvement tracking
--

-- ============================================
-- 1. MODEL PERFORMANCE METRICS
-- ============================================
-- Time-series storage for all performance indicators
-- Enables decay detection, trend analysis, calibration monitoring

CREATE TABLE IF NOT EXISTS model_performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    sport VARCHAR(10) NOT NULL,              -- 'cbb', 'mlb'
    metric_type VARCHAR(50) NOT NULL,        -- 'accuracy', 'clv', 'win_rate', 'mae', 'roi'
    value DECIMAL(10, 6) NOT NULL,           -- The metric value
    sample_size INTEGER NOT NULL,            -- Number of games/data points
    window_days INTEGER NOT NULL,            -- Analysis window (7, 14, 30, etc.)
    calculated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',             -- Additional context (market breakdown, etc.)
    
    -- Composite indexes for common queries
    CONSTRAINT chk_metric_type CHECK (metric_type IN (
        'accuracy', 'clv', 'win_rate', 'mae', 'roi', 'sharpe', 'calibration'
    ))
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_perf_metrics_sport_type 
    ON model_performance_metrics(sport, metric_type);
CREATE INDEX IF NOT EXISTS idx_perf_metrics_time 
    ON model_performance_metrics(calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_perf_metrics_composite 
    ON model_performance_metrics(sport, metric_type, calculated_at DESC);

-- Partial index for recent data (most queries use last 30 days)
-- CREATE INDEX IF NOT EXISTS idx_perf_metrics_recent 
--    ON model_performance_metrics(sport, metric_type, calculated_at DESC)
--    WHERE calculated_at > NOW() - INTERVAL '30 days';

COMMENT ON TABLE model_performance_metrics IS 
    'Time-series performance metrics for model monitoring and decay detection';

-- ============================================
-- 2. VULNERABILITY REPORTS
-- ============================================
-- Detected pattern vulnerabilities from PatternDetector
-- Tracks systematic issues requiring attention or recalibration

CREATE TABLE IF NOT EXISTS vulnerability_reports (
    id BIGSERIAL PRIMARY KEY,
    sport VARCHAR(10) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,       -- 'conference_bias', 'clv_decay', etc.
    confidence DECIMAL(4, 3) NOT NULL,       -- 0.0-1.0
    severity VARCHAR(20) NOT NULL,           -- 'CRITICAL', 'WARNING', 'INFO'
    description TEXT NOT NULL,
    affected_games INTEGER NOT NULL,
    sample_win_rate DECIMAL(5, 4),           -- Actual win rate in affected sample
    expected_win_rate DECIMAL(5, 4),         -- Model's expected win rate
    edge_impact DECIMAL(5, 4) NOT NULL,      -- Projected edge reduction
    recommended_action TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',             -- Pattern-specific details
    
    -- Lifecycle tracking
    detected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    first_game_at TIMESTAMP WITH TIME ZONE,  -- First affected game
    last_game_at TIMESTAMP WITH TIME ZONE,   -- Most recent affected game
    resolved_at TIMESTAMP WITH TIME ZONE,    -- NULL = still active
    resolution_notes TEXT,
    
    -- Constraints
    CONSTRAINT chk_confidence CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT chk_severity CHECK (severity IN ('CRITICAL', 'WARNING', 'INFO')),
    CONSTRAINT chk_dates CHECK (resolved_at IS NULL OR resolved_at >= detected_at)
);

-- Indexes for active vulnerability queries
CREATE INDEX IF NOT EXISTS idx_vuln_sport_active 
    ON vulnerability_reports(sport, severity) 
    WHERE resolved_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_vuln_pattern 
    ON vulnerability_reports(pattern_type, sport);
CREATE INDEX IF NOT EXISTS idx_vuln_detected 
    ON vulnerability_reports(detected_at DESC);

COMMENT ON TABLE vulnerability_reports IS 
    'Systematic vulnerabilities detected by PatternDetector requiring model attention';

-- ============================================
-- 3. LEARNING JOURNAL
-- ============================================
-- Experimental features and A/B test results
-- Populated post-Apr 7 when self-improvement activates

CREATE TABLE IF NOT EXISTS learning_journal (
    id BIGSERIAL PRIMARY KEY,
    experiment_id VARCHAR(100) NOT NULL UNIQUE,
    experiment_type VARCHAR(50) NOT NULL,    -- 'feature', 'model', 'threshold', 'calibration'
    sport VARCHAR(10) NOT NULL,
    
    -- Experiment description
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    hypothesis TEXT,                         -- Expected outcome
    
    -- Implementation
    changes_summary TEXT NOT NULL,           -- What was changed
    code_version VARCHAR(50),                -- Git commit/tag
    rollback_commit VARCHAR(50),             -- Pre-change commit for rollback
    
    -- Test parameters
    test_start_at TIMESTAMP WITH TIME ZONE,
    test_end_at TIMESTAMP WITH TIME ZONE,
    sample_size INTEGER,
    control_group_size INTEGER,
    
    -- Results (populated after test completion)
    results JSONB,                           -- Detailed metrics
    primary_metric VARCHAR(50),              -- Main KPI measured
    primary_metric_change DECIMAL(10, 6),    -- Delta vs control
    statistical_significance DECIMAL(5, 4),  -- p-value
    
    -- Outcome
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- 'pending', 'running', 'success', 'failed', 'rolled_back'
    outcome_notes TEXT,
    
    -- Lifecycle
    proposed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    approved_at TIMESTAMP WITH TIME ZONE,
    deployed_at TIMESTAMP WITH TIME ZONE,
    concluded_at TIMESTAMP WITH TIME ZONE,
    
    -- Audit
    proposed_by VARCHAR(100),
    approved_by VARCHAR(100),
    
    CONSTRAINT chk_experiment_status CHECK (status IN (
        'pending', 'running', 'success', 'failed', 'rolled_back'
    ))
);

CREATE INDEX IF NOT EXISTS idx_learning_status 
    ON learning_journal(status, sport);
CREATE INDEX IF NOT EXISTS idx_learning_experiment_type 
    ON learning_journal(experiment_type, sport);

COMMENT ON TABLE learning_journal IS 
    'Experimental changes and A/B test results (populated post-Guardian freeze)';

-- ============================================
-- 4. ROADMAP STATE
-- ============================================
-- Prioritized improvement backlog
-- Maintained by RoadmapMaintainer agent

CREATE TABLE IF NOT EXISTS roadmap_state (
    id BIGSERIAL PRIMARY KEY,
    improvement_id VARCHAR(100) NOT NULL UNIQUE,
    sport VARCHAR(10) NOT NULL,
    
    -- Categorization
    category VARCHAR(50) NOT NULL,           -- 'calibration', 'feature', 'data', 'infrastructure'
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    
    -- Priority scoring
    priority_score DECIMAL(5, 2) NOT NULL,   -- 0-100 calculated score
    expected_roi DECIMAL(5, 4),              -- Projected ROI improvement
    implementation_cost INTEGER,             -- 1-10 effort estimate
    risk_level VARCHAR(20),                  -- 'low', 'medium', 'high'
    
    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'proposed',  -- 'proposed', 'approved', 'in_progress', 'completed', 'deferred'
    blocked_by VARCHAR(100)[],               -- Dependencies (other improvement_ids)
    
    -- Attribution
    source VARCHAR(50),                      -- 'pattern_detector', 'performance_monitor', 'manual', 'backtest'
    source_vulnerability_id BIGINT REFERENCES vulnerability_reports(id),
    
    -- Target dates
    target_phase INTEGER,                    -- OpenClaw phase (1-4)
    proposed_for_date DATE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notes TEXT,
    
    CONSTRAINT chk_roadmap_status CHECK (status IN (
        'proposed', 'approved', 'in_progress', 'completed', 'deferred'
    ))
);

CREATE INDEX IF NOT EXISTS idx_roadmap_priority 
    ON roadmap_state(sport, priority_score DESC) 
    WHERE status IN ('proposed', 'approved');
CREATE INDEX IF NOT EXISTS idx_roadmap_status 
    ON roadmap_state(status, sport);

COMMENT ON TABLE roadmap_state IS 
    'Prioritized improvement backlog maintained by RoadmapMaintainer';

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_roadmap_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_roadmap_updated ON roadmap_state;
CREATE TRIGGER trg_roadmap_updated
    BEFORE UPDATE ON roadmap_state
    FOR EACH ROW
    EXECUTE FUNCTION update_roadmap_timestamp();

-- ============================================
-- 5. VIEWS FOR COMMON QUERIES
-- ============================================

-- Active vulnerabilities summary
CREATE OR REPLACE VIEW active_vulnerabilities AS
SELECT 
    sport,
    severity,
    COUNT(*) as count,
    MAX(detected_at) as latest_detection,
    SUM(affected_games) as total_affected_games
FROM vulnerability_reports
WHERE resolved_at IS NULL
GROUP BY sport, severity
ORDER BY 
    CASE severity 
        WHEN 'CRITICAL' THEN 1 
        WHEN 'WARNING' THEN 2 
        ELSE 3 
    END,
    latest_detection DESC;

-- Recent performance trends (7-day rolling)
CREATE OR REPLACE VIEW performance_trends AS
SELECT 
    sport,
    metric_type,
    window_days,
    value,
    sample_size,
    calculated_at,
    LAG(value) OVER (PARTITION BY sport, metric_type ORDER BY calculated_at) as prev_value,
    value - LAG(value) OVER (PARTITION BY sport, metric_type ORDER BY calculated_at) as change
FROM model_performance_metrics
WHERE calculated_at > NOW() - INTERVAL '14 days'
ORDER BY sport, metric_type, calculated_at DESC;

-- ============================================
-- 6. INITIAL DATA POPULATION
-- ============================================
-- Insert baseline records for tracking

-- Guardian freeze marker
INSERT INTO model_performance_metrics (
    sport, metric_type, value, sample_size, window_days, metadata
) VALUES (
    'system',
    'guardian_status',
    1.0,  -- Active
    0,
    0,
    '{"freeze_until": "2026-04-07", "phase": "1", "note": "OpenClaw monitoring initialized"}'
)
ON CONFLICT DO NOTHING;

-- ============================================
-- MIGRATION COMPLETE
-- ============================================
-- 
-- Next steps:
-- 1. Run Performance Monitor: python -m backend.services.openclaw.performance_monitor
-- 2. Run Pattern Detector: python -m backend.services.openclaw.pattern_detector
-- 3. Enable in scheduler: daily_ingestion.py will auto-detect and schedule
--
-- Post-Apr 7 activation:
-- UPDATE model_performance_metrics 
-- SET value = 0.0, metadata = '{"phase": "4", "note": "Self-improvement activated"}'
-- WHERE sport = 'system' AND metric_type = 'guardian_status';

-- Migration: Add auto_stream_config column to UserPreferences
-- Date: 2026-06-25
-- Author: Claude Code (Loop Iteration 17)

-- Add JSONB column for Auto-Stream configuration
ALTER TABLE user_preferences
ADD COLUMN IF NOT EXISTS auto_stream_config JSONB DEFAULT '{"enabled": false, "min_confidence": "HIGH", "min_recommendation": "EXCELLENT", "max_adds_per_week": 2, "drop_priority": [], "updated_at": null}'::jsonb;

-- Add comment
COMMENT ON COLUMN user_preferences.auto_stream_config IS 'Auto-Stream feature configuration: enabled, drop_priority, min_confidence, min_recommendation, max_adds_per_week';

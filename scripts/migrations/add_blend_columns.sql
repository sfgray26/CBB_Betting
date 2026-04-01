-- Migration: Add ensemble blend columns to player_daily_metrics
-- Phase 2.2 — In-Season Projection Pipeline
-- Date: 2026-04-01
-- Author: Claude Code (Master Architect)
--
-- These columns store the weighted RoS projection blend:
--   ATC 30% / THE BAT 30% / Steamer 20% / ZiPS 20%
-- NULL means no blend computed yet for that date.

ALTER TABLE player_daily_metrics
    ADD COLUMN IF NOT EXISTS blend_hr FLOAT,
    ADD COLUMN IF NOT EXISTS blend_rbi FLOAT,
    ADD COLUMN IF NOT EXISTS blend_avg FLOAT,
    ADD COLUMN IF NOT EXISTS blend_era FLOAT,
    ADD COLUMN IF NOT EXISTS blend_whip FLOAT;

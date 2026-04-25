-- Migration: Create ingested_injuries table
-- Phase 1 — BDL GOAT Foundation (Injury Integration)
-- Date: 2026-04-25
-- Author: Claude Code (Master Architect)
--
-- Stores MLB injury reports from BDL /mlb/v1/player_injuries.
-- Natural key: (bdl_player_id, injury_status, injury_type).
-- Active injuries are upserted hourly; inactive injuries expire when no longer
-- returned by BDL (cleanup job handles deletion).

CREATE TABLE IF NOT EXISTS ingested_injuries (
    id BIGSERIAL PRIMARY KEY,
    bdl_player_id INTEGER NOT NULL,
    player_name VARCHAR(150) NOT NULL,

    -- Injury details from BDL MLBInjury contract
    injury_date TIMESTAMP WITH TIME ZONE,
    return_date TIMESTAMP WITH TIME ZONE,
    injury_type VARCHAR(100) NOT NULL,
    injury_detail VARCHAR(100),
    injury_side VARCHAR(10),
    injury_status VARCHAR(20) NOT NULL,

    -- Narrative fields
    long_comment TEXT NOT NULL,
    short_comment VARCHAR(500) NOT NULL,

    -- Audit columns
    raw_payload JSONB NOT NULL,
    ingested_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Unique constraint: one injury per player, status, and type
-- Allows tracking of multi-injury players (e.g., same player with two DTD entries)
CREATE UNIQUE INDEX IF NOT EXISTS _ii_player_status_type_uc
    ON ingested_injuries (bdl_player_id, injury_status, injury_type);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ingested_injuries_active
    ON ingested_injuries (ingested_at, injury_status);

CREATE INDEX IF NOT EXISTS idx_ingested_injuries_player
    ON ingested_injuries (bdl_player_id, ingested_at);

-- Updated at trigger (optional, but good practice)
CREATE OR REPLACE FUNCTION update_ingested_injuries_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_ingested_injuries_updated_at ON ingested_injuries;
CREATE TRIGGER set_ingested_injuries_updated_at
    BEFORE UPDATE ON ingested_injuries
    FOR EACH ROW
    EXECUTE FUNCTION update_ingested_injuries_updated_at();

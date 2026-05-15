-- Migration: Add auto-heal tracking columns to player_id_mapping
-- Task: BDL #2 - Player Search Auto-Heal
-- Created: 2025-05-15

-- Add heal_attempts column to track auto-heal attempts
ALTER TABLE player_id_mapping
ADD COLUMN IF NOT EXISTS heal_attempts INTEGER NOT NULL DEFAULT 0;

-- Add healed_at column to track when auto-heal succeeded
ALTER TABLE player_id_mapping
ADD COLUMN IF NOT EXISTS healed_at TIMESTAMP WITH TIME ZONE;

-- Add index for querying healed players
CREATE INDEX IF NOT EXISTS idx_pim_healed_at 
ON player_id_mapping(healed_at) 
WHERE healed_at IS NOT NULL;

-- Add index for querying by source (useful for filtering auto-healed rows)
CREATE INDEX IF NOT EXISTS idx_pim_source 
ON player_id_mapping(source);

-- Update existing rows: set healed_at for rows with source='bdl_search' and no healed_at
UPDATE player_id_mapping
SET healed_at = updated_at,
    heal_attempts = 1
WHERE source = 'bdl_search'
  AND healed_at IS NULL
  AND updated_at IS NOT NULL;

-- Verification query
SELECT 
    source,
    COUNT(*) as total,
    COUNT(healed_at) as healed_count,
    AVG(heal_attempts) as avg_attempts
FROM player_id_mapping
GROUP BY source;

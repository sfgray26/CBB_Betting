-- Migration: Add handedness column to probable_pitchers table
-- Date: 2026-05-15
-- Author: Hermes Agent
--
-- This column stores the pitcher's throwing hand: "L" for left-handed, "R" for right-handed.
-- NULL values are allowed for cases where handedness is not yet known or unavailable.

ALTER TABLE probable_pitchers
    ADD COLUMN IF NOT EXISTS handedness VARCHAR(1);

-- Add comment for documentation
COMMENT ON COLUMN probable_pitchers.handedness IS 'Pitcher throwing hand: L (left) or R (right)';

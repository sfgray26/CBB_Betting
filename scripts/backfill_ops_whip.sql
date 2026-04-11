-- Task 26: Backfill ops and whip data
-- This script populates NULL values in mlb_player_stats

-- Backfill ops = obp + slg
UPDATE mlb_player_stats
SET ops = obp + slg
WHERE ops IS NULL
  AND obp IS NOT NULL
  AND slg IS NOT NULL;

-- Backfill whip = (walks_allowed + hits_allowed) / innings_pitched
-- Handles innings_pitched string format "6.2" -> 6.667 decimal
UPDATE mlb_player_stats
SET whip = (walks_allowed + hits_allowed)::numeric /
          NULLIF(
              CAST(SPLIT_PART(innings_pitched, '.', 1) AS NUMERIC) +
              CAST(SPLIT_PART(innings_pitched, '.', 2) AS NUMERIC) / 3.0,
              0
          )
WHERE whip IS NULL
  AND walks_allowed IS NOT NULL
  AND hits_allowed IS NOT NULL
  AND innings_pitched IS NOT NULL
  AND innings_pitched != '';

-- Task 26: Verify backfill results
-- Check remaining NULL counts

SELECT
    'ops' as field_name,
    COUNT(*) as total_rows,
    COUNT(ops) as non_null_count,
    COUNT(*) - COUNT(ops) as null_count,
    ROUND(100.0 * COUNT(ops) / COUNT(*), 2) as fill_percentage
FROM mlb_player_stats

UNION ALL

SELECT
    'whip' as field_name,
    COUNT(*) as total_rows,
    COUNT(whip) as non_null_count,
    COUNT(*) - COUNT(whip) as null_count,
    ROUND(100.0 * COUNT(whip) / COUNT(*), 2) as fill_percentage
FROM mlb_player_stats;

-- Sample backfilled data
SELECT
    player_name,
    obp, slg, ops,
    walks_allowed, hits_allowed, innings_pitched, whip
FROM mlb_player_stats
WHERE ops IS NOT NULL OR whip IS NOT NULL
ORDER BY updated_at DESC
LIMIT 5;

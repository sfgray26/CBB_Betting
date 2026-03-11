-- Check for duplicate games in the database
-- This would cause multiple predictions for the same matchup

-- Find games with same teams on same date (potential duplicates)
SELECT 
    DATE(game_date) as game_day,
    home_team,
    away_team,
    COUNT(*) as game_count,
    STRING_AGG(id::text, ', ') as game_ids
FROM games
WHERE DATE(game_date) >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY DATE(game_date), home_team, away_team
HAVING COUNT(*) > 1
ORDER BY game_day DESC, game_count DESC;

-- Check if external_id is causing duplicates (NULL vs populated)
SELECT 
    CASE WHEN external_id IS NULL THEN 'NULL' ELSE 'Populated' END as id_status,
    COUNT(*) as count
FROM games
WHERE DATE(game_date) >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY CASE WHEN external_id IS NULL THEN 'NULL' ELSE 'Populated' END;

-- Find predictions for same game on same date (should be 1 per run_tier)
SELECT 
    g.home_team,
    g.away_team,
    DATE(g.game_date) as game_date,
    p.run_tier,
    COUNT(*) as prediction_count,
    STRING_AGG(p.id::text, ', ') as prediction_ids
FROM predictions p
JOIN games g ON p.game_id = g.id
WHERE DATE(g.game_date) >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY g.home_team, g.away_team, DATE(g.game_date), p.run_tier
HAVING COUNT(*) > 1
ORDER BY game_date DESC, prediction_count DESC;

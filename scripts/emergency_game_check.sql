-- Emergency Game Check SQL
-- Run this query directly on your Railway Postgres database

-- Today's date (UTC)
SELECT 
    CURRENT_DATE AS today_utc,
    NOW() AS current_time_utc;

-- Game count by date
SELECT 
    DATE(game_date) AS game_day,
    COUNT(*) AS game_count
FROM games
GROUP BY DATE(game_date)
ORDER BY game_day DESC
LIMIT 10;

-- All games for TODAY
SELECT 
    g.id,
    g.away_team,
    g.home_team,
    g.game_date,
    g.data_source,
    p.verdict,
    p.edge_conservative
FROM games g
LEFT JOIN predictions p ON g.id = p.game_id
WHERE DATE(g.game_date) = CURRENT_DATE
ORDER BY g.game_date;

-- Check for Duke vs UNC (SUSPICIOUS if found)
SELECT 
    g.id,
    g.away_team,
    g.home_team,
    g.game_date,
    g.data_source
FROM games g
WHERE 
    (LOWER(g.home_team) LIKE '%duke%' AND LOWER(g.away_team) LIKE '%north carolina%')
    OR (LOWER(g.home_team) LIKE '%north carolina%' AND LOWER(g.away_team) LIKE '%duke%')
    OR (LOWER(g.home_team) LIKE '%duke%' AND LOWER(g.away_team) LIKE '%unc%')
    OR (LOWER(g.home_team) LIKE '%unc%' AND LOWER(g.away_team) LIKE '%duke%');

-- Oldest games (stale data detection)
SELECT 
    g.away_team,
    g.home_team,
    g.game_date
FROM games g
ORDER BY g.game_date ASC
LIMIT 5;

-- Count total games in database
SELECT COUNT(*) AS total_games FROM games;

-- Games with future dates (possible error)
SELECT 
    g.away_team,
    g.home_team,
    g.game_date
FROM games g
WHERE g.game_date > NOW() + INTERVAL '7 days'
ORDER BY g.game_date;

-- Check games mentioned in Discord line alerts to see if they started
-- Run this to see which alerted games have already started

SELECT 
    g.id,
    g.home_team,
    g.away_team,
    g.game_date,
    g.game_date < NOW() as has_started,
    bl.pick,
    bl.outcome
FROM games g
JOIN bet_logs bl ON g.id = bl.game_id
WHERE 
    -- Games from today with pending bets
    DATE(g.game_date) = CURRENT_DATE
    AND bl.outcome IS NULL
ORDER BY g.game_date;

-- Count how many pending bets are on games that already started
SELECT 
    COUNT(*) as started_games_with_pending_bets
FROM games g
JOIN bet_logs bl ON g.id = bl.game_id
WHERE 
    DATE(g.game_date) = CURRENT_DATE
    AND bl.outcome IS NULL
    AND g.game_date < NOW();

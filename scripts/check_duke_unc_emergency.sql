-- 🚨 EMERGENCY: Check for fake/stale Duke vs UNC games

-- Step 1: What ACC games are scheduled for TODAY?
SELECT 
    'TODAY (March 10)' as check_type,
    g.home_team,
    g.away_team,
    g.game_date,
    g.data_source,
    CASE 
        WHEN g.game_date < NOW() - INTERVAL '24 hours' THEN '⚠️ STALE - Game already happened!'
        WHEN g.game_date > NOW() + INTERVAL '48 hours' THEN '⚠️ FUTURE - Wrong date assigned!'
        ELSE '✅ Seems correct'
    END as status
FROM games g
WHERE 
    (LOWER(g.home_team) LIKE '%duke%' OR LOWER(g.away_team) LIKE '%duke%'
     OR LOWER(g.home_team) LIKE '%north carolina%' OR LOWER(g.away_team) LIKE '%north carolina%'
     OR LOWER(g.home_team) LIKE '%unc %' OR LOWER(g.away_team) LIKE '%unc %')
    AND DATE(g.game_date) = CURRENT_DATE;

-- Step 2: ALL Duke vs UNC games in database (ever)
SELECT 
    'ALL TIME' as check_type,
    g.home_team,
    g.away_team,
    g.game_date,
    g.data_source,
    CASE 
        WHEN g.game_date < NOW() - INTERVAL '7 days' THEN '⚠️ STALE DATA - Should be deleted'
        ELSE 'Recent'
    END as status
FROM games g
WHERE 
    ((LOWER(g.home_team) LIKE '%duke%' AND (LOWER(g.away_team) LIKE '%north carolina%' OR LOWER(g.away_team) LIKE '%unc %'))
    OR (LOWER(g.away_team) LIKE '%duke%' AND (LOWER(g.home_team) LIKE '%north carolina%' OR LOWER(g.home_team) LIKE '%unc %')))
ORDER BY g.game_date DESC;

-- Step 3: Check all games with timestamps in the past
SELECT 
    'PAST GAMES WITH TODAY DATE' as check_type,
    COUNT(*) as count
FROM games g
WHERE DATE(g.game_date) = CURRENT_DATE
    AND g.game_date < NOW() - INTERVAL '6 hours';

-- Step 4: List them if any found
SELECT 
    g.home_team,
    g.away_team,
    g.game_date,
    g.data_source
FROM games g
WHERE DATE(g.game_date) = CURRENT_DATE
    AND g.game_date < NOW() - INTERVAL '6 hours'
ORDER BY g.game_date;

-- Step 5: What games does Odds API show for today?
-- (Check if API is returning wrong dates)
SELECT DISTINCT
    data_source,
    COUNT(*) as game_count,
    MIN(game_date) as earliest,
    MAX(game_date) as latest
FROM games
WHERE DATE(game_date) = CURRENT_DATE
GROUP BY data_source;

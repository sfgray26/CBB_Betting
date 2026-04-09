-- Quick table audit for Railway database
-- Copy this into Railway connect psql session

SELECT
    schemaname,
    tablename,
    n_live_tup as "rows",
    CASE
        WHEN n_live_tup = 0 THEN 'EMPTY'
        WHEN n_live_tup < 100 THEN 'SMALL'
        WHEN n_live_tup < 1000 THEN 'MEDIUM'
        ELSE 'LARGE'
    END as "size",
    last_autovacuum,
    last_autoanalyze
FROM pg_stat_user_tables
ORDER BY tablename;

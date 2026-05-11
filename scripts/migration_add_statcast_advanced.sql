-- Add advanced Statcast columns to enable persistence of stuff_plus, location_plus, sprint_speed
-- Mission: Wire up missing Statcast columns end-to-end

ALTER TABLE statcast_batter_metrics ADD COLUMN sprint_speed FLOAT;
ALTER TABLE statcast_pitcher_metrics ADD COLUMN stuff_plus FLOAT;
ALTER TABLE statcast_pitcher_metrics ADD COLUMN location_plus FLOAT;

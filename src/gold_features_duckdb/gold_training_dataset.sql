WITH
results_filtered AS (
  SELECT *
  FROM silver_results
  {where_clause}
),

race_labels AS (
  SELECT
    CAST(event_year AS INTEGER) AS event_year,
    event,
    CAST(driver_number AS INTEGER) AS driver_number,
    CAST(finishing_position AS INTEGER) AS race_finishing_position,
    -- Race time (seconds): total time or gap to winner; NULL for DNF / no time
    time_seconds AS race_time_seconds
  FROM results_filtered
  WHERE session = 'R'
)

SELECT
  f.*,
  rl.race_finishing_position,
  rl.race_time_seconds
FROM gold_driver_event_features f
LEFT JOIN race_labels rl
  ON rl.event_year = f.event_year
 AND rl.event = f.event
 AND rl.driver_number = f.driver_number
ORDER BY f.event_year, f.event, f.driver_number
;


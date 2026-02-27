WITH
results_filtered AS (
  SELECT *
  FROM silver_results
  {where_clause}
),
laps_filtered AS (
  SELECT *
  FROM silver_laps
  {where_clause}
),

-- Conventional qualifying results (Q)
q_results AS (
  SELECT
    CAST(event_year AS INTEGER) AS event_year,
    event,
    event_name,
    session AS session,
    CAST(driver_number AS INTEGER) AS driver_number,
    driver_id,
    full_name,
    abbreviation,
    team_id,
    team_name,
    team_color,
    -- In your silver transform, for qualifying sessions `finishing_position` is the quali position.
    CAST(finishing_position AS INTEGER) AS qualifying_position,
    q1_seconds,
    q2_seconds,
    q3_seconds
  FROM results_filtered
  WHERE session = 'Q'
),

-- Sprint qualifying results (SQ - 2024+ format) or Sprint Shootout (SS - 2023 format)
-- Used for sprint race grid positions
sq_results AS (
  SELECT
    CAST(event_year AS INTEGER) AS event_year,
    event,
    event_name,
    session AS session,
    CAST(driver_number AS INTEGER) AS driver_number,
    driver_id,
    full_name,
    abbreviation,
    team_id,
    team_name,
    team_color,
    CAST(finishing_position AS INTEGER) AS sprint_qualifying_position,
    q1_seconds,
    q2_seconds
  FROM results_filtered
  WHERE session IN ('SQ', 'SS')
),

-- Determine event format and get appropriate session dates
target_events AS (
  SELECT
    q.event_year,
    q.event,
    q.event_name,
    ds_q.event_number AS target_event_number,
    ds_q.session_date_utc AS qualifying_session_date_utc,
    -- For sprint weekends, get SQ/SS session date
    ds_sq.session_date_utc AS sprint_qualifying_session_date_utc,
    -- Determine if this is a sprint weekend
    CASE WHEN ds_sq.session_date_utc IS NOT NULL THEN TRUE ELSE FALSE END AS is_sprint_weekend
  FROM (
    SELECT DISTINCT event_year, event, event_name
    FROM q_results
  ) q
  LEFT JOIN dim_sessions ds_q
    ON ds_q.event_year = q.event_year
   AND ds_q.event = q.event
   AND ds_q.session = 'Q'
  LEFT JOIN dim_sessions ds_sq
    ON ds_sq.event_year = q.event_year
   AND ds_sq.event = q.event
   AND ds_sq.session IN ('SQ', 'SS')
),

driver_targets AS (
  SELECT
    q.*,
    te.target_event_number,
    te.qualifying_session_date_utc,
    te.sprint_qualifying_session_date_utc,
    te.is_sprint_weekend
  FROM q_results q
  LEFT JOIN target_events te
    ON te.event_year = q.event_year
   AND te.event = q.event
),

practice_laps AS (
  SELECT
    CAST(event_year AS INTEGER) AS event_year,
    event,
    CAST(driver_number AS INTEGER) AS driver_number,
    session,
    lap_time_seconds,
    sector1_time_seconds,
    sector2_time_seconds,
    sector3_time_seconds,
    speed_fl,
    speed_st,
    compound,
    tyre_life
  FROM laps_filtered
  WHERE session_type = 'practice'
    AND lap_time_seconds IS NOT NULL
    AND COALESCE(deleted, FALSE) = FALSE
    AND COALESCE(is_pit_lap, FALSE) = FALSE
    AND COALESCE(is_accurate, TRUE) = TRUE
),

practice_agg AS (
  SELECT
    event_year,
    event,
    driver_number,
    COUNT(*) AS practice_laps_count,
    MIN(lap_time_seconds) AS practice_best_lap_time_seconds,
    MEDIAN(lap_time_seconds) AS practice_median_lap_time_seconds,
    AVG(lap_time_seconds) AS practice_mean_lap_time_seconds,
    STDDEV_SAMP(lap_time_seconds) AS practice_std_lap_time_seconds,
    MIN(sector1_time_seconds) AS practice_best_sector1_time_seconds,
    MIN(sector2_time_seconds) AS practice_best_sector2_time_seconds,
    MIN(sector3_time_seconds) AS practice_best_sector3_time_seconds,
    MAX(speed_fl) AS practice_max_speed_fl,
    MAX(speed_st) AS practice_max_speed_st,

    -- Pace by compound (best lap on each compound)
    MIN(lap_time_seconds) FILTER (compound = 'SOFT') AS practice_best_soft_lap_time_seconds,
    MIN(lap_time_seconds) FILTER (compound = 'MEDIUM') AS practice_best_medium_lap_time_seconds,
    MIN(lap_time_seconds) FILTER (compound = 'HARD') AS practice_best_hard_lap_time_seconds
  FROM practice_laps
  GROUP BY 1,2,3
),

q_laps AS (
  SELECT
    CAST(event_year AS INTEGER) AS event_year,
    event,
    CAST(driver_number AS INTEGER) AS driver_number,
    lap_time_seconds
  FROM laps_filtered
  WHERE session = 'Q'
    AND lap_time_seconds IS NOT NULL
    AND COALESCE(deleted, FALSE) = FALSE
    AND COALESCE(is_pit_lap, FALSE) = FALSE
    AND COALESCE(is_accurate, TRUE) = TRUE
),

q_laps_agg AS (
  SELECT
    event_year,
    event,
    driver_number,
    MIN(lap_time_seconds) AS qualifying_best_lap_time_seconds,
    MEDIAN(lap_time_seconds) AS qualifying_median_lap_time_seconds
  FROM q_laps
  GROUP BY 1,2,3
),

-- Sprint qualifying/shootout laps (SQ/SS) - used for sprint race grid and pace features
sq_laps AS (
  SELECT
    CAST(event_year AS INTEGER) AS event_year,
    event,
    CAST(driver_number AS INTEGER) AS driver_number,
    lap_time_seconds,
    sector1_time_seconds,
    sector2_time_seconds,
    sector3_time_seconds,
    speed_fl,
    speed_st,
    compound
  FROM laps_filtered
  WHERE session IN ('SQ', 'SS')
    AND lap_time_seconds IS NOT NULL
    AND COALESCE(deleted, FALSE) = FALSE
    AND COALESCE(is_pit_lap, FALSE) = FALSE
    AND COALESCE(is_accurate, TRUE) = TRUE
),

sq_laps_agg AS (
  SELECT
    event_year,
    event,
    driver_number,
    MIN(lap_time_seconds) AS sprint_qualifying_best_lap_time_seconds,
    MEDIAN(lap_time_seconds) AS sprint_qualifying_median_lap_time_seconds,
    MIN(sector1_time_seconds) AS sprint_qualifying_best_sector1_time_seconds,
    MIN(sector2_time_seconds) AS sprint_qualifying_best_sector2_time_seconds,
    MIN(sector3_time_seconds) AS sprint_qualifying_best_sector3_time_seconds,
    MAX(speed_fl) AS sprint_qualifying_max_speed_fl,
    MAX(speed_st) AS sprint_qualifying_max_speed_st
  FROM sq_laps
  GROUP BY 1,2,3
),

points_history AS (
  -- All prior race/sprint sessions that occurred before the target event's Qualifying timestamp
  SELECT
    dt.event_year,
    dt.event AS target_event,
    dt.driver_number,
    SUM(COALESCE(TRY_CAST(sr.points AS DOUBLE), 0.0)) AS season_points_to_date
  FROM driver_targets dt
  JOIN dim_sessions ds
    ON ds.event_year = dt.event_year
  JOIN silver_results sr
    ON sr.event_year = ds.event_year
   AND sr.event = ds.event
   AND sr.session = ds.session
   AND TRY_CAST(sr.driver_number AS INTEGER) = dt.driver_number
  WHERE ds.session IN ('R', 'S')
    AND ds.session_date_utc IS NOT NULL
    AND dt.qualifying_session_date_utc IS NOT NULL
    AND ds.session_date_utc < dt.qualifying_session_date_utc
  GROUP BY 1,2,3
),

race_history_ranked AS (
  -- Prior races (R only) for last-3 form, ranked most-recent-first per target event and driver
  SELECT
    dt.event_year,
    dt.event AS target_event,
    dt.driver_number,
    ds.event_number AS hist_event_number,
    TRY_CAST(sr.finishing_position AS INTEGER) AS hist_finishing_position,
    COALESCE(TRY_CAST(sr.is_retired AS BOOLEAN), FALSE) AS hist_is_retired,
    COALESCE(TRY_CAST(sr.points AS DOUBLE), 0.0) AS hist_points,
    ROW_NUMBER() OVER (
      PARTITION BY dt.event_year, dt.event, dt.driver_number
      ORDER BY ds.event_number DESC
    ) AS rn
  FROM driver_targets dt
  JOIN dim_sessions ds
    ON ds.event_year = dt.event_year
   AND ds.session = 'R'
  JOIN silver_results sr
    ON sr.event_year = ds.event_year
   AND sr.event = ds.event
   AND sr.session = ds.session
   AND TRY_CAST(sr.driver_number AS INTEGER) = dt.driver_number
  WHERE ds.session_date_utc IS NOT NULL
    AND dt.qualifying_session_date_utc IS NOT NULL
    AND ds.session_date_utc < dt.qualifying_session_date_utc
),

last3_race_form AS (
  SELECT
    event_year,
    target_event,
    driver_number,
    SUM(hist_points) AS last3_race_points,
    AVG(hist_finishing_position) AS last3_avg_finish,
    SUM(CASE WHEN hist_is_retired THEN 1 ELSE 0 END) AS last3_dnfs,
    COUNT(*) AS last3_race_count
  FROM race_history_ranked
  WHERE rn <= 3
  GROUP BY 1,2,3
),

circuit_last3y AS (
  -- Circuit performance over previous 3 seasons (race sessions only), excluding the current season.
  SELECT
    dt.event_year AS target_event_year,
    dt.event AS target_event,
    dt.driver_number,
    AVG(TRY_CAST(sr.finishing_position AS INTEGER)) AS circuit_last3y_avg_finish,
    AVG(COALESCE(TRY_CAST(sr.points AS DOUBLE), 0.0)) AS circuit_last3y_avg_points,
    COUNT(*) AS circuit_last3y_race_count
  FROM driver_targets dt
  JOIN dim_sessions ds
    ON ds.event = dt.event
   AND ds.session = 'R'
   AND ds.event_year BETWEEN (dt.event_year - 3) AND (dt.event_year - 1)
  JOIN silver_results sr
    ON sr.event_year = ds.event_year
   AND sr.event = ds.event
   AND sr.session = ds.session
   AND TRY_CAST(sr.driver_number AS INTEGER) = dt.driver_number
  GROUP BY 1,2,3
),

-- Race lap pace history: historical race lap times per driver per circuit
race_laps_hist AS (
  SELECT
    dt.event_year AS target_event_year,
    dt.event AS target_event,
    dt.driver_number,
    AVG(ll.lap_time_seconds) AS hist_avg_race_lap_time,
    STDDEV_SAMP(ll.lap_time_seconds) AS hist_race_lap_time_std,
    COUNT(*) AS hist_race_laps_count
  FROM driver_targets dt
  JOIN dim_sessions ds
    ON ds.event = dt.event
   AND ds.session = 'R'
   AND ds.event_year BETWEEN (dt.event_year - 3) AND (dt.event_year - 1)
  JOIN silver_laps ll
    ON ll.event_year = ds.event_year
   AND ll.event = ds.event
   AND ll.session = 'R'
   AND TRY_CAST(ll.driver_number AS INTEGER) = dt.driver_number
   AND ll.lap_time_seconds IS NOT NULL
   AND COALESCE(ll.deleted, FALSE) = FALSE
   AND COALESCE(ll.is_pit_lap, FALSE) = FALSE
   AND COALESCE(ll.is_accurate, TRUE) = TRUE
  GROUP BY 1,2,3
)

SELECT
  dt.event_year,
  dt.event,
  dt.event_name,
  dt.driver_number,
  dt.driver_id,
  dt.full_name,
  dt.abbreviation,
  dt.team_id,
  dt.team_name,
  dt.team_color,

  dt.qualifying_position,
  dt.q1_seconds,
  dt.q2_seconds,
  dt.q3_seconds,

  qla.qualifying_best_lap_time_seconds,
  qla.qualifying_median_lap_time_seconds,

  pa.practice_laps_count,
  pa.practice_best_lap_time_seconds,
  pa.practice_median_lap_time_seconds,
  pa.practice_mean_lap_time_seconds,
  pa.practice_std_lap_time_seconds,
  pa.practice_best_sector1_time_seconds,
  pa.practice_best_sector2_time_seconds,
  pa.practice_best_sector3_time_seconds,
  pa.practice_max_speed_fl,
  pa.practice_max_speed_st,
  pa.practice_best_soft_lap_time_seconds,
  pa.practice_best_medium_lap_time_seconds,
  pa.practice_best_hard_lap_time_seconds,

  -- Sprint qualifying/shootout features (SQ/SS)
  sqla.sprint_qualifying_best_lap_time_seconds,
  sqla.sprint_qualifying_median_lap_time_seconds,
  sqla.sprint_qualifying_best_sector1_time_seconds,
  sqla.sprint_qualifying_best_sector2_time_seconds,
  sqla.sprint_qualifying_best_sector3_time_seconds,
  sqla.sprint_qualifying_max_speed_fl,
  sqla.sprint_qualifying_max_speed_st,

  dt.target_event_number,
  dt.qualifying_session_date_utc,
  dt.sprint_qualifying_session_date_utc,
  dt.is_sprint_weekend,

  ph.season_points_to_date,
  l3.last3_race_points,
  l3.last3_avg_finish,
  l3.last3_dnfs,
  l3.last3_race_count,
  c3.circuit_last3y_avg_finish,
  c3.circuit_last3y_avg_points,
  c3.circuit_last3y_race_count,

  -- Race pace features (historical, from prior seasons)
  rlh.hist_avg_race_lap_time,
  rlh.hist_race_lap_time_std,
  rlh.hist_race_laps_count,

  -- Weather features (LEFT JOIN: NULL for events without weather data)
  -- is_wet_race: precipitation_sum_mm >= 1.0 mm (derived here; silver may add column later)
  (COALESCE(w.precipitation_sum_mm, 0) >= 1.0) AS is_wet_race,
  w.temperature_max_c,
  w.temperature_min_c,
  w.wind_speed_max_kmh

FROM driver_targets dt
LEFT JOIN practice_agg pa
  ON pa.event_year = dt.event_year
 AND pa.event = dt.event
 AND pa.driver_number = dt.driver_number
LEFT JOIN q_laps_agg qla
  ON qla.event_year = dt.event_year
 AND qla.event = dt.event
 AND qla.driver_number = dt.driver_number
LEFT JOIN sq_laps_agg sqla
  ON sqla.event_year = dt.event_year
 AND sqla.event = dt.event
 AND sqla.driver_number = dt.driver_number
LEFT JOIN points_history ph
  ON ph.event_year = dt.event_year
 AND ph.target_event = dt.event
 AND ph.driver_number = dt.driver_number
LEFT JOIN last3_race_form l3
  ON l3.event_year = dt.event_year
 AND l3.target_event = dt.event
 AND l3.driver_number = dt.driver_number
LEFT JOIN circuit_last3y c3
  ON c3.target_event_year = dt.event_year
 AND c3.target_event = dt.event
 AND c3.driver_number = dt.driver_number
LEFT JOIN race_laps_hist rlh
  ON rlh.target_event_year = dt.event_year
 AND rlh.target_event = dt.event
 AND rlh.driver_number = dt.driver_number
LEFT JOIN weather w
  ON w.event_year = dt.event_year
 AND w.event = dt.event
;


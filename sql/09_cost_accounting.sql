-- Dollar cost of the service run, from Snowflake's own metering views, at the list prices in
-- PRICE_ASSUMPTIONS (Standard edition, On Demand, AWS US West (Oregon)).
-- ACCOUNT_USAGE lags: SPCS and warehouse metering by up to ~3 h, query attribution by up
-- to ~8 h. Re-run until no line says PENDING.
-- Run lines are statement wall time x list rate: the node and the warehouse are both
-- reserved for the whole statement. QUERY_ATTRIBUTION_HISTORY is kept as a check only
-- (last query below). On this single-session XS warehouse it spread the metered credits
-- unevenly: the service run's 18 s statement got 0.0121 credits, about 44 s of XS time.
-- $SNOW -f sql/09_cost_accounting.sql
USE SCHEMA <% database %>.<% schema %>;
USE WAREHOUSE <% analysis_warehouse %>;

SET platform_usd = (SELECT VALUE FROM PRICE_ASSUMPTIONS WHERE ITEM = 'PLATFORM_CREDIT_USD');
SET gpu_rate     = (SELECT VALUE FROM PRICE_ASSUMPTIONS WHERE ITEM = 'GPU_NV_S_CREDITS_PER_HOUR');
SET wh_rate      = (SELECT VALUE FROM PRICE_ASSUMPTIONS WHERE ITEM = 'WH_XS_STANDARD_CREDITS_PER_HOUR');
SET bench_start  = (SELECT DATEADD(hour, -3, MIN(START_TIME)) FROM BENCH_QUERY_TIMES);

CREATE OR REPLACE TABLE COST_LINES AS
WITH stmt AS (SELECT * FROM BENCH_QUERY_TIMES WHERE RUN_ID = 'service'),
-- GPU node time while the run's service-function statement ran.
gpu_run AS (
  SELECT RUN_ID, 'gpu_node' AS COMPONENT, SUM(ELAPSED_S) / 3600 * $gpu_rate AS CREDITS,
         'statement wall time x GPU_NV_S rate' AS BASIS
  FROM stmt WHERE STEP = 'decider' GROUP BY RUN_ID
),
-- XS warehouse time while the run's statements ran.
wh_run AS (
  SELECT RUN_ID, 'warehouse' AS COMPONENT, SUM(ELAPSED_S) / 3600 * $wh_rate AS CREDITS,
         'statement wall time x XS rate' AS BASIS
  FROM stmt GROUP BY RUN_ID
),
per_run AS (SELECT * FROM gpu_run UNION ALL SELECT * FROM wh_run),
pool_total AS (
  SELECT COMPUTE_POOL_NAME, SUM(CREDITS_USED) AS CREDITS
  FROM SNOWFLAKE.ACCOUNT_USAGE.SNOWPARK_CONTAINER_SERVICES_HISTORY
  WHERE COMPUTE_POOL_NAME IN ('DECIDER_BENCH_GPU_POOL', 'DECIDER_BENCH_BUILD_POOL') AND START_TIME >= $bench_start
  GROUP BY 1
),
wh_total AS (
  SELECT SUM(CREDITS_USED_COMPUTE) AS CREDITS
  FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
  WHERE WAREHOUSE_NAME = 'DECIDER_BENCH_WH' AND START_TIME >= $bench_start
),
shared AS (
  -- GPU node time outside the run: node start, model load and warm-up, pre-flight,
  -- and shutdown.
  SELECT 'shared' AS RUN_ID, 'gpu_node' AS COMPONENT,
         (SELECT CREDITS FROM pool_total WHERE COMPUTE_POOL_NAME = 'DECIDER_BENCH_GPU_POOL')
           - (SELECT SUM(CREDITS) FROM gpu_run) AS CREDITS,
         'SNOWPARK_CONTAINER_SERVICES_HISTORY minus the run lines' AS BASIS
  UNION ALL
  SELECT 'shared', 'image_build',
         (SELECT CREDITS FROM pool_total WHERE COMPUTE_POOL_NAME = 'DECIDER_BENCH_BUILD_POOL'),
         'SNOWPARK_CONTAINER_SERVICES_HISTORY'
  UNION ALL
  -- Warehouse time outside the run's statement: pre-flight, idle before auto-suspend, and the
  -- 60-second minimum per resume.
  SELECT 'shared', 'warehouse', (SELECT CREDITS FROM wh_total) - (SELECT SUM(CREDITS) FROM wh_run),
         'WAREHOUSE_METERING_HISTORY minus the run lines'
)
SELECT RUN_ID, COMPONENT, CREDITS, BASIS,
       CREDITS * $platform_usd AS USD,
       CASE WHEN CREDITS IS NULL THEN 'PENDING'
            WHEN BASIS LIKE 'statement wall time%' THEN 'METERED_TIME'
            ELSE 'METERED' END AS STATUS
FROM (SELECT * FROM per_run UNION ALL SELECT * FROM shared);

-- Line items, and the run's cost per 1,000 decisions (231 decisions).
SELECT RUN_ID, COMPONENT, ROUND(CREDITS, 6) AS CREDITS, ROUND(USD, 4) AS USD,
       IFF(RUN_ID = 'shared', NULL, ROUND(USD / 231 * 1000, 4)) AS USD_PER_1000, STATUS
FROM COST_LINES ORDER BY RUN_ID, COMPONENT;

SELECT COALESCE(RUN_ID, 'ALL-IN') AS RUN_ID, ROUND(SUM(USD), 4) AS USD,
       IFF(RUN_ID = 'service', ROUND(SUM(USD) / 231 * 1000, 4), NULL) AS USD_PER_1000,
       COUNT_IF(STATUS = 'PENDING') AS PENDING
FROM COST_LINES GROUP BY ROLLUP (RUN_ID) ORDER BY RUN_ID NULLS LAST;

-- Check: the warehouse's metered credits split into attributed and idle, beside the
-- benchmark statements' attribution and their wall time at the XS rate.
SELECT 'session total' AS SCOPE, NULL AS RUN_ID, NULL AS STEP, SUM(CREDITS_USED_COMPUTE) AS METERED,
       SUM(CREDITS_ATTRIBUTED_COMPUTE_QUERIES) AS ATTRIBUTED,
       SUM(CREDITS_USED_COMPUTE) - SUM(CREDITS_ATTRIBUTED_COMPUTE_QUERIES) AS IDLE, NULL AS WALL_TIME_X_RATE
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE WAREHOUSE_NAME = 'DECIDER_BENCH_WH' AND START_TIME >= $bench_start
UNION ALL
SELECT 'statement', b.RUN_ID, b.STEP, NULL, a.CREDITS_ATTRIBUTED_COMPUTE, NULL,
       b.ELAPSED_S / 3600 * $wh_rate
FROM BENCH_QUERY_TIMES b
LEFT JOIN SNOWFLAKE.ACCOUNT_USAGE.QUERY_ATTRIBUTION_HISTORY a ON a.QUERY_ID = b.QUERY_ID
ORDER BY SCOPE, RUN_ID, STEP;

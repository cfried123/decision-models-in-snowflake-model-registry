-- DTR-Bench through the Model Registry service from SQL: every DTR_ITEMS row calls
-- DTR_DECIDER_HTTP!SYSTEM_ONE once (batched by the service, max_batch_rows=32).
-- Results land in DECIDER_ANSWERS under RUN_ID = '<% run_id %>' and score through DTR_SCORES.
USE SCHEMA <% database %>.<% schema %>;
USE WAREHOUSE DECIDER_BENCH_WH;
ALTER SESSION SET USE_CACHED_RESULT = FALSE;
ALTER SESSION SET QUERY_TAG = 'decider_bench:dtr_service:<% run_id %>';
DELETE FROM DECIDER_ANSWERS WHERE RUN_ID = '<% run_id %>';
INSERT INTO DECIDER_ANSWERS (RUN_ID, ITEM_ID, RESULT)
SELECT '<% run_id %>', ITEM_ID, <% service %>!SYSTEM_ONE(STATE_JSON, QUESTIONS_JSON)
FROM DTR_ITEMS;
SET qid = LAST_QUERY_ID();
ALTER SESSION UNSET QUERY_TAG;
INSERT INTO RUN_LOG (RUN_ID, STEP, QUERY_ID, QUERY_TAG, WAREHOUSE_NAME, N_ROWS)
SELECT '<% run_id %>', 'decider', $qid, 'decider_bench:dtr_service:<% run_id %>', 'DECIDER_BENCH_WH', COUNT(*)
FROM DECIDER_ANSWERS WHERE RUN_ID = '<% run_id %>';
SELECT QTYPE, COUNT(*) AS N, COUNT_IF(ANSWER IS NULL) AS MISSING, ROUND(MEDIAN(ELAPSED_MS)) AS P50_MS
FROM DTR_DECISIONS WHERE RUN_ID = '<% run_id %>' GROUP BY QTYPE ORDER BY QTYPE;

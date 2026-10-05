-- The service run: decider-2b alone on all 231 JevBench public decisions.
-- $SNOW -f sql/05_service_decider_only.sql
USE SCHEMA <% database %>.<% schema %>;
USE WAREHOUSE DECIDER_BENCH_WH;
ALTER SESSION SET QUERY_TAG = 'decider_bench:service_decider_only';

INSERT INTO DECIDER_ANSWERS (RUN_ID, ITEM_ID, RESULT)
SELECT 'service', ITEM_ID, DECIDER_2B_SVC!SYSTEM_ONE(STATE_JSON, QUESTIONS_JSON)
FROM JEVBENCH_ITEMS;
SET qid = LAST_QUERY_ID();

-- Bookkeeping runs on another warehouse so the benchmark warehouse meters only the run.
ALTER SESSION UNSET QUERY_TAG;
USE WAREHOUSE <% analysis_warehouse %>;
INSERT INTO RUN_LOG (RUN_ID, STEP, QUERY_ID, QUERY_TAG, WAREHOUSE_NAME, N_ROWS)
SELECT 'service', 'decider', $qid, 'decider_bench:service_decider_only', 'DECIDER_BENCH_WH', COUNT(*)
FROM DECIDER_ANSWERS WHERE RUN_ID = 'service';

SELECT QTYPE, COUNT(*) AS N, COUNT_IF(ANSWER IS NULL) AS MISSING,
       ROUND(MEDIAN(ELAPSED_MS)) AS P50_MS
FROM DECIDER_DECISIONS WHERE RUN_ID = 'service' GROUP BY QTYPE ORDER BY QTYPE;

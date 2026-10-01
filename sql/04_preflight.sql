-- Pre-flight: one item of each answer type through the service, before any timed run.
-- $SNOW -f sql/04_preflight.sql
USE SCHEMA <% database %>.<% schema %>;
USE WAREHOUSE DECIDER_BENCH_WH;
ALTER SESSION SET QUERY_TAG = 'decider_bench:preflight';

INSERT INTO DECIDER_ANSWERS (RUN_ID, ITEM_ID, RESULT)
SELECT 'preflight', ITEM_ID, DECIDER_2B_SVC!SYSTEM_ONE(STATE_JSON, QUESTIONS_JSON)
FROM JEVBENCH_ITEMS
WHERE ITEM_ID IN ('easy-intent-00', 'easy-fact-00', 'original-ordinal-01-0');
SET qid = LAST_QUERY_ID();

ALTER SESSION UNSET QUERY_TAG;
USE WAREHOUSE <% analysis_warehouse %>;
INSERT INTO RUN_LOG (RUN_ID, STEP, QUERY_ID, QUERY_TAG, WAREHOUSE_NAME, N_ROWS)
SELECT 'preflight', 'decider', $qid, 'decider_bench:preflight', 'DECIDER_BENCH_WH', COUNT(*)
FROM DECIDER_ANSWERS WHERE RUN_ID = 'preflight';

SELECT ITEM_ID, QTYPE, DECIDER_LABEL, ESCALATE, ROUND(ELAPSED_MS) AS ELAPSED_MS, INPUT_TOKENS,
       ANSWER:confidence::FLOAT AS CONFIDENCE, ANSWER:noul::FLOAT AS P_YES
FROM DECIDER_DECISIONS WHERE RUN_ID = 'preflight' ORDER BY ITEM_ID;

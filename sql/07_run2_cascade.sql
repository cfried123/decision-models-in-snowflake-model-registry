-- Run 2: decider-2b on every decision, AI_CLASSIFY on the ones decider is unsure of.
-- Rule, fixed before Run 1: escalate Choice and Score answers with confidence < 0.5,
-- and Noul answers with 0.2 < P(yes) < 0.8 (see DECIDER_DECISIONS.ESCALATE).
-- $SNOW -f sql/07_run2_cascade.sql
USE SCHEMA <% database %>.<% schema %>;
USE WAREHOUSE DECIDER_BENCH_WH;
ALTER SESSION SET QUERY_TAG = 'decider_bench:run2_cascade';

-- Stage 1: a fresh decider pass over all 231 items.
INSERT INTO DECIDER_ANSWERS (RUN_ID, ITEM_ID, RESULT)
SELECT 'run2', ITEM_ID, DECIDER_2B_V11_SVC!SYSTEM_ONE(STATE_JSON, QUESTIONS_JSON)
FROM JEVBENCH_ITEMS;
SET qid_decider = LAST_QUERY_ID();

-- Stage 2: pick the rows to escalate, in their own statement.
CREATE OR REPLACE TABLE RUN2_ESCALATED AS
SELECT ITEM_ID FROM DECIDER_DECISIONS WHERE RUN_ID = 'run2' AND ESCALATE;
SET qid_filter = LAST_QUERY_ID();

-- Stage 3: AI_CLASSIFY on those rows only.
INSERT INTO AI_CLASSIFY_ANSWERS (RUN_ID, ITEM_ID, RESULT)
SELECT 'run2', i.ITEM_ID,
       AI_CLASSIFY(i.CLASSIFY_INPUT, i.CLASSIFY_CATEGORIES,
                   OBJECT_CONSTRUCT('task_description', i.CLASSIFY_TASK_DESCRIPTION, 'output_mode', 'single'))
FROM RUN2_ESCALATED e
JOIN JEVBENCH_ITEMS i ON i.ITEM_ID = e.ITEM_ID;
SET qid_classify = LAST_QUERY_ID();

-- Stage 4: the cascade's answers.
CREATE OR REPLACE TABLE RUN2_FINAL AS
SELECT * FROM CASCADE_DECISIONS WHERE RUN_ID = 'run2';
SET qid_final = LAST_QUERY_ID();

ALTER SESSION UNSET QUERY_TAG;
USE WAREHOUSE <% analysis_warehouse %>;
INSERT INTO RUN_LOG (RUN_ID, STEP, QUERY_ID, QUERY_TAG, WAREHOUSE_NAME, N_ROWS)
SELECT 'run2', 'decider', $qid_decider, 'decider_bench:run2_cascade', 'DECIDER_BENCH_WH',
       (SELECT COUNT(*) FROM DECIDER_ANSWERS WHERE RUN_ID = 'run2')
UNION ALL SELECT 'run2', 'escalation_filter', $qid_filter, 'decider_bench:run2_cascade', 'DECIDER_BENCH_WH',
       (SELECT COUNT(*) FROM RUN2_ESCALATED)
UNION ALL SELECT 'run2', 'ai_classify', $qid_classify, 'decider_bench:run2_cascade', 'DECIDER_BENCH_WH',
       (SELECT COUNT(*) FROM AI_CLASSIFY_ANSWERS WHERE RUN_ID = 'run2')
UNION ALL SELECT 'run2', 'final_answers', $qid_final, 'decider_bench:run2_cascade', 'DECIDER_BENCH_WH',
       (SELECT COUNT(*) FROM RUN2_FINAL);

SELECT SOURCE, QTYPE, COUNT(*) AS N, COUNT_IF(ESCALATE AND AI_LABEL IS NULL) AS AI_NULLS
FROM RUN2_FINAL GROUP BY 1, 2 ORDER BY 1, 2;

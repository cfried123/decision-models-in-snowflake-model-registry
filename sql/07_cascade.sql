-- The cascade: decider-2b on every decision, AI_CLASSIFY on the ones decider is unsure of.
-- Rule, fixed before any run: escalate Choice and Score answers with confidence < 0.5,
-- and Noul answers with 0.2 < P(yes) < 0.8 (see DECIDER_DECISIONS.ESCALATE).
-- $SNOW -f sql/07_cascade.sql
USE SCHEMA <% database %>.<% schema %>;
USE WAREHOUSE DECIDER_BENCH_WH;
ALTER SESSION SET QUERY_TAG = 'decider_bench:cascade';

-- Stage 1: a fresh decider pass over all 231 items.
INSERT INTO DECIDER_ANSWERS (RUN_ID, ITEM_ID, RESULT)
SELECT 'cascade', ITEM_ID, DECIDER_2B_SVC!SYSTEM_ONE(STATE_JSON, QUESTIONS_JSON)
FROM JEVBENCH_ITEMS;
SET qid_decider = LAST_QUERY_ID();

-- Stage 2: pick the rows to escalate, in their own statement.
CREATE OR REPLACE TABLE ESCALATED AS
SELECT ITEM_ID FROM DECIDER_DECISIONS WHERE RUN_ID = 'cascade' AND ESCALATE;
SET qid_filter = LAST_QUERY_ID();

-- Stage 3: AI_CLASSIFY on those rows only.
INSERT INTO AI_CLASSIFY_ANSWERS (RUN_ID, ITEM_ID, RESULT)
SELECT 'cascade', i.ITEM_ID,
       AI_CLASSIFY(i.CLASSIFY_INPUT, i.CLASSIFY_CATEGORIES,
                   OBJECT_CONSTRUCT('task_description', i.CLASSIFY_TASK_DESCRIPTION, 'output_mode', 'single'))
FROM ESCALATED e
JOIN JEVBENCH_ITEMS i ON i.ITEM_ID = e.ITEM_ID;
SET qid_classify = LAST_QUERY_ID();

-- Stage 4: the cascade's answers.
CREATE OR REPLACE TABLE CASCADE_FINAL AS
SELECT * FROM CASCADE_DECISIONS WHERE RUN_ID = 'cascade';
SET qid_final = LAST_QUERY_ID();

ALTER SESSION UNSET QUERY_TAG;
USE WAREHOUSE <% analysis_warehouse %>;
INSERT INTO RUN_LOG (RUN_ID, STEP, QUERY_ID, QUERY_TAG, WAREHOUSE_NAME, N_ROWS)
SELECT 'cascade', 'decider', $qid_decider, 'decider_bench:cascade', 'DECIDER_BENCH_WH',
       (SELECT COUNT(*) FROM DECIDER_ANSWERS WHERE RUN_ID = 'cascade')
UNION ALL SELECT 'cascade', 'escalation_filter', $qid_filter, 'decider_bench:cascade', 'DECIDER_BENCH_WH',
       (SELECT COUNT(*) FROM ESCALATED)
UNION ALL SELECT 'cascade', 'ai_classify', $qid_classify, 'decider_bench:cascade', 'DECIDER_BENCH_WH',
       (SELECT COUNT(*) FROM AI_CLASSIFY_ANSWERS WHERE RUN_ID = 'cascade')
UNION ALL SELECT 'cascade', 'final_answers', $qid_final, 'decider_bench:cascade', 'DECIDER_BENCH_WH',
       (SELECT COUNT(*) FROM CASCADE_FINAL);

SELECT SOURCE, QTYPE, COUNT(*) AS N, COUNT_IF(ESCALATE AND AI_LABEL IS NULL) AS AI_NULLS
FROM CASCADE_FINAL GROUP BY 1, 2 ORDER BY 1, 2;

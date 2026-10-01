-- Threshold sweep, run after the fact (not pre-registered): AI_CLASSIFY answers for the rows
-- the cascade didn't escalate whose confidence is below 0.9. Confidence is decider's `confidence`
-- for Choice and Score, and |2 * P(yes) - 1| for Noul (decider's formula with two options).
-- The 58 rows the cascade escalated keep its labels. Every row decider got wrong in the service run
-- has confidence below 0.9. Safe to re-run: rows that already have a label are skipped.
-- $SNOW -f sql/10_threshold_sweep.sql
USE SCHEMA <% database %>.<% schema %>;
USE WAREHOUSE <% analysis_warehouse %>;
ALTER SESSION SET QUERY_TAG = 'decider_bench:threshold_sweep';

INSERT INTO AI_CLASSIFY_ANSWERS (RUN_ID, ITEM_ID, RESULT)
SELECT 'sweep', i.ITEM_ID,
       AI_CLASSIFY(i.CLASSIFY_INPUT, i.CLASSIFY_CATEGORIES,
                   OBJECT_CONSTRUCT('task_description', i.CLASSIFY_TASK_DESCRIPTION, 'output_mode', 'single'))
FROM JEVBENCH_ITEMS i
JOIN DECIDER_DECISIONS d ON d.ITEM_ID = i.ITEM_ID AND d.RUN_ID = 'service'
WHERE i.ITEM_ID NOT IN (SELECT ITEM_ID FROM AI_CLASSIFY_ANSWERS WHERE RUN_ID IN ('cascade', 'sweep'))
  AND IFF(d.QTYPE = 'noul', ABS(2 * d.ANSWER:noul::FLOAT - 1), d.ANSWER:confidence::FLOAT) < 0.9;
SET qid = LAST_QUERY_ID();

ALTER SESSION UNSET QUERY_TAG;
INSERT INTO RUN_LOG (RUN_ID, STEP, QUERY_ID, QUERY_TAG, WAREHOUSE_NAME, N_ROWS)
SELECT 'sweep', 'ai_classify', $qid, 'decider_bench:threshold_sweep', '<% analysis_warehouse %>',
       (SELECT COUNT(*) FROM AI_CLASSIFY_ANSWERS WHERE RUN_ID = 'sweep');

SELECT COUNT(*) AS N, COUNT_IF(RESULT:labels[0] IS NULL) AS NULL_LABELS
FROM AI_CLASSIFY_ANSWERS WHERE RUN_ID = 'sweep';

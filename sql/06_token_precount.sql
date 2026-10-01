-- Token pre-count for the cascade (AI_COUNT_TOKENS costs warehouse time, no AI credits).
-- Counts the rows Run 1's decider answers would escalate, and all 231 as an upper bound.
-- $SNOW -f sql/06_token_precount.sql
USE SCHEMA <% database %>.<% schema %>;
USE WAREHOUSE DECIDER_BENCH_WH;
ALTER SESSION SET QUERY_TAG = 'decider_bench:token_precount';

-- AI_COUNT_TOKENS returns NULL, with no error, when the input contains { or } (every
-- JSON state). AI_CLASSIFY handles those inputs normally, so those rows are counted
-- with the braces swapped for parentheses and flagged TOKENS_ESTIMATED.
CREATE OR REPLACE TABLE TOKEN_PRECOUNT AS
WITH counted AS (
  SELECT i.ITEM_ID, i.TIER, i.QTYPE, d.ESCALATE AS WOULD_ESCALATE_RUN1,
         AI_COUNT_TOKENS('ai_classify', i.CLASSIFY_INPUT, i.CLASSIFY_CATEGORIES,
                         OBJECT_CONSTRUCT('task_description', i.CLASSIFY_TASK_DESCRIPTION)) AS EXACT,
         AI_COUNT_TOKENS('ai_classify', TRANSLATE(i.CLASSIFY_INPUT, '{}', '()'), i.CLASSIFY_CATEGORIES,
                         OBJECT_CONSTRUCT('task_description', i.CLASSIFY_TASK_DESCRIPTION)) AS NO_BRACES
  FROM JEVBENCH_ITEMS i
  JOIN DECIDER_DECISIONS d ON d.ITEM_ID = i.ITEM_ID AND d.RUN_ID = 'run1'
)
SELECT ITEM_ID, TIER, QTYPE, WOULD_ESCALATE_RUN1,
       COALESCE(EXACT, NO_BRACES) AS TOKENS, EXACT IS NULL AS TOKENS_ESTIMATED
FROM counted;
SET qid = LAST_QUERY_ID();

ALTER SESSION UNSET QUERY_TAG;
USE WAREHOUSE <% analysis_warehouse %>;
INSERT INTO RUN_LOG (RUN_ID, STEP, QUERY_ID, QUERY_TAG, WAREHOUSE_NAME, N_ROWS)
SELECT 'shared', 'token_precount', $qid, 'decider_bench:token_precount', 'DECIDER_BENCH_WH', COUNT(*)
FROM TOKEN_PRECOUNT;

SELECT COUNT_IF(WOULD_ESCALATE_RUN1)                  AS ROWS_TO_ESCALATE,
       SUM(IFF(WOULD_ESCALATE_RUN1, TOKENS, 0))       AS ESCALATED_INPUT_TOKENS,
       ROUND(AVG(IFF(WOULD_ESCALATE_RUN1, TOKENS, NULL))) AS AVG_TOKENS_PER_ESCALATED_ROW,
       SUM(TOKENS)                                    AS ALL_231_INPUT_TOKENS,
       COUNT_IF(TOKENS IS NULL)                       AS UNCOUNTED_ROWS
FROM TOKEN_PRECOUNT;

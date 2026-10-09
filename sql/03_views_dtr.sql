-- DTR-Bench view over the raw model output. snow sql -f sql/03_views_dtr.sql -D database=... -D schema=...
-- DECISION applies the review policy of strands_decider.dtr_eval.decide: a yes/no abstains
-- when |P(true) - 0.5| < 0.25, a choice or score when its top probability is < 0.5.
USE SCHEMA <% database %>.<% schema %>;

CREATE OR REPLACE VIEW DTR_DECISIONS AS
WITH parsed AS (
  SELECT d.RUN_ID, d.ITEM_ID, i.TASK, i.QTYPE, i.SPLIT, i.EXPECTED, i.DTR_CITATION,
         PARSE_JSON(d.RESULT:ANSWER_JSON::STRING):answers:decision AS ANSWER,
         d.RESULT:ELAPSED_MS::FLOAT     AS ELAPSED_MS,
         d.RESULT:INPUT_TOKENS::NUMBER  AS INPUT_TOKENS,
         d.INSERTED_AT
  FROM DECIDER_ANSWERS d
  JOIN DTR_ITEMS i ON i.ITEM_ID = d.ITEM_ID
),
top_label AS (
  SELECT p.RUN_ID, p.ITEM_ID, MAX_BY(f.key, f.value::FLOAT) AS TOP_LABEL, MAX(f.value::FLOAT) AS TOP_P
  FROM parsed p, LATERAL FLATTEN(INPUT => p.ANSWER:probabilities) f
  GROUP BY 1, 2
),
labelled AS (
  SELECT p.*,
         IFF(p.QTYPE = 'noul', IFF(p.ANSWER:noul::FLOAT >= 0.5, 'true', 'false'), t.TOP_LABEL) AS TOP_LABEL,
         IFF(p.QTYPE = 'noul', GREATEST(p.ANSWER:noul::FLOAT, 1 - p.ANSWER:noul::FLOAT), t.TOP_P) AS CONFIDENCE,
         IFF(p.QTYPE = 'noul', ABS(p.ANSWER:noul::FLOAT - 0.5) < 0.25, t.TOP_P < 0.5) AS ABSTAINED
  FROM parsed p
  LEFT JOIN top_label t ON t.RUN_ID = p.RUN_ID AND t.ITEM_ID = p.ITEM_ID
)
SELECT *,
       IFF(ABSTAINED, 'abstain', TOP_LABEL) AS DECISION,
       (IFF(ABSTAINED, 'abstain', TOP_LABEL) = EXPECTED) AS STRICT_CORRECT
FROM labelled;

-- Per run and split: strict accuracy, coverage, selective accuracy, abstain recall.
CREATE OR REPLACE VIEW DTR_SCORES AS
SELECT RUN_ID, SPLIT, COUNT(*) AS N,
       AVG(STRICT_CORRECT::INT)                                          AS STRICT_ACCURACY,
       AVG((NOT ABSTAINED)::INT)                                         AS COVERAGE,
       AVG(IFF(NOT ABSTAINED AND EXPECTED <> 'abstain', (TOP_LABEL = EXPECTED)::INT, NULL)) AS SELECTIVE_ACCURACY,
       AVG(IFF(EXPECTED = 'abstain', ABSTAINED::INT, NULL))              AS ABSTAIN_RECALL,
       MEDIAN(ELAPSED_MS)                                                AS P50_MODEL_MS
FROM DTR_DECISIONS
GROUP BY ROLLUP (RUN_ID, SPLIT)
HAVING RUN_ID IS NOT NULL;

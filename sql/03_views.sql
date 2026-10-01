-- Views over the raw service output. $SNOW -f sql/03_views.sql
USE SCHEMA <% database %>.<% schema %>;

-- One row per decider answer. The service returns
-- {"ANSWER_JSON": <decider /v1/systemone body>, "ELAPSED_MS", "INPUT_TOKENS", "RUNTIME_JSON"}.
CREATE OR REPLACE VIEW DECIDER_DECISIONS AS
WITH parsed AS (
  SELECT d.RUN_ID, d.ITEM_ID, i.TIER, i.FAMILY, i.QTYPE, i.N_OPTIONS,
         PARSE_JSON(d.RESULT:ANSWER_JSON::STRING):answers:decision AS ANSWER,
         d.RESULT:ELAPSED_MS::FLOAT     AS ELAPSED_MS,
         d.RESULT:INPUT_TOKENS::NUMBER  AS INPUT_TOKENS,
         d.INSERTED_AT
  FROM DECIDER_ANSWERS d
  JOIN JEVBENCH_ITEMS i ON i.ITEM_ID = d.ITEM_ID
),
top_label AS (   -- most likely option for Choice and Score
  SELECT p.RUN_ID, p.ITEM_ID, MAX_BY(f.key, f.value::FLOAT) AS TOP_LABEL
  FROM parsed p, LATERAL FLATTEN(INPUT => p.ANSWER:probabilities) f
  GROUP BY 1, 2
)
SELECT p.*,
       CASE p.QTYPE WHEN 'noul' THEN IFF(p.ANSWER:noul::FLOAT > 0.5, 'yes', 'no')
                    ELSE t.TOP_LABEL END AS DECIDER_LABEL,
       -- Pre-registered escalation rule: Choice and Score below 0.5 confidence,
       -- Noul with P(yes) strictly between 0.2 and 0.8, and any missing answer.
       CASE WHEN p.ANSWER IS NULL THEN TRUE
            WHEN p.QTYPE = 'noul' THEN p.ANSWER:noul::FLOAT > 0.2 AND p.ANSWER:noul::FLOAT < 0.8
            ELSE p.ANSWER:confidence::FLOAT < 0.5 END AS ESCALATE
FROM parsed p
LEFT JOIN top_label t ON t.RUN_ID = p.RUN_ID AND t.ITEM_ID = p.ITEM_ID;

-- The cascade's final answer: AI_CLASSIFY's label where the row was escalated,
-- decider's label otherwise (and wherever AI_CLASSIFY returned NULL).
CREATE OR REPLACE VIEW CASCADE_DECISIONS AS
SELECT d.RUN_ID, d.ITEM_ID, d.TIER, d.FAMILY, d.QTYPE, d.ESCALATE, d.DECIDER_LABEL,
       a.RESULT:labels[0]::STRING AS AI_LABEL,
       IFF(d.ESCALATE AND a.RESULT:labels[0] IS NOT NULL, a.RESULT:labels[0]::STRING, d.DECIDER_LABEL) AS FINAL_LABEL,
       IFF(d.ESCALATE AND a.RESULT:labels[0] IS NOT NULL, 'ai_classify', 'decider') AS SOURCE
FROM DECIDER_DECISIONS d
LEFT JOIN AI_CLASSIFY_ANSWERS a ON a.RUN_ID = d.RUN_ID AND a.ITEM_ID = d.ITEM_ID;

-- View over the raw model output. $SNOW -f sql/03_views.sql
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
                    ELSE t.TOP_LABEL END AS DECIDER_LABEL
FROM parsed p
LEFT JOIN top_label t ON t.RUN_ID = p.RUN_ID AND t.ITEM_ID = p.ITEM_ID;

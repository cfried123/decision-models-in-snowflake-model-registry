-- Objects the DTR Travel Entitlement Adjudicator needs beyond the benchmark schema:
-- the human-review queue, the PAT secret and the egress to the decider's endpoint.
-- DTR_ITEMS comes from python/load_dtr.py.
-- snow sql -c <connection> -f app/dtr-adjudicator/sql/01_dtr_app.sql -D database=DECIDER_BENCH -D schema=BENCH
USE SCHEMA <% database %>.<% schema %>;

CREATE TABLE IF NOT EXISTS DTR_REVIEW_QUEUE (
  REVIEW_ID        NUMBER AUTOINCREMENT START 1 INCREMENT 1,
  REQUEST_ID       STRING,
  TASK             STRING NOT NULL,
  SIDE             STRING NOT NULL,   -- decider | llm
  SCENARIO         STRING NOT NULL,
  DTR_CITATION     STRING,
  MODEL_TOP        STRING,            -- what the model would have said
  MODEL_CONFIDENCE FLOAT,
  CREATED_AT       TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
  RESOLUTION       STRING,            -- the reviewer's determination (an option key)
  REVIEWER_NOTE    STRING,
  RESOLVED_BY      STRING,
  RESOLVED_AT      TIMESTAMP_LTZ
) COMMENT = 'DTR determinations routed to a human because the model abstained';

-- The app authenticates to the decider's public endpoint with a PAT held as a secret.
-- Create the PAT for a least-privilege service user, then:
--   CREATE SECRET IF NOT EXISTS DTR_DECIDER_PAT TYPE = GENERIC_STRING SECRET_STRING = '<pat>';
CREATE NETWORK RULE IF NOT EXISTS DTR_DECIDER_EGRESS
  MODE = EGRESS TYPE = HOST_PORT VALUE_LIST = ('<decider ingress host>.snowflakecomputing.app:443');
-- CREATE EXTERNAL ACCESS INTEGRATION IF NOT EXISTS DTR_DECIDER_EAI
--   ALLOWED_NETWORK_RULES = (DTR_DECIDER_EGRESS) ALLOWED_AUTHENTICATION_SECRETS = (DTR_DECIDER_PAT) ENABLED = TRUE;

-- Support tickets for the triage board, generated once with AI_COMPLETE.
-- A different model family (openai-gpt-5) writes the tickets than the one the board
-- compares against (claude-sonnet-4-6), so the LLM side isn't judging its own text.
-- GEN_CATEGORY and GEN_ESCALATE are what each ticket was generated to be: synthetic
-- labels, so the board reports agreement with them, not accuracy.
-- snow sql -c <connection> -f app/triage-board/sql/01_demo_tickets.sql
USE SCHEMA DEVREL.DECISION_MODEL_BLOG;
USE WAREHOUSE DECIDER_DEMO_WH;

CREATE OR REPLACE TABLE DEMO_TICKETS (
  TICKET_ID    NUMBER,
  TEXT         STRING,
  GEN_CATEGORY STRING,
  GEN_ESCALATE BOOLEAN
) COMMENT = 'Synthetic support tickets for the decider-2b triage board demo';

-- 5 categories x 10 batches x 20 tickets. Each batch gets a different tone and
-- ticket length so the text varies; about a quarter of tickets should need escalation.
INSERT INTO DEMO_TICKETS
WITH cats AS (
  SELECT column1 AS category, column2 AS meaning FROM VALUES
    ('billing', 'charges, invoices, plan prices, payment methods, taxes'),
    ('bug', 'something in the product is broken, erroring or behaving wrongly'),
    ('account_access', 'login, password, SSO, 2FA, locked or deactivated accounts, permissions'),
    ('refund', 'the customer explicitly wants money back'),
    ('feature_request', 'asking for something the product does not do yet')
),
styles AS (
  SELECT column1 AS batch, column2 AS style FROM VALUES
    (1, 'calm and polite, two to three sentences'),
    (2, 'frustrated, one or two sentences, some typos'),
    (3, 'very detailed, four to six sentences with specifics like dates and amounts'),
    (4, 'terse, under fifteen words, no punctuation'),
    (5, 'from an IT admin at a large company, formal'),
    (6, 'from a small business owner, casual'),
    (7, 'angry, threatening to cancel'),
    (8, 'confused, not sure what went wrong, rambling'),
    (9, 'non-native English speaker, simple words'),
    (10, 'mentions a second, minor issue in passing but the main issue is clear')
),
gen AS (
  SELECT c.category, s.batch,
    AI_COMPLETE(
      model => 'openai-gpt-5',
      prompt => 'Write 20 distinct customer support tickets for Lattice, a project-management SaaS app. '
             || 'Every ticket must clearly be about: ' || c.category || ' (' || c.meaning || '). '
             || 'Style: ' || s.style || '. '
             || 'Write in the customer''s voice; no greeting lines like "Dear support", no names or signatures, no ticket IDs. '
             || 'Mark about 5 of the 20 with escalate=true: ones with legal threats, outages for many users, security concerns, '
             || 'or large financial impact. The rest escalate=false.',
      response_format => {
        'type': 'json',
        'schema': {
          'type': 'object', 'additionalProperties': false,
          'properties': {
            'tickets': {
              'type': 'array',
              'items': {
                'type': 'object', 'additionalProperties': false,
                'properties': {'text': {'type': 'string'}, 'escalate': {'type': 'boolean'}},
                'required': ['text', 'escalate']
              }
            }
          },
          'required': ['tickets']
        }
      }
    ) AS out
  FROM cats c CROSS JOIN styles s
)
SELECT
  ROW_NUMBER() OVER (ORDER BY RANDOM(42)) AS ticket_id,
  t.value:text::STRING,
  g.category,
  t.value:escalate::BOOLEAN
FROM gen g, LATERAL FLATTEN(input => g.out:tickets) t;

SELECT GEN_CATEGORY, COUNT(*) n, COUNT_IF(GEN_ESCALATE) escalate FROM DEMO_TICKETS GROUP BY 1 ORDER BY 1;

# DTR Travel Entitlement Adjudicator

A travel scenario goes to two models with the same DTR Part I question and the same
governing excerpt:

- **strands-decider**, fine-tuned on DTR Part I (Passenger Movement), served from the
  Snowflake Model Registry as a real-time SPCS service (`DTR_DECIDER_HTTP`, 1x A10G), and
- **claude-sonnet-5** through `AI_COMPLETE` with a JSON schema for the answer.

Each side shows its categorical answer, calibrated confidence, full probability
distribution and the cited DTR paragraph. When a side abstains (a yes/no within 0.25 of
50%, or a top option below 50%; the same policy `strands_decider.dtr_eval` scores
DTR-Bench with), the answer is **not** issued: it goes to the human-review queue
(`DTR_REVIEW_QUEUE`), where a reviewer records the determination. "Replay DTR-Bench" streams
the benchmark through both sides for live accuracy, review rate, decisions/sec, latency and
cost per 1,000 decisions.

This is prototype decision support, not an official determination. DTR-Bench labels are
self-reviewed by the prototype author; DoD travel SME review is pending.

## Setup

1. Benchmark schema and DTR_ITEMS: `sql/00_setup.sql`, then `python/load_dtr.py`.
2. Model: `python/log_model.py` (version `SERVICE`), then
   `python/create_service_http.py SERVICE --name DTR_DECIDER_HTTP`.
3. App objects: `snow sql -f app/dtr-adjudicator/sql/01_dtr_app.sql -D database=DECIDER_BENCH -D schema=BENCH`
   (review queue, egress rule; create the PAT secret and EAI as described there).
4. Deploy with `app.yml` (Snowflake Apps), or run locally:

```bash
npm ci
SNOWFLAKE_CONNECTION_NAME=<connection> \
SNOWFLAKE_SECRET_DECIDER_PAT_SECRET_STRING=<pat> npm run dev
```

| Variable | Default |
| --- | --- |
| `DTR_ITEMS_TABLE` | `DECIDER_BENCH.BENCH.DTR_ITEMS` |
| `DTR_REVIEW_TABLE` | `DECIDER_BENCH.BENCH.DTR_REVIEW_QUEUE` |
| `DECIDER_SERVICE` | `DECIDER_BENCH.BENCH.DTR_DECIDER_HTTP` |
| `DECIDER_ENDPOINT` | looked up with `SHOW ENDPOINTS` |
| `LLM_MODEL` | `claude-sonnet-5` |
| `LLM_WAREHOUSE` | `DECIDER_BENCH_WH` |

The question catalog `lib/dtr_catalog.json` is generated, not hand-edited:
`python -m strands_decider.data.dtr --catalog lib/dtr_catalog.json` in strands-decider.

## Security notes

- Scenario text, model name and table names reach Snowflake only as bind variables
  (`IDENTIFIER(?)` for tables); the decide routes accept only catalog task ids and attach the
  catalog's DTR excerpt server-side, so a request can't supply its own "regulation".
- The decider PAT is read from a Snowflake secret on the server and sent only in the
  `Authorization` header; error responses are generic and details go to the server log.
- Review resolutions must be one of the task's option keys.

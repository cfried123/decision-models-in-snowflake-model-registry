-- decider-2b on JevBench: shared objects for the batch job and the service.
-- $SNOW -f sql/00_setup.sql

CREATE DATABASE IF NOT EXISTS <% database %>;
CREATE SCHEMA IF NOT EXISTS <% database %>.<% schema %>
  COMMENT = 'decider-2b (open weights) on JevBench public items';
USE SCHEMA <% database %>.<% schema %>;

-- Used only by the pre-flight and the service run, so its
-- metering is the benchmark's warehouse cost. Setup and analysis run elsewhere.
CREATE WAREHOUSE IF NOT EXISTS DECIDER_BENCH_WH
  WAREHOUSE_SIZE = 'XSMALL'
  WAREHOUSE_TYPE = 'STANDARD'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'decider-2b JevBench runs only';

CREATE STAGE IF NOT EXISTS DECIDER_BENCH_STAGE COMMENT = 'JevBench items and run artifacts';
CREATE IMAGE REPOSITORY IF NOT EXISTS DECIDER_BENCH_IMAGES;
CREATE FILE FORMAT IF NOT EXISTS JSONL_FF TYPE = JSON;

-- One row per benchmark statement. Timings come from QUERY_HISTORY by QUERY_ID.
CREATE TABLE IF NOT EXISTS RUN_LOG (
  RUN_ID         STRING,
  STEP           STRING,
  QUERY_ID       STRING,
  QUERY_TAG      STRING,
  WAREHOUSE_NAME STRING,
  N_ROWS         NUMBER,
  LOGGED_AT      TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
  NOTES          STRING
);

-- Raw service-function output: {"ANSWER_JSON": ..., "ELAPSED_MS": ..., "INPUT_TOKENS": ...}
CREATE TABLE IF NOT EXISTS DECIDER_ANSWERS (
  RUN_ID      STRING,
  ITEM_ID     STRING,
  RESULT      VARIANT,
  INSERTED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
);

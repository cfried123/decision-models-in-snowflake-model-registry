-- Load a run_batch job's Parquet output into DECIDER_ANSWERS, in the same RESULT shape the
-- service function returns, so DECIDER_DECISIONS, the cascade and the scoring script read
-- it unchanged. The job writes the input columns next to the outputs, so ITEM_ID comes back.
-- $SNOW -f sql/05_batch_load.sql -D "job=DECIDER_BATCH_V11B" -D "run_id=batch"
USE SCHEMA <% database %>.<% schema %>;
USE WAREHOUSE <% analysis_warehouse %>;

CREATE OR REPLACE TEMPORARY FILE FORMAT BATCH_PARQUET TYPE = PARQUET;

DELETE FROM DECIDER_ANSWERS WHERE RUN_ID = '<% run_id %>';

INSERT INTO DECIDER_ANSWERS (RUN_ID, ITEM_ID, RESULT)
SELECT '<% run_id %>', $1:ITEM_ID::STRING,
       OBJECT_CONSTRUCT('ANSWER_JSON', $1:ANSWER_JSON::STRING, 'ELAPSED_MS', $1:ELAPSED_MS::FLOAT,
                        'INPUT_TOKENS', $1:INPUT_TOKENS::NUMBER, 'RUNTIME_JSON', $1:RUNTIME_JSON::STRING)
FROM @DECIDER_BENCH_STAGE/batch/<% job %>/ (FILE_FORMAT => 'BATCH_PARQUET', PATTERN => '.*[.]parquet');

SELECT COUNT(*) AS ROWS_LOADED, COUNT(DISTINCT ITEM_ID) AS ITEMS
FROM DECIDER_ANSWERS WHERE RUN_ID = '<% run_id %>';

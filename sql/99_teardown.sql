-- Teardown. Metering history in ACCOUNT_USAGE survives these drops, so
-- sql/09_cost_accounting.sql still works afterwards.
-- $SNOW -f sql/99_teardown.sql
USE SCHEMA <% database %>.<% schema %>;

DROP SERVICE IF EXISTS DECIDER_2B_V11_SVC;
DROP COMPUTE POOL IF EXISTS DECIDER_BENCH_GPU_POOL;
DROP COMPUTE POOL IF EXISTS DECIDER_BENCH_BUILD_POOL;
DROP WAREHOUSE IF EXISTS DECIDER_BENCH_WH;
DROP EXTERNAL ACCESS INTEGRATION IF EXISTS DECIDER_BENCH_BUILD_EAI;
DROP NETWORK RULE IF EXISTS DECIDER_BENCH_BUILD_EGRESS;

-- Kept on purpose: the registered model (DECIDER_2B V11, about 3.8 GB), the serving image
-- in DECIDER_BENCH_IMAGES (redeploying reuses it and skips the build), and the tables.
-- To remove everything:
--   DROP MODEL IF EXISTS DECIDER_2B;
--   DROP IMAGE REPOSITORY IF EXISTS DECIDER_BENCH_IMAGES;
--   DROP SCHEMA IF EXISTS <% database %>.<% schema %>;

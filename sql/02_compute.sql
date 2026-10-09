-- The GPU pool. It starts suspended and resumes when a batch job (or the service)
-- is submitted; run_batch also builds its image on this pool.
-- $SNOW -f sql/02_compute.sql
USE SCHEMA <% database %>.<% schema %>;

-- 1x NVIDIA A10G (24 GB). decider-2b in bf16 needs about 4 GB of it.
CREATE COMPUTE POOL IF NOT EXISTS DECIDER_BENCH_GPU_POOL
  MIN_NODES = 1 MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_S
  AUTO_RESUME = TRUE
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'strands-decider DTR batch jobs and service';

SHOW COMPUTE POOLS LIKE 'DECIDER_BENCH%';

-- Only for the service path (python/create_service.py): a CPU pool for the serving
-- image build and the egress that build needs. The batch job needs neither.
-- $SNOW -f sql/02_service_build.sql
USE SCHEMA <% database %>.<% schema %>;

-- Builds the serving image; no GPU needed for that.
CREATE COMPUTE POOL IF NOT EXISTS DECIDER_BENCH_BUILD_POOL
  MIN_NODES = 1 MAX_NODES = 1
  INSTANCE_FAMILY = CPU_X64_S
  AUTO_RESUME = TRUE
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'decider-2b image build';

-- log_model(cuda_version="12.8") pins torch==2.10.0+cu128 and adds the PyTorch
-- wheel index, which serves files from download-r2.pytorch.org.
CREATE NETWORK RULE IF NOT EXISTS DECIDER_BENCH_BUILD_EGRESS
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('pypi.org', 'files.pythonhosted.org', 'download.pytorch.org', 'download-r2.pytorch.org')
  COMMENT = 'decider-2b image build: PyPI and PyTorch wheels only';

CREATE EXTERNAL ACCESS INTEGRATION IF NOT EXISTS DECIDER_BENCH_BUILD_EAI
  ALLOWED_NETWORK_RULES = (<% database %>.<% schema %>.DECIDER_BENCH_BUILD_EGRESS)
  ENABLED = TRUE
  COMMENT = 'decider-2b image build only; the running service gets no egress';

SHOW COMPUTE POOLS LIKE 'DECIDER_BENCH%';

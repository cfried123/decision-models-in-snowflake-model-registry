-- List prices used to turn metered credits into dollars.
-- Standard edition, On Demand, AWS US West (Oregon). Source: Snowflake Service
-- Consumption Table effective 2026-09-28, and docs.snowflake.com "Snowflake AI pricing".
-- To price another edition, update PLATFORM_CREDIT_USD (Enterprise is 3.00).
USE SCHEMA <% database %>.<% schema %>;

CREATE OR REPLACE TABLE PRICE_ASSUMPTIONS (
  ITEM           STRING,
  VALUE          NUMBER(12, 6),
  UNIT           STRING,
  BASIS          STRING,
  SOURCE         STRING,
  EFFECTIVE_DATE DATE
);

INSERT INTO PRICE_ASSUMPTIONS VALUES
  ('PLATFORM_CREDIT_USD',              2.00,  'USD per platform credit',
   'Standard edition, On Demand, AWS US West (Oregon)', 'Consumption Table, Table 2(a)', '2026-09-28'),
  ('AI_CREDIT_USD',                    2.00,  'USD per AI credit',
   'On Demand, global routing (account has CORTEX_ENABLED_CROSS_REGION = ANY_REGION); regional routing is 2.20; same for every edition',
   'Consumption Table, Table 2(b); Snowflake AI pricing docs', '2026-09-28'),
  ('GPU_NV_S_CREDITS_PER_HOUR',        0.57,  'platform credits per node-hour',
   'SPCS compute node GPU_NV_S (1x NVIDIA A10G)', 'Consumption Table, Table 1(f)', '2026-09-28'),
  ('CPU_X64_S_CREDITS_PER_HOUR',       0.11,  'platform credits per node-hour',
   'SPCS compute node CPU_X64_S (image build)', 'Consumption Table, Table 1(f)', '2026-09-28'),
  ('WH_XS_STANDARD_CREDITS_PER_HOUR',  1.00,  'platform credits per hour',
   'Standard warehouse, XS', 'Consumption Table, Table 1(a)', '2026-09-28'),
  ('AI_CLASSIFY_AI_CREDITS_PER_M_TOKENS', 1.62, 'AI credits per 1M tokens',
   'AI_CLASSIFY, Snowflake-managed compute', 'Consumption Table, Table 6', '2026-09-28'),
  ('CLOUD_SERVICES_CREDITS_PER_HOUR',  4.40,  'platform credits per hour',
   'Billed only above 10% of daily warehouse credits', 'Consumption Table, Cloud Services', '2026-09-28'),
  ('STORAGE_USD_PER_TB_MONTH',        23.00,  'USD per TB-month',
   'On Demand storage, AWS US West (Oregon)', 'Consumption Table, Table 3(a)', '2026-09-28'),
  ('MIN_COMPUTE_NODE_SECONDS',       300.00,  'seconds',
   'Minimum billed per compute node start or resume', 'Consumption Table, Compute', '2026-09-28'),
  ('MIN_WAREHOUSE_SECONDS',           60.00,  'seconds',
   'Minimum billed per warehouse start or resume', 'Consumption Table, Compute', '2026-09-28');

SELECT ITEM, VALUE, UNIT FROM PRICE_ASSUMPTIONS ORDER BY ITEM;

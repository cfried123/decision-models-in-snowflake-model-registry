# Run open-weight decision models on Snowflake

Code for the Snowflake Developers Blog post "Decisions Are All You Need: Run Open-Weight Decision Models on Snowflake" (link to come).

It runs [decider-2b](https://huggingface.co/Mapika/decider-2b), an open-weight decision model, from the Snowflake Model Registry as a batch inference job (`run_batch`) on one NVIDIA A10G, and scores the answers on [JevBench](https://github.com/fstandhartinger/jevbench)'s 231 public decisions with JevBench's own harness. The same model also runs as a service called from SQL. Throughput comes from query history and cost from Snowflake's metering views.

decider-2b is an independent open model; it isn't TypeSafe AI's Jev, and JevBench isn't affiliated with TypeSafe AI. Nothing here is a JevBench Score or a leaderboard entry.

## What it costs

The GPU pool bills for every minute its node is up, including the five minutes before auto-suspend. Run `sql/99_teardown.sql` when you're done.

At Standard edition list prices in AWS US West (Oregon), the batch job's GPU time while scoring the 231 decisions is $0.0057, or $0.025 per 1,000 decisions. Each job also holds the node for about 6 minutes to start, load and warm up the model and write its output (about $0.11), and the first job builds the image (about 7 minutes, $0.13). The service path builds its image on a CPU pool (about $0.05), and its GPU node bills for every hour it stays up, busy or not: 0.57 credits, or $1.14, an hour. Storing a model version (3.78 GB) is about $0.09 a month.

## Requirements

- Python 3.12 and the [Snowflake CLI](https://docs.snowflake.com/en/developer-guide/snowflake-cli/index) with a connection configured.
- A region with GPU_NV_S compute pools.
- A role that can:
  - create a database, a warehouse and a compute pool (`CREATE DATABASE`, `CREATE WAREHOUSE`, `CREATE COMPUTE POOL` on the account), and for the service path an external access integration (`CREATE INTEGRATION`);
  - create an image repository, a model and services (batch jobs are services) in the schema, and for the service path a network rule;
  - read `SNOWFLAKE.ACCOUNT_USAGE` for the cost queries.

  The scripts don't set a role; they use your connection's default.

## Layout

| Path | What it does |
|---|---|
| `python/decider_model.py` | The Model Registry `CustomModel` wrapper: one `system_one` method that takes JevBench's request and returns decider's response |
| `python/log_model_batch.py` | Register the weights for batch jobs as version `BATCH`: the same wrapper, with ranges for the helper libraries that the batch base image needs |
| `python/bench_batch.py` | Run a version over the 231 items with `run_batch` on the GPU pool, save the job's output and score it with JevBench's harness |
| `python/log_model.py`, `python/create_service.py` | The service path: register version `SERVICE` with every package pinned, then build the image on a CPU pool and start the service |
| `python/decider_model_batched.py`, `python/replay_batch_plans.py` | The batched serving path I also tried (slower on these items), and an offline replay of its batch plans |
| `python/load_jevbench.py` | Load the 231 public items into `JEVBENCH_ITEMS` |
| `python/smoke_test_local.py` | Run the wrapper in-process on six items and check JevBench parses the output, before any GPU starts |
| `python/score_with_jevbench.py` | Score the service run's stored answers with JevBench's harness at commit `bb05a335` |
| `python/bench_config.py` | Database and schema for the Python scripts |
| `sql/00`–`09` | Setup, prices, the GPU pool (`02_compute`), the service's build pool and egress (`02_service_build`), views, the service pre-flight, loading a batch job's output (`05_batch_load`), the service run, throughput and cost |
| `sql/99_teardown.sql` | Drop the pools, warehouse, service and integration |
| `results/` | The scored outputs behind the post |

## Run it

Set the names once. The SQL scripts take them as variables, and the Python scripts read the environment:

```sh
export SNOWFLAKE_CONNECTION_NAME=<connection>
export DECIDER_BENCH_DATABASE=DECIDER_BENCH DECIDER_BENCH_SCHEMA=BENCH
export SNOW="snow sql -c $SNOWFLAKE_CONNECTION_NAME -D database=$DECIDER_BENCH_DATABASE -D schema=$DECIDER_BENCH_SCHEMA -D analysis_warehouse=<your existing warehouse>"
```

`analysis_warehouse` runs logging and reporting statements, so the benchmark's own warehouse, `DECIDER_BENCH_WH`, meters only the runs.

Fetch the pinned inputs and install the local environment:

```sh
mkdir -p .cache/jevbench models
curl -sL https://codeload.github.com/fstandhartinger/jevbench/tar.gz/bb05a335bc809e61b20c0f745d25499a82b326fc \
  | tar xz --strip-components=1 -C .cache/jevbench
python3.12 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
.venv/bin/hf download Mapika/decider-2b --revision 533964dae8be954c5b5e19fa4948e48408094c1e \
  --local-dir models/decider-2b
```

Then run the batch job, in order:

```sh
$SNOW -f sql/00_setup.sql
$SNOW -f sql/01_price_assumptions.sql
.venv/bin/python python/load_jevbench.py
.venv/bin/python python/smoke_test_local.py
$SNOW -f sql/02_compute.sql
.venv/bin/python python/log_model_batch.py BATCH
.venv/bin/python python/bench_batch.py BATCH batch 32   # about 13 minutes the first time, with the image build
$SNOW -f sql/03_views.sql
$SNOW -f sql/05_batch_load.sql -D job=DECIDER_BATCH -D run_id=batch
$SNOW -f sql/99_teardown.sql
```

`bench_batch.py` scores the job's answers itself and writes them to `runs/batch/`. `05_batch_load.sql` puts them in `DECIDER_ANSWERS`, where the views and the SQL reports read them.

The service path:

```sh
$SNOW -f sql/02_service_build.sql
.venv/bin/python python/log_model.py
.venv/bin/python python/create_service.py      # about 15 minutes to a ready container
$SNOW -f sql/03_views.sql
$SNOW -f sql/04_preflight.sql
$SNOW -f sql/05_service_decider_only.sql
.venv/bin/python python/score_with_jevbench.py
$SNOW -f sql/08_report_throughput.sql
$SNOW -f sql/99_teardown.sql
```

`sql/08_report_throughput.sql` reads `INFORMATION_SCHEMA`, so run it before teardown. `sql/09_cost_accounting.sql` reads `ACCOUNT_USAGE`, which lags by up to a few hours, and works after teardown.

## Third-party code and data

- JevBench (harness and the public decisions) is MIT-licensed, by Florian Standhartinger and contributors. It's downloaded at a pinned commit, not vendored; see `third_party/jevbench/`.
- decider-2b's weights and the `decider-ai` package are Apache-2.0, by their author. The weights are downloaded at a pinned revision and their SHA-256 is checked before logging.

## License

Apache-2.0. See `LICENSE`.

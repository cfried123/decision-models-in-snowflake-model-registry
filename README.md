# Run open-weight decision models on Snowflake

Code for the Snowflake Developers Blog post "Decisions Are All You Need: Run Open-Weight Decision Models on Snowflake" (link to come).

It serves [decider-2b](https://huggingface.co/Mapika/decider-2b) V11, an open-weight decision model, from the Snowflake Model Registry on Snowpark Container Services (one NVIDIA A10G), calls it from SQL, and scores the answers on [JevBench](https://github.com/fstandhartinger/jevbench)'s 231 public decisions with JevBench's own harness. A second run sends the rows decider-2b is least sure of to `AI_CLASSIFY`, and a sweep replays that cascade at every confidence threshold up to 0.9. Throughput comes from query history and cost from Snowflake's metering views.

decider-2b is an independent open model; it isn't TypeSafe AI's Jev, and JevBench isn't affiliated with TypeSafe AI. Nothing here is a JevBench Score or a leaderboard entry.

## What it costs

The GPU pool bills for every minute its node is up, whether or not a query is running. Run `sql/99_teardown.sql` when you're done.

At Standard edition list prices in AWS US West (Oregon), the whole benchmark (setup, image build, both runs) came to $0.72, and the threshold sweep's extra `AI_CLASSIFY` calls to $0.21. Storing the model version (3.78 GB) is about $0.09 a month.

## Requirements

- Python 3.12 and the [Snowflake CLI](https://docs.snowflake.com/en/developer-guide/snowflake-cli/index) with a connection configured.
- A region with GPU_NV_S compute pools and `AI_CLASSIFY`.
- A role that can:
  - create a database, a warehouse, compute pools and an external access integration (`CREATE DATABASE`, `CREATE WAREHOUSE`, `CREATE COMPUTE POOL`, `CREATE INTEGRATION` on the account);
  - create a network rule, an image repository, a model and a service in the schema;
  - call Cortex AI functions (the `SNOWFLAKE.CORTEX_USER` database role);
  - read `SNOWFLAKE.ACCOUNT_USAGE` for the cost queries.

  The scripts don't set a role; they use your connection's default.

## Layout

| Path | What it does |
|---|---|
| `python/decider_model.py` | The Model Registry `CustomModel` wrapper: one `system_one` method that takes JevBench's request and returns decider's response |
| `python/log_model.py`, `python/create_service.py` | Register the weights with pinned dependencies, then build the image and start the service |
| `python/load_jevbench.py` | Load the 231 public items into `JEVBENCH_ITEMS` |
| `python/smoke_test_local.py` | Run the wrapper in-process on six items and check JevBench parses the output, before any GPU starts |
| `python/score_with_jevbench.py`, `python/threshold_sweep.py` | Score stored answers with JevBench's harness at commit `bb05a335` |
| `python/plot_threshold_sweep.py` | Draw the sweep chart from `runs/sweep/report.json`, or `results/sweep_report.json` if you haven't run the sweep |
| `python/bench_config.py` | Database and schema for the Python scripts |
| `sql/00`–`10` | Setup, prices, compute, views, the runs, the cascade, throughput, cost and the sweep |
| `sql/99_teardown.sql` | Drop the service, pools, warehouse and integration |
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
  --local-dir models/decider-2b-v11
```

Then, in order:

```sh
$SNOW -f sql/00_setup.sql
$SNOW -f sql/01_price_assumptions.sql
.venv/bin/python python/load_jevbench.py
.venv/bin/python python/smoke_test_local.py
$SNOW -f sql/02_compute.sql
.venv/bin/python python/log_model.py
.venv/bin/python python/create_service.py      # about 15 minutes to a ready container
$SNOW -f sql/03_views.sql
$SNOW -f sql/04_preflight.sql
$SNOW -f sql/05_run1_decider_only.sql
$SNOW -f sql/06_token_precount.sql
$SNOW -f sql/07_run2_cascade.sql
.venv/bin/python python/score_with_jevbench.py
$SNOW -f sql/99_teardown.sql
```

`sql/08_report_throughput.sql` reads `INFORMATION_SCHEMA` and works right away. `sql/09_cost_accounting.sql` reads `ACCOUNT_USAGE`, which lags by up to a few hours, and works after teardown.

For the threshold sweep, after Run 2:

```sh
$SNOW -f sql/10_threshold_sweep.sql
.venv/bin/python python/threshold_sweep.py
uv run --no-project --with matplotlib==3.10.1 python python/plot_threshold_sweep.py sweep.png
```

## Third-party code and data

- JevBench (harness and the public decisions) is MIT-licensed, by Florian Standhartinger and contributors. It's downloaded at a pinned commit, not vendored; see `third_party/jevbench/`.
- decider-2b's weights and the `decider-ai` package are Apache-2.0, by their author. The weights are downloaded at a pinned revision and their SHA-256 is checked before logging.

## License

Apache-2.0. See `LICENSE`.

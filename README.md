# Run Jev-class decision models on Snowflake

Code for the Snowflake Developers Blog post [Decisions Are All You Need: Run Jev-Class Decision Models on Snowflake](https://www.snowflake.com/en/developers/blog/decisions-are-all-you-need-run-jev-class-decision-models-on-snowflake/).

It runs [decider-2b](https://huggingface.co/Mapika/decider-2b), an open-weight decision model, from the Snowflake Model Registry as a batch inference job (`run_batch`) on one NVIDIA A10G. JevBench's own harness scores the answers on [JevBench](https://github.com/fstandhartinger/jevbench)'s 231 public decisions.

The same model also runs as a real-time inference service, called over its REST endpoint and from SQL. Both paths scale across GPUs. A Snowflake App Runtime app in `app/triage-board/` uses the real-time service to triage support tickets side by side with a frontier LLM.

Throughput comes from query history and client-side timers. Cost comes from Snowflake's metering views.

decider-2b is an independent open model. It isn't TypeSafe AI's Jev, and JevBench isn't affiliated with TypeSafe AI. Nothing here is a JevBench Score or a leaderboard entry.

## What it costs

The GPU pool bills for every minute its node is up, including the five minutes before auto-suspend. Run `sql/99_teardown.sql` when you're done.

Prices below are Standard edition list prices in AWS US West (Oregon).

- **Batch job.** GPU time while scoring the 231 decisions is $0.0057, or $0.025 per 1,000 decisions. Each job also holds the node for about 6 minutes to start, load and warm up the model and write its output (about $0.11). The first job builds the image (about 7 minutes, $0.13).
- **Service.** The image builds on a CPU pool (about $0.05). The GPU node bills for every hour it stays up, busy or not: 0.57 credits, or $1.14, an hour.
- **REST.** The 231 decisions took 17.7 s from 8 concurrent clients ($0.024 per 1,000 in GPU node time) and 37.4 s from one ($0.051 per 1,000).
- **Storage.** A model version (3.78 GB) is about $0.09 a month.
- **Triage board.** The app keeps its own decider-2b service on one GPU_NV_S node ($1.14 an hour while it's up) and runs `AI_COMPLETE` on an XS warehouse. In our 25-ticket run with one request in flight per side, decider-2b cost $0.049 per 1,000 tickets in GPU time and `claude-sonnet-5` cost $4.98 per 1,000 in tokens and warehouse time. Its README has the teardown.

## Requirements

- Python 3.12 and the [Snowflake CLI](https://docs.snowflake.com/en/developer-guide/snowflake-cli/index) with a connection configured.
- A region with GPU_NV_S compute pools.
- A role that can:
  - create a database, a warehouse and a compute pool (`CREATE DATABASE`, `CREATE WAREHOUSE` and `CREATE COMPUTE POOL` on the account), plus an external access integration for the service path (`CREATE INTEGRATION`);
  - create an image repository, a model and services in the schema (batch jobs are services), plus a network rule for the service path;
  - read `SNOWFLAKE.ACCOUNT_USAGE` for the cost queries.

  The scripts don't set a role. They use your connection's default.
- For the triage board: Node.js and npm, Snowflake CLI 3.26 or later (for `snow app deploy`), Snowflake App Runtime, and `AI_COMPLETE` access to `claude-sonnet-5`.

## Layout

| Path | What it does |
|---|---|
| `python/decider_model.py` | The Model Registry `CustomModel` wrapper. Its one method, `system_one`, takes JevBench's request and returns decider's response |
| `python/log_model_batch.py` | Registers the weights for batch jobs as version `BATCH`: the same wrapper, with version ranges for the helper libraries the batch base image needs |
| `python/bench_batch.py` | Runs a version over the 231 items with `run_batch` on the GPU pool, saves the job's output and scores it with JevBench's harness |
| `python/log_model.py`, `python/create_service.py` | The service path: registers version `SERVICE` with every package pinned, builds the image on a CPU pool and starts the service |
| `python/create_service_http.py`, `python/bench_http.py` | Real-time inference over REST. Starts the same version as `DECIDER_2B_HTTP` with a public endpoint, then times the 231 items one request at a time, from 8 concurrent clients, and as 20 single-row SQL calls. Scores the answers with JevBench's harness and prices the GPU time |
| `python/bench_scaling.py` | Batch throughput as you add model copies per GPU (`num_workers`) or nodes (`replicas`). Repeats the 231 items and runs one job per config. Records scoring time, decisions per second, when each worker joined, and cost at list price |
| `python/bench_http_sweep.py` | Real-time inference under load. Against a service with several instances, runs 1, 4, 16 and 64 closed-loop REST clients for 30 seconds each. Records decisions per second, p50/p95/p99 latency, errors, how requests spread across instances, and cost at list price |
| `python/decider_model_batched.py`, `python/replay_batch_plans.py` | The batched serving path I also tried (slower on these items), and an offline replay of its batch plans |
| `python/load_jevbench.py` | Loads the 231 public items into `JEVBENCH_ITEMS` |
| `python/smoke_test_local.py` | Runs the wrapper in-process on six items and checks that JevBench parses the output, before any GPU starts |
| `python/score_with_jevbench.py` | Scores the service run's stored answers with JevBench's harness at commit `bb05a335` |
| `python/bench_config.py`, `python/snowpark_session.py` | The database and schema for the Python scripts, and a Snowpark session from your CLI connection |
| `python/requirements.txt` | The local Python environment |
| `sql/00`–`09` | Setup, prices, the GPU pool (`02_compute`), the service's build pool and egress (`02_service_build`), views, the service pre-flight, loading a batch job's output (`05_batch_load`), the service run, throughput and cost |
| `sql/99_teardown.sql` | Drops the pools, warehouse, services and integration |
| `results/` | The scored outputs and summaries behind the post |
| `app/triage-board/` | A Snowflake App Runtime (Next.js) app. It triages the same support tickets live with decider-2b's REST endpoint and with `claude-sonnet-5` through `AI_COMPLETE`, and shows elapsed time, decisions per second, latency and cost side by side. Setup and deploy steps are in its README |

## Set up

### Configure

Set the names once. The SQL scripts take them as variables, and the Python scripts read them from the environment:

```sh
export SNOWFLAKE_CONNECTION_NAME=<connection>
export DECIDER_BENCH_DATABASE=DECIDER_BENCH DECIDER_BENCH_SCHEMA=BENCH
export SNOW="snow sql -c $SNOWFLAKE_CONNECTION_NAME -D database=$DECIDER_BENCH_DATABASE -D schema=$DECIDER_BENCH_SCHEMA -D analysis_warehouse=<your existing warehouse>"
```

`analysis_warehouse` runs the logging and reporting statements. That way the benchmark's own warehouse, `DECIDER_BENCH_WH`, meters only the runs.

### Install

Fetch the pinned inputs and set up the local environment:

```sh
mkdir -p .cache/jevbench models
curl -sL https://codeload.github.com/fstandhartinger/jevbench/tar.gz/bb05a335bc809e61b20c0f745d25499a82b326fc \
  | tar xz --strip-components=1 -C .cache/jevbench
python3.12 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
.venv/bin/hf download Mapika/decider-2b --revision 533964dae8be954c5b5e19fa4948e48408094c1e \
  --local-dir models/decider-2b
```

## decider-2b performance on JevBench with Snowflake ML

The same 231 decisions run once as a batch inference job and once through a real-time inference service. Both runs scored 175 of 231 correct (48 of 48 easy, 64 of 72 standard, 63 of 111 hard).

### Batch inference job

Run these in order:

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

`bench_batch.py` scores the job's answers and writes them to `runs/batch/`. `05_batch_load.sql` loads them into `DECIDER_ANSWERS`, which the views and SQL reports read.

### Real-time inference service

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

`sql/08_report_throughput.sql` reads `INFORMATION_SCHEMA`, so run it before teardown. `sql/09_cost_accounting.sql` reads `ACCOUNT_USAGE`, which can lag by a few hours, so it works after teardown.

### Real-time inference over REST

Run this after the service run, since it compares its answers with the `service` run:

```sh
$SNOW -f sql/02_service_build.sql
.venv/bin/python python/create_service_http.py SERVICE    # about 15 minutes to a ready endpoint
DECIDER_BENCH_PAT=<programmatic access token> .venv/bin/python python/bench_http.py
$SNOW -f sql/99_teardown.sql
```

The endpoint takes `Authorization: Snowflake Token="<PAT>"`. The PAT's role needs the service role `DECIDER_2B_HTTP!ALL_ENDPOINTS_USAGE`, which the owner role has. In our runs, a session token from the Python connector got HTTP 500 from the ingress.

`bench_http.py` writes `runs/http/` and `runs/rest/`. `results/rest_summary.json` is our run. Client times include the network between your machine and the endpoint.

## Scaling decision models

### Batch inference across GPUs

This measures throughput and cost only. The answers aren't scored. Start the nodes before the job so every replica is up when scoring begins:

```sh
$SNOW -q "CREATE COMPUTE POOL IF NOT EXISTS DECIDER_BENCH_GPU_POOL_S4 MIN_NODES = 4 MAX_NODES = 4 INSTANCE_FAMILY = GPU_NV_S AUTO_RESUME = TRUE AUTO_SUSPEND_SECS = 300"
# wait until SHOW COMPUTE POOLS LIKE 'DECIDER_BENCH_GPU_POOL_S4' shows 4 idle nodes
.venv/bin/python python/bench_scaling.py BATCH --pool DECIDER_BENCH_GPU_POOL_S4 --copies 217 out4:4:1
$SNOW -f sql/99_teardown.sql
```

`--copies 217` repeats the 231 items into 50,127 rows. In our run, four A10G nodes scored 50.2 decisions per second, against 13.8 on one. The job took 22.5 minutes for $1.71 at list price, or $0.034 per 1,000 decisions including start-up.

`results/scaling_summary.json` has that run and the earlier 4,620-row runs on one pool (one to three model copies per GPU, one or two nodes).

### Real-time inference across instances

This measures throughput and latency only. The answers aren't scored. It uses the same four-node pool:

```sh
$SNOW -f sql/02_service_build.sql
$SNOW -q "CREATE COMPUTE POOL IF NOT EXISTS DECIDER_BENCH_GPU_POOL_S4 MIN_NODES = 4 MAX_NODES = 4 INSTANCE_FAMILY = GPU_NV_S AUTO_RESUME = TRUE AUTO_SUSPEND_SECS = 300"
.venv/bin/python python/create_service_http.py SERVICE --name DECIDER_2B_HTTP4 --pool DECIDER_BENCH_GPU_POOL_S4 --instances 4
DECIDER_BENCH_PAT=<programmatic access token> .venv/bin/python python/bench_http_sweep.py DECIDER_2B_HTTP4 4 1 4 16 64
$SNOW -f sql/99_teardown.sql
```

In our run there were no errors. Four instances answered in a median of 115 ms with one client and 126 ms with four. They reached 40.2 decisions per second at 16 clients (median 356 ms) and 47.5 at 64 (median 1.1 s), or $0.027 per 1,000 decisions at list price. Requests spread evenly across the four instances. `results/rest_sweep_summary.json` is that run.

## Deploy in App Runtime to Get Work Done Fast

Run this after the service path has registered version `SERVICE`. You generate the tickets, start a one-instance service with a public endpoint, create the secret and external access integration, and deploy with `snow app deploy`. The steps are in [`app/triage-board/README.md`](app/triage-board/README.md).

In our 25-ticket run with one request in flight per side, decider-2b made 6.5 decisions per second (median 115 ms). `claude-sonnet-5` made 0.5 per second (median 2.17 s).

## Third-party code and data

- JevBench (the harness and the public decisions) is MIT-licensed, by Florian Standhartinger and contributors. It's downloaded at a pinned commit, not vendored. See `third_party/jevbench/`.
- decider-2b's weights and the `decider-ai` package are Apache-2.0, by their author. The weights are downloaded at a pinned revision, and their SHA-256 is checked before logging.

## License

Apache-2.0. See `LICENSE`.

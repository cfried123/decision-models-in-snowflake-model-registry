"""Where the benchmark's objects live. Override with environment variables to
match the -D database=... -D schema=... values you pass to the SQL scripts."""
import os

DATABASE = os.environ.get("DECIDER_BENCH_DATABASE", "DECIDER_BENCH")
SCHEMA = os.environ.get("DECIDER_BENCH_SCHEMA", "BENCH")
FQ_SCHEMA = f"{DATABASE}.{SCHEMA}"

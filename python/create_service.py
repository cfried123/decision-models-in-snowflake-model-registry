"""Deploy DECIDER_2B version SERVICE as the SPCS service DECIDER_2B_SVC (1x A10G, one instance).

Runs in the background; poll with DESCRIBE SERVICE. SQL calls it as
DECIDER_2B_SVC!SYSTEM_ONE(STATE_JSON, QUESTIONS_JSON).

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/create_service.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from snowflake.ml.registry import Registry  # noqa: E402

from bench_config import DATABASE, FQ_SCHEMA, SCHEMA  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402


def main():
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    mv = reg.get_model("DECIDER_2B").version("SERVICE")
    t0 = time.time()
    print("creating service ...", flush=True)
    mv.create_service(
        service_name="DECIDER_2B_SVC",
        service_compute_pool="DECIDER_BENCH_GPU_POOL",
        image_build_compute_pool="DECIDER_BENCH_BUILD_POOL",
        image_repo=f"{FQ_SCHEMA}.DECIDER_BENCH_IMAGES",
        build_external_access_integrations=["DECIDER_BENCH_BUILD_EAI"],
        ingress_enabled=False,        # SQL only; no public endpoint
        min_instances=1,
        max_instances=1,
        gpu_requests="1",
        num_workers=1,                # one model copy on the one GPU
        max_batch_rows=32,            # rows per service-function request (the minimum allowed)
    )
    print(f"service created in {time.time() - t0:.0f}s", flush=True)
    session.close()


if __name__ == "__main__":
    main()

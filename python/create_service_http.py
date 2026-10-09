"""Deploy a DECIDER_2B version as a real-time inference service with a public HTTPS
endpoint: the same settings as create_service.py (1x A10G per instance), plus ingress.

Arguments: version (default SERVICE), and optionally --name, --pool and --instances.
With --instances N the service runs N instances (min = max = N), one per GPU node, so
the pool needs N nodes. The endpoint's URL comes from SHOW ENDPOINTS IN SERVICE <name>;
python/bench_http.py and python/bench_http_sweep.py call it.

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/create_service_http.py [version]
    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/create_service_http.py SERVICE \\
        --name DECIDER_2B_HTTP4 --pool DECIDER_BENCH_GPU_POOL_S4 --instances 4
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from snowflake.ml.registry import Registry  # noqa: E402

from bench_config import DATABASE, FQ_SCHEMA, MODEL_NAME, SCHEMA  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402


def main(version, name, pool, instances):
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    mv = reg.get_model(MODEL_NAME).version(version)
    t0 = time.time()
    print(f"creating {name} from version {version}, {instances} instance(s) on {pool} ...", flush=True)
    mv.create_service(
        service_name=name,
        service_compute_pool=pool,
        image_build_compute_pool="DECIDER_BENCH_BUILD_POOL",
        image_repo=f"{FQ_SCHEMA}.DECIDER_BENCH_IMAGES",
        build_external_access_integrations=["DECIDER_BENCH_BUILD_EAI"],
        ingress_enabled=True,         # public HTTPS endpoint; requires BIND SERVICE ENDPOINT
        min_instances=instances,
        max_instances=instances,
        gpu_requests="1",
        num_workers=1,
        max_batch_rows=32,
    )
    print(f"service created in {time.time() - t0:.0f}s", flush=True)
    session.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("version", nargs="?", default="SERVICE")
    p.add_argument("--name", default="DTR_DECIDER_HTTP")
    p.add_argument("--pool", default="DECIDER_BENCH_GPU_POOL")
    p.add_argument("--instances", type=int, default=1)
    a = p.parse_args()
    main(a.version, a.name, a.pool, a.instances)

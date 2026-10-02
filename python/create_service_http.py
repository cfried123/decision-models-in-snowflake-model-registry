"""Deploy a DECIDER_2B version as DECIDER_2B_HTTP: the same settings as create_service.py
(1x A10G, one instance), plus a public HTTPS endpoint for real-time inference.

Takes the version name (default SERVICE). The endpoint's URL comes from
SHOW ENDPOINTS IN SERVICE DECIDER_2B_HTTP; python/bench_http.py calls it.

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/create_service_http.py [version]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from snowflake.ml.registry import Registry  # noqa: E402

from bench_config import DATABASE, FQ_SCHEMA, SCHEMA  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402


def main(version):
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    mv = reg.get_model("DECIDER_2B").version(version)
    t0 = time.time()
    print(f"creating DECIDER_2B_HTTP from version {version} ...", flush=True)
    mv.create_service(
        service_name="DECIDER_2B_HTTP",
        service_compute_pool="DECIDER_BENCH_GPU_POOL",
        image_build_compute_pool="DECIDER_BENCH_BUILD_POOL",
        image_repo=f"{FQ_SCHEMA}.DECIDER_BENCH_IMAGES",
        build_external_access_integrations=["DECIDER_BENCH_BUILD_EAI"],
        ingress_enabled=True,         # public HTTPS endpoint; requires BIND SERVICE ENDPOINT
        min_instances=1,
        max_instances=1,
        gpu_requests="1",
        num_workers=1,
        max_batch_rows=32,
    )
    print(f"service created in {time.time() - t0:.0f}s", flush=True)
    session.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "SERVICE")

# DTR typed decisions on Snowflake: demo narrative

A walkthrough for presenting the DTR Part I (Passenger Movement) prototype as a Snowflake
product demo. It covers what was built, which Snowflake feature each step shows, the trade-offs
we made, and what to do next. Numbers come from runs on a Snowflake trial account (AWS us-east-2).
Two model families were used: strands-decider (Qwen3.5-2B + LoRA) and Laya (421M ModernBERT-large,
from the `cfried123/laya` fork).

## 1. The business problem (2 min)

Travel approvers must answer the same few questions over and over, citing the governing DTR paragraph each time:
*which class of service is authorized?*, *is this City Pair fare exception allowed?*,
*how much compliance risk does this voucher carry?* Each answer is a **typed decision**
(one option, yes/no, or an ordinal score), not an essay. We have to show three things:

* **Speed and cost:** cents per thousand decisions instead of dollars.
* **Calibration:** the model knows when it doesn't know.
* **Governance:** uncertain cases go to a human. The data never leaves the Snowflake account.

**Talking point:** a generative LLM is the wrong tool for a high-volume, auditable yes/no decision.
A typed decision model returns a probability for every option. That makes abstention a
policy you set (a threshold), not a prompt you hope the LLM follows.

## 2. Data: from regulation to training rows (3 min)

* Source: DTR Part I chapters (Passenger Movement). We started from DTR Part II and switched,
  because Part II covers cargo, not travel entitlements.
* A recipe (`strands-decider` `data/dtr.py`) generates scenario + cited DTR excerpt + question
  for 10 task families, written in all three answer shapes (`choice`, `noul`, `score`). Ambiguous cases
  (where a decisive fact is missing) are deliberately included, labelled "should abstain".
* **Out-of-domain holdout:** Chapter 106 (rental cars) and Chapter 107 (ship travel) are kept out of training.
* **DTR-Bench:** 142 items with different wording, self-reviewed against the cited text. SME review is pending.
* The same rows feed both model families: `python/laya_dtr/dtr_data.py` converts them to Laya's
  `{state, questions, expected|gold}` schema. Each pair of opposite-label ambiguous rows becomes a single 50/50 `gold` target.

**Snowflake features:** stages, `DTR_ITEMS` table (`load_dtr.py`), SQL views over the benchmark.

**Trade-off:** synthetic, template-based data let us build a full pipeline in days. The cost is
that in-domain scores flatter the models (shared templates). That's why the held-out chapters exist and why we report them separately.

## 3. Training on Snowflake GPUs (4 min)

* Both fine-tunes run as **Snowflake ML Jobs** (`jobs.submit_directory`) on a `GPU_NV_S` (A10G) pool.
  The code, base weights and data are staged as one payload. Packages come from Snowflake's **managed PyPI
  repository**, so no external access integration is needed (trial accounts can't create one).
* strands-decider: LoRA, `head_lr` > `lr`, shuffled options, EMA, starting from v21. One epoch ≈ 14 min.
* Laya: the fork's own `laya.train.finetune` (RLCD loss, shuffled options for choice/score,
  `max_len` 768). Temperatures and abstention thresholds are fitted on a calibration split, and
  before/after metrics are taken on the Ch.106/107 holdout inside the job. LoRA/EMA don't apply: Laya
  fully fine-tunes a 421M encoder, roughly 5× smaller than the decider.
* We first ran training on the VM's CPU, then moved it to Snowflake GPUs: a full epoch
  took minutes there, versus about 10 hours on CPU.

**Talking point:** training happens next to the governed data, on per-second-billed GPUs that
suspend on their own. You don't need a separate ML platform account.

## 4. Model Registry: every version is a governed object (3 min)

* `DTR_DECIDER` versions: V21 base and the DTR fine-tune. `LAYA_DTR` versions: BASE (zero-shot) and FT.
* Each version is a pickle-free checkpoint (safetensors), stored as a registry artifact
  with pinned pip requirements, so the serving container never downloads weights or code.
* Both families share one `system_one` contract (STATE_JSON + QUESTIONS_JSON → ANSWER_JSON,
  ELAPSED_MS, INPUT_TOKENS, RUNTIME_JSON). Swapping a model is a version change, not a code change.

## 5. Serving: Snowpark Container Services + SQL service functions (3 min)

* `mv.create_service(...)` builds the image on a build pool and runs it on the GPU pool.
  Every service gets a SQL function (`<service>!SYSTEM_ONE`) and, if wanted, an HTTPS endpoint.
* Measured on strands-decider: 68 ms per decision on the GPU, 7.3 decisions/s one at a time, 14.2/s with 8
  clients. At list price ($2/credit, 0.57 credits/h for GPU_NV_S) that is about $0.02–0.04 per 1K decisions of
  GPU time. The SQL warehouse adds about $0.04 per 1K.
* Laya BASE and FT run as SPCS services on a **CPU** pool (`LAYA_CPU_POOL`, `CPU_X64_M`), registered from the
  same `LAYA_DTR` model with a `numpy==2.2.6` pin (the CPU inference image's pyarrow needs NumPy 2).
  Measured through `LAYA_DTR_*_SVC!SYSTEM_ONE` over all 142 DTR-Bench rows: about 510–610 ms of model time
  per decision, about 1.6–1.9 decisions/s for one set-based SQL query. The same checkpoints score
  about 28 ms per decision on the GPU_NV_S training job. Scores through the CPU service match the GPU job
  (FT 94.4% vs 93.7% strict, differences are floating-point at the abstention boundary).
* Tradeoff: the CPU pool is cheaper per hour but roughly 20× slower per decision, so the GPU pool
  costs less per 1K decisions at any sustained volume. CPU is the right choice only for an
  always-on, low-traffic demo endpoint.
* Known gap: the registry's `run_batch` image conflicts with the transformers version the models need,
  so bulk scoring goes through the service from SQL.

## 6. The Streamlit in Snowflake app (5 min, the live part)

`DTR_MODEL_EXPLORER` (`streamlit/dtr_model_explorer/`, deployed with `sql/10_model_explorer.sql`):

1. **Ask one version:** pick a registry version, a determination, a governing DTR paragraph, a scenario
   (or a DTR-Bench sample). The app shows the typed answer, every option's probability, latency and $/1K.
2. **Compare:** run any set of versions side by side, plus a closed-source model through `AI_COMPLETE`
   (Claude by default). Every model gets the same question and the same excerpt.
3. **Abstention → review queue:** an answer inside the uncertainty band is never issued. It is written to
   `DTR_REVIEW_QUEUE`, where a reviewer resolves it (with their user name and a timestamp).
4. **Leaderboard:** DTR-Bench scores for every version from `DTR_MODEL_SCORES`.
5. **Registry & cost:** versions with comments from `SHOW VERSIONS`, service status, and
   resume/suspend controls, so the presenter controls spend live.

The scenario text is always passed as a bind variable. Service names come from an allow-list table (`MODEL_ENDPOINTS`).
Nothing leaves the account: models through service functions, the LLM through Cortex.

**Suggested live script:** (a) show a clear-cut class-of-service case; every fine-tune answers
confidently. (b) Show a booking-policy case with a missing fact; the fine-tunes abstain and it goes to
the queue. (c) Resolve it in the Review tab. (d) Open the leaderboard and the cost tab.

## 7. Results

| Model (DTR-Bench, 142 items) | Strict acc. | Forced acc. | Coverage | Answered acc. | Abstain recall | ECE | Held-out strict (n=22) |
|---|---|---|---|---|---|---|---|
| strands-decider v21 (base) | 48.6% | 77.1% | 58.0% | 84.2% | 45.5% | 0.135 | 50.0% |
| strands-decider DTR fine-tune | 98.6% | 98.5% | 98.5% | 100% | 100% | 0.027 | 90.9% |
| Laya base (zero-shot) | 30.3% | 53.4% | 35.9% | 76.6% | 63.6% | 0.114 | 50.0% |
| Laya DTR fine-tune | 93.7% | 98.5% | 96.2% | 99.2% | 72.7% | 0.015 | 63.6% (forced 90.0%) |
| Claude via AI_COMPLETE | not run: trial accounts block Cortex AI functions | | | | | | |

**Strict accuracy:** an abstention counts as correct only on the ambiguous items. **Forced accuracy:** the top option is scored as if the model had to answer.

**Laya, read honestly:**
* Fine-tuning took 465 s on one A10G (2 epochs, 2,648 train rows), and the whole ML Job took 492 s.
* Scoring is about 28 ms per decision on the GPU, against about 240 ms on an 8-thread CPU.
* In-domain, the fine-tune nearly matches the strands-decider fine-tune.
* On the held-out chapters it picks the right option 90% of the time when forced to answer. But it abstained on 0 of the 2 ambiguous held-out items.
* On ambiguous items overall it abstains less (72.7% vs 100% for the strands-decider fine-tune).
* So Laya is roughly 5× smaller and faster, but less cautious out of domain. That's a good trade-off to show,
  and a reason to tune the review band per model.

Laya base/FT numbers come from the GPU training job, with the same items and the same scorer as the strands-decider rows.
The strands-decider numbers were scored through the Snowflake service.

## 8. Trade-offs to call out

| Decision | Why | Cost |
|---|---|---|
| Typed decision models, not a generative LLM | Calibrated per-option probabilities, ms latency, cents/1K | Narrow: one question shape per call |
| Synthetic, template-based data | Fast, covers ambiguity on purpose | In-domain scores are optimistic; SME review needed |
| Snowflake ML Jobs on GPU | Data stays put, minutes not hours, no extra platform | GPU queue/start-up time (pool cold start of several minutes) |
| Weights stored as registry artifacts | No runtime egress; reproducible versions | Larger model objects, slower log_model |
| Service function, not run_batch | Works today on the trial | Bulk scoring rides a long-running service |
| One shared abstention policy (band 0.25 / floor 0.5) | Same numbers across all models | Not tuned per model; Laya also ships fitted thresholds |
| Claude panel kept, not faked | Honest trial limitation | No frontier-LLM baseline yet |

## 9. Limitations

* DTR-Bench is generated from templates and self-reviewed. Don't present it as operational accuracy.
* Held-out chapters are only 22 benchmark items. Expect wide confidence intervals.
* Cortex `AI_COMPLETE` and external access are unavailable on trial accounts, so the Claude comparison hasn't run yet.
* Laya's shipped temperatures are flagged by the library as out of range. The base model's confidence is uncalibrated.

## 10. Next steps

1. **SME review** of DTR-Bench, plus 100–200 real (redacted) historical adjudications as the real test set.
2. **Upgrade the account** and run the Claude/Cortex column (no code change: `bench_llm_dtr.py` and the app's Compare tab).
3. **Load and cost test:** CPU vs GPU pools for Laya, autoscaling (`min/max_instances`), concurrency sweep.
4. **Governance:** registry aliases (`PRODUCTION`/`CANDIDATE`), role-based access to the app and the review queue,
   tags/lineage, an approval step before an alias moves.
5. **Monitoring:** log every decision to a table and track drift and the abstention rate. Refit calibration
   on resolved review-queue items (the queue doubles as labelled data).
6. Adversarial and contradictory scenarios, and multi-paragraph cases.

## Teardown

`sql/99_teardown.sql` drops services, pools, stages, the app and the PAT network policy. Suspend
(not drop) between demos: `ALTER SERVICE ... SUSPEND`, `ALTER COMPUTE POOL ... SUSPEND`.

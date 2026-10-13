# DTR-Bench results (decider only)

142 DTR-Bench items (DTR Part I, Passenger Movement): 120 in-domain, and 22 held-out items
from Ch. 106 rental cars and Ch. 107 commercial ships, which never appear in training. The
labels were self-reviewed against the cited excerpts; no DoD travel SME has reviewed them
yet. Abstention policy (`strands_decider.dtr_eval.decide`): a yes/no answer goes to review
when |P(true) - 0.5| < 0.25, and a choice/score goes to review when its top probability is
below 0.5.

**Strict accuracy:** a definite answer must be correct, and an ambiguous item must be
abstained on. **Coverage**, **answered accuracy** and **forced accuracy** count only the
131 items with a definite gold answer. **Abstain recall** counts only the 11 ambiguous
items.

## Accuracy and calibration

| model, where scored | strict | forced | coverage | answered acc. | abstain recall | ECE | holdout strict |
|---|---|---|---|---|---|---|---|
| base v21, local CPU | 0.493 | 0.763 | 0.588 | 0.844 | 0.455 | 0.123 | 0.545 |
| base v21, Snowflake REST service | 0.486 | 0.771 | 0.580 | 0.842 | 0.455 | 0.135 | 0.500 |
| DTR fine-tune `dtr-ft-2`, ML Job eval | 0.979 | 0.985 | 0.977 | 1.000 | 1.000 | 0.027 | 0.864 |
| DTR fine-tune `dtr-ft-2`, Snowflake REST service | **0.986** | 0.985 | 0.985 | 1.000 | 1.000 | 0.027 | **0.909** |

- The SQL service-function path (`sql/05_service_dtr.sql`, which writes `DTR_SCORES`) matched
  REST on strict accuracy for both versions: 0.486 for base v21 and 0.986 for the fine-tune.
- The same `dtr-ft-2` weights scored one item differently in the ML Job eval and in the
  service; the GPU numerics differ between those two runtimes.
- An earlier run, `dtr-ft-1`, used the same configuration. Its export failed and it was not
  deployed. Its ML Job eval scored 0.986 strict and 0.909 on the holdout.

**Caveat.** The in-domain items use a separate vocabulary from training, but they come from the
same scenario templates, so the in-domain score of 1.000 partly reflects template overlap.
The held-out chapters are the better signal, and with n=22 that signal is noisy. The real
test is a set of scenarios written by an SME.

Files: `bench_*.json` in strands-decider `results/dtr/`, and `v21_rest_summary.json` and
`ft_rest_summary.json` in this directory.

## Serving on Snowflake

One `GPU_NV_S` node (A10G), `DTR_DECIDER_HTTP`, `num_workers=1`, `max_batch_rows=32`. The
client ran on a VM outside Snowflake.

| version | mode | decisions/s | GPU p50 | client p50 | GPU $ per 1K decisions |
|---|---|---|---|---|---|
| V21_SERVICE | REST, sequential | 7.41 | 67.8 ms | 131.6 ms | $0.043 |
| V21_SERVICE | REST, 8 clients | 14.08 | – | 555 ms | $0.023 |
| FT_SERVICE | REST, sequential | 7.33 | 68.1 ms | 132.3 ms | $0.043 |
| FT_SERVICE | REST, 8 clients | 14.19 | – | 553 ms | $0.022 |
| FT_SERVICE | SQL service function, 142 rows | 12.5 | 67 ms | – | $0.025 GPU + $0.044 XS warehouse |

- Concurrent answers were identical to sequential answers (142/142).
- Cost is node time spent scoring × list price: 0.57 credits/h × $2/credit = $1.14/h for
  GPU_NV_S, and $2/h for an XS warehouse. An idle service still bills $1.14/h.
- `sql/09_cost_accounting.sql` reads metered `ACCOUNT_USAGE`, which lags by up to about 3 h, and
  is keyed to the JevBench `service` run. These figures are computed from statement wall time
  at list price.

## Training cost

- Each GPU fine-tune ran as one Snowflake ML Job on `SYSTEM_COMPUTE_POOL_GPU` (GPU_NV_S,
  A10G): 201 optimizer steps, about 14.5 min of training, and about 16 min of payload time in
  total, including both DTR-Bench evals.
- At 0.57 credits/h, that is about 0.15 credits (about $0.30) per run, plus node start-up
  time.

## Not run on this trial

- `claude-sonnet-5` through `AI_COMPLETE`: Cortex AI functions are unavailable on trial
  accounts. The code is ready (`python/bench_llm_dtr.py` and the app's LLM route).
- Model Registry `run_batch`: the batch base image pins `click<8.3`, and the Hugging Face
  Hub version that transformers 5.x needs requires `click>=8.4.2`, so the image cannot
  resolve. Bulk scoring uses the service function from SQL instead.
- Hosting the app inside Snowflake: external access integrations are unavailable on trial
  accounts. The app was run locally against the live service and tables.

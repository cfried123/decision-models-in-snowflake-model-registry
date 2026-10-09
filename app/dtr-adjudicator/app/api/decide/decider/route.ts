/**
 * POST /api/decide/decider {requestId, taskId, scenario} — one DTR determination from
 * the fine-tuned strands-decider's real-time inference service over REST.
 *
 * The state sent is the scenario plus the governing DTR excerpt from the catalog, so a
 * user can't substitute their own "regulation". The service's public endpoint takes a
 * programmatic access token as `Authorization: Snowflake Token="<PAT>"`; the PAT comes
 * from the DECIDER_PAT secret (mounted in SPCS; SNOWFLAKE_SECRET_DECIDER_PAT_SECRET_STRING
 * locally) and never leaves the server.
 */

import { getSecret, querySnowflake, SecretType } from "@/lib/snowflake"
import { DECIDER_SERVICE } from "@/lib/config"
import { cleanScenario, decide, isTaskId, stateFor, TASKS, type Determination } from "@/lib/dtr"

export const dynamic = "force-dynamic"

const SERVICE_RE = /^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*){0,2}$/
let endpointUrl: string | null = process.env.DECIDER_ENDPOINT ? `https://${process.env.DECIDER_ENDPOINT}/system-one` : null

async function getEndpoint(): Promise<string> {
  if (endpointUrl) return endpointUrl
  // SHOW can't take a bind; the name is deployment config, checked to be an identifier.
  if (!SERVICE_RE.test(DECIDER_SERVICE)) throw new Error("DECIDER_SERVICE is not a valid identifier")
  const rows = await querySnowflake(`SHOW ENDPOINTS IN SERVICE ${DECIDER_SERVICE}`)
  const host = rows
    .map((r: Record<string, unknown>) => String(r.ingress_url ?? ""))
    .find((u: string) => u.includes(".snowflakecomputing.app"))
  if (!host) throw new Error("the decider service has no public endpoint yet")
  endpointUrl = `https://${host}/system-one`
  return endpointUrl
}

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as { requestId?: unknown; taskId?: unknown; scenario?: unknown }
  const requestId = String(body.requestId ?? "").slice(0, 64)
  const scenario = cleanScenario(body.scenario)
  if (!isTaskId(body.taskId) || !scenario) {
    return Response.json({ requestId, error: "taskId and scenario are required" }, { status: 400 })
  }
  const taskId = body.taskId
  const task = TASKS[taskId]
  try {
    const url = await getEndpoint()
    const pat = getSecret("DECIDER_PAT", SecretType.GENERIC_STRING)
    const payload = {
      dataframe_split: {
        index: [0],
        columns: ["STATE_JSON", "QUESTIONS_JSON"],
        data: [[JSON.stringify(stateFor(taskId, scenario)), JSON.stringify({ decision: task.question })]],
      },
    }
    const t0 = performance.now()
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Snowflake Token="${pat}"` },
      body: JSON.stringify(payload),
    })
    const modelMs = performance.now() - t0
    if (!resp.ok) throw new Error(`decider endpoint returned HTTP ${resp.status}`)
    const out = (await resp.json()).data[0][1]
    const answer = JSON.parse(out.ANSWER_JSON)
    if (answer.error) throw new Error("decider could not answer this request")
    const a = answer.answers.decision
    const distribution: Record<string, number> =
      task.kind === "noul" ? { true: Number(a.noul), false: 1 - Number(a.noul) } : { ...(a.probabilities ?? {}) }
    const d = decide(task.kind, distribution)
    const result: Determination = {
      requestId,
      taskId,
      kind: task.kind,
      decision: d.decision,
      top: d.top,
      confidence: d.confidence,
      distribution,
      citation: task.dtr_citation,
      needsReview: d.decision === "abstain",
      modelMs,
      inputTokens: Number(out.INPUT_TOKENS ?? 0),
    }
    return Response.json(result)
  } catch (e) {
    console.error(new Date().toISOString(), "decider call failed", e)
    return Response.json({ requestId, error: "decider call failed" }, { status: 502 })
  }
}

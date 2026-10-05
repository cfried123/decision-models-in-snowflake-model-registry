/**
 * POST /api/decide/decider {ticketId, text} — one ticket to decider-2b's real-time
 * inference service over REST.
 *
 * The service's public endpoint takes a programmatic access token as
 * `Authorization: Snowflake Token="<PAT>"`. The PAT comes from the DECIDER_PAT secret
 * (mounted in SPCS; SNOWFLAKE_SECRET_DECIDER_PAT_SECRET_STRING locally) and never
 * leaves the server. The app reaches the endpoint through the DECIDER_DEMO_EAI egress.
 */

import { getSecret, querySnowflake, SecretType } from "@/lib/snowflake"
import { DECIDER_SERVICE } from "@/lib/config"
import { DECIDER_QUESTIONS, CATEGORY_KEYS, type Category, type Decision } from "@/lib/questions"

export const dynamic = "force-dynamic"

const QUESTIONS_JSON = JSON.stringify(DECIDER_QUESTIONS)
let endpointUrl: string | null = process.env.DECIDER_ENDPOINT ? `https://${process.env.DECIDER_ENDPOINT}/system-one` : null

async function getEndpoint(): Promise<string> {
  if (endpointUrl) return endpointUrl
  const rows = await querySnowflake(`SHOW ENDPOINTS IN SERVICE ${DECIDER_SERVICE}`)
  const host = rows
    .map((r: Record<string, unknown>) => String(r.ingress_url ?? ""))
    .find((u: string) => u.includes(".snowflakecomputing.app"))
  if (!host) throw new Error(`no public endpoint on ${DECIDER_SERVICE} yet`)
  endpointUrl = `https://${host}/system-one`
  return endpointUrl
}

export async function POST(req: Request) {
  const { ticketId, text } = (await req.json()) as { ticketId: number; text: string }
  try {
    const url = await getEndpoint()
    const pat = getSecret("DECIDER_PAT", SecretType.GENERIC_STRING)
    const body = {
      dataframe_split: {
        index: [0],
        columns: ["STATE_JSON", "QUESTIONS_JSON"],
        data: [[JSON.stringify(String(text).slice(0, 4000)), QUESTIONS_JSON]],
      },
    }
    const t0 = performance.now()
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Snowflake Token="${pat}"` },
      body: JSON.stringify(body),
    })
    const modelMs = performance.now() - t0
    if (!resp.ok) throw new Error(`decider endpoint returned HTTP ${resp.status}`)
    const out = (await resp.json()).data[0][1]
    const answer = JSON.parse(out.ANSWER_JSON)
    if (answer.error) throw new Error(answer.error)
    const cat = answer.answers.category
    const esc = answer.answers.escalate
    const category = (CATEGORY_KEYS as string[]).includes(cat.choice) ? (cat.choice as Category) : "bug"
    const escalateProb = Number(esc.probabilities?.escalate ?? (esc.choice === "escalate" ? 1 : 0))
    const decision: Decision = {
      ticketId,
      category,
      categoryConfidence: Number(cat.probabilities?.[cat.choice] ?? cat.confidence ?? 0),
      escalate: esc.choice === "escalate",
      escalateProb,
      modelMs,
      inputTokens: Number(out.INPUT_TOKENS ?? 0),
    }
    return Response.json(decision)
  } catch (e) {
    console.error(new Date().toISOString(), "decider call failed", e)
    return Response.json({ ticketId, error: e instanceof Error ? e.message : "decider call failed" }, { status: 502 })
  }
}

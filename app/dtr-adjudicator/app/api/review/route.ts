/**
 * The human-review queue: determinations a model declined to make on its own.
 *
 * POST /api/review {requestId, taskId, scenario, side, top, confidence} — enqueue
 * GET  /api/review                                                     — open items
 * PATCH /api/review {reviewId, resolution, note}                        — record the reviewer's call
 *
 * All values are bind variables; resolution must be one of the task's options.
 */

import { querySnowflake } from "@/lib/snowflake"
import { REVIEW_TABLE } from "@/lib/config"
import { clamp01, cleanScenario, isTaskId, TASKS } from "@/lib/dtr"

export const dynamic = "force-dynamic"

const SIDES = new Set(["decider", "llm"])

export async function POST(req: Request) {
  const b = (await req.json().catch(() => ({}))) as Record<string, unknown>
  /* Security: server-side allowlist validation of task, side and length-capped text (OWASP ASVS V5). */
  const scenario = cleanScenario(b.scenario)
  const side = String(b.side ?? "")
  if (!isTaskId(b.taskId) || !scenario || !SIDES.has(side)) {
    return Response.json({ error: "taskId, scenario and side are required" }, { status: 400 })
  }
  const top = String(b.top ?? "").slice(0, 64)
  try {
    await querySnowflake(
      /* Security: every reviewer/user value is a bind variable (OWASP A03, NIST SP 800-53 SI-10). */
      `INSERT INTO IDENTIFIER(?) (REQUEST_ID, TASK, SIDE, SCENARIO, DTR_CITATION, MODEL_TOP, MODEL_CONFIDENCE)
       SELECT ?, ?, ?, ?, ?, ?, ?`,
      {
        binds: [REVIEW_TABLE, String(b.requestId ?? "").slice(0, 64), b.taskId, side, scenario,
          TASKS[b.taskId].dtr_citation, top, clamp01(b.confidence)],
      },
    )
    return Response.json({ queued: true })
  } catch (e) {
    console.error(new Date().toISOString(), "review enqueue failed", e)
    return Response.json({ error: "Failed to queue for review" }, { status: 500 })
  }
}

export async function GET() {
  try {
    const rows = await querySnowflake(
      `SELECT REVIEW_ID, REQUEST_ID, TASK, SIDE, SCENARIO, DTR_CITATION, MODEL_TOP, MODEL_CONFIDENCE, CREATED_AT
       FROM IDENTIFIER(?) WHERE RESOLUTION IS NULL ORDER BY CREATED_AT DESC LIMIT 100`,
      { binds: [REVIEW_TABLE] },
    )
    return Response.json(
      rows.map((r: Record<string, unknown>) => ({
        reviewId: Number(r.REVIEW_ID),
        requestId: String(r.REQUEST_ID),
        taskId: String(r.TASK),
        side: String(r.SIDE),
        scenario: String(r.SCENARIO),
        citation: String(r.DTR_CITATION),
        top: String(r.MODEL_TOP),
        confidence: Number(r.MODEL_CONFIDENCE),
        createdAt: String(r.CREATED_AT),
      })),
    )
  } catch (e) {
    console.error(new Date().toISOString(), "review query failed", e)
    return Response.json({ error: "Failed to load the review queue" }, { status: 500 })
  }
}

export async function PATCH(req: Request) {
  const b = (await req.json().catch(() => ({}))) as Record<string, unknown>
  const reviewId = Number(b.reviewId)
  const resolution = String(b.resolution ?? "")
  const taskId = b.taskId
  /* Security: a resolution must be one of the task's option keys, so the queue only stores valid determinations. */
  if (!Number.isInteger(reviewId) || !isTaskId(taskId) || !TASKS[taskId].option_names.includes(resolution)) {
    return Response.json({ error: "reviewId, taskId and a valid resolution are required" }, { status: 400 })
  }
  try {
    await querySnowflake(
      `UPDATE IDENTIFIER(?) SET RESOLUTION = ?, REVIEWER_NOTE = ?, RESOLVED_BY = CURRENT_USER(), RESOLVED_AT = CURRENT_TIMESTAMP()
       WHERE REVIEW_ID = ? AND TASK = ? AND RESOLUTION IS NULL`,
      { binds: [REVIEW_TABLE, resolution, String(b.note ?? "").slice(0, 1000), reviewId, taskId] },
    )
    return Response.json({ resolved: true })
  } catch (e) {
    console.error(new Date().toISOString(), "review resolve failed", e)
    return Response.json({ error: "Failed to record the review" }, { status: 500 })
  }
}

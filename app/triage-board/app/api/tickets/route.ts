/**
 * GET /api/tickets?offset=0&limit=200 — the next tickets in a fixed order, so both
 * sides of the board see the same stream.
 */

import { querySnowflake } from "@/lib/snowflake"
import { TICKETS_TABLE } from "@/lib/config"

export const dynamic = "force-dynamic"

export async function GET(req: Request) {
  const params = new URL(req.url).searchParams
  const offset = Math.max(0, Number(params.get("offset") ?? 0) || 0)
  const limit = Math.min(500, Math.max(1, Number(params.get("limit") ?? 200) || 200))
  try {
    const rows = await querySnowflake(
      `SELECT TICKET_ID, TEXT, GEN_CATEGORY, GEN_ESCALATE FROM ${TICKETS_TABLE} ORDER BY TICKET_ID LIMIT ? OFFSET ?`,
      { binds: [limit, offset] },
    )
    return Response.json(
      rows.map((r: Record<string, unknown>) => ({
        id: Number(r.TICKET_ID),
        text: String(r.TEXT),
        genCategory: String(r.GEN_CATEGORY),
        genEscalate: Boolean(r.GEN_ESCALATE),
      })),
    )
  } catch (e) {
    console.error(new Date().toISOString(), "tickets query failed", e)
    return Response.json({ error: e instanceof Error ? e.message : "Failed to load tickets" }, { status: 500 })
  }
}

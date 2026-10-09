/**
 * GET /api/items — the DTR-Bench scenarios, in a fixed order, for the sample picker and
 * the side-by-side replay. EXPECTED is the self-reviewed gold label (SME review pending).
 */

import { querySnowflake } from "@/lib/snowflake"
import { ITEMS_TABLE } from "@/lib/config"

export const dynamic = "force-dynamic"

export async function GET() {
  try {
    const rows = await querySnowflake(
      `SELECT ITEM_ID, TASK, SPLIT, SCENARIO, EXPECTED, SME_REVIEW FROM IDENTIFIER(?) ORDER BY ITEM_ID`,
      { binds: [ITEMS_TABLE] },
    )
    return Response.json(
      rows.map((r: Record<string, unknown>) => ({
        id: String(r.ITEM_ID),
        taskId: String(r.TASK),
        split: String(r.SPLIT),
        scenario: String(r.SCENARIO),
        expected: String(r.EXPECTED),
        smeReview: String(r.SME_REVIEW),
      })),
    )
  } catch (e) {
    console.error(new Date().toISOString(), "items query failed", e)
    return Response.json({ error: "Failed to load DTR-Bench items" }, { status: 500 })
  }
}

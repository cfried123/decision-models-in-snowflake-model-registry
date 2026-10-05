import { describe, it, expect, vi, beforeEach } from "vitest"

vi.mock("../../lib/snowflake", () => ({
  querySnowflake: vi.fn(),
  getSecret: vi.fn(() => "test-pat"),
  SecretType: { GENERIC_STRING: "GENERIC_STRING" },
}))

import { querySnowflake } from "../../lib/snowflake"
import { POST as deciderPost } from "../../app/api/decide/decider/route"
import { POST as llmPost } from "../../app/api/decide/llm/route"

const req = (body: unknown) => new Request("http://x/api", { method: "POST", body: JSON.stringify(body) })

describe("POST /api/decide/decider", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(querySnowflake).mockResolvedValue([{ ingress_url: "abc-org-acct.snowflakecomputing.app" }])
  })

  it("maps decider-2b's answer and sends the PAT only in the header", async () => {
    const answer = {
      answers: {
        category: { type: "choice", choice: "refund", probabilities: { refund: 0.8, billing: 0.2 } },
        escalate: { type: "choice", choice: "routine", probabilities: { escalate: 0.1, routine: 0.9 } },
      },
    }
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ data: [[0, { ANSWER_JSON: JSON.stringify(answer), INPUT_TOKENS: 120 }]] })),
    )
    vi.stubGlobal("fetch", fetchMock)

    const res = await deciderPost(req({ ticketId: 7, text: "Charged twice, refund please" }))
    const json = await res.json()

    expect(res.status).toBe(200)
    expect(json).toMatchObject({ ticketId: 7, category: "refund", categoryConfidence: 0.8, escalate: false, escalateProb: 0.1 })
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe("https://abc-org-acct.snowflakecomputing.app/system-one")
    expect(init.headers.Authorization).toBe('Snowflake Token="test-pat"')
    expect(JSON.stringify(json)).not.toContain("test-pat")
  })

  it("returns 502 when the endpoint fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("no", { status: 500 })))
    const res = await deciderPost(req({ ticketId: 1, text: "x" }))
    expect(res.status).toBe(502)
  })
})

describe("POST /api/decide/llm", () => {
  beforeEach(() => vi.clearAllMocks())

  it("binds the ticket text and maps the structured output", async () => {
    vi.mocked(querySnowflake).mockResolvedValue([
      {
        R: JSON.stringify({
          structured_output: [{ raw_message: { category: "bug", category_confidence: 0.9, escalate: "escalate", escalate_confidence: 0.7 } }],
          usage: { prompt_tokens: 400, completion_tokens: 30 },
        }),
      },
    ])
    const res = await llmPost(req({ ticketId: 3, text: "'; DROP TABLE x; --" }))
    const json = await res.json()

    expect(res.status).toBe(200)
    expect(json).toMatchObject({ ticketId: 3, category: "bug", escalate: true, inputTokens: 400, outputTokens: 30 })
    expect(json.escalateProb).toBeCloseTo(0.7)
    const [sql, opts] = vi.mocked(querySnowflake).mock.calls[0]
    expect(sql).not.toContain("DROP TABLE")
    expect(opts?.binds?.[1]).toContain("DROP TABLE")
  })
})

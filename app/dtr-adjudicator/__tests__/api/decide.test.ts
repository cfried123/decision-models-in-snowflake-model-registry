import { describe, it, expect, vi, beforeEach } from "vitest"

vi.mock("../../lib/snowflake", () => ({
  querySnowflake: vi.fn(),
  getSecret: vi.fn(() => "test-pat"),
  SecretType: { GENERIC_STRING: "GENERIC_STRING" },
}))

import { querySnowflake } from "../../lib/snowflake"
import { POST as deciderPost } from "../../app/api/decide/decider/route"
import { POST as llmPost } from "../../app/api/decide/llm/route"
import { POST as reviewPost, PATCH as reviewPatch } from "../../app/api/review/route"
import { TASKS, decide } from "../../lib/dtr"

const req = (body: unknown, method = "POST") => new Request("http://x/api", { method, body: JSON.stringify(body) })
const SCENARIO = "A GS-13 flew business class Washington, DC to Honolulu without premium class approval."

describe("decide (review policy, mirrors strands_decider.dtr_eval.decide)", () => {
  it("routes yes/no answers near 50% to review", () => {
    expect(decide("noul", { true: 0.6, false: 0.4 })).toMatchObject({ decision: "abstain", top: "true" })
    expect(decide("noul", { true: 0.9, false: 0.1 })).toMatchObject({ decision: "true", confidence: 0.9 })
    expect(decide("noul", { true: 0.1, false: 0.9 })).toMatchObject({ decision: "false" })
  })
  it("routes low-confidence choices to review", () => {
    expect(decide("choice", { a: 0.45, b: 0.35, c: 0.2 })).toMatchObject({ decision: "abstain", top: "a" })
    expect(decide("choice", { a: 0.7, b: 0.3 })).toMatchObject({ decision: "a" })
  })
})

describe("POST /api/decide/decider", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(querySnowflake).mockResolvedValue([{ ingress_url: "abc-org-acct.snowflakecomputing.app" }])
  })

  it("sends the catalog excerpt, maps the answer and keeps the PAT server-side", async () => {
    const answer = {
      answers: { decision: { type: "choice", choice: "personal_upgrade", probabilities: { personal_upgrade: 0.8, premium_class: 0.2 } } },
    }
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ data: [[0, { ANSWER_JSON: JSON.stringify(answer), INPUT_TOKENS: 420 }]] })),
    )
    vi.stubGlobal("fetch", fetchMock)

    const res = await deciderPost(req({ requestId: "r1", taskId: "dtr_class_of_service", scenario: SCENARIO }))
    const json = await res.json()

    expect(res.status).toBe(200)
    expect(json).toMatchObject({ decision: "personal_upgrade", confidence: 0.8, needsReview: false, inputTokens: 420 })
    expect(json.citation).toBe(TASKS.dtr_class_of_service.dtr_citation)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe("https://abc-org-acct.snowflakecomputing.app/system-one")
    expect(init.headers.Authorization).toBe('Snowflake Token="test-pat"')
    const sent = JSON.parse(init.body).dataframe_split.data[0]
    expect(JSON.parse(sent[0])).toMatchObject({ scenario: SCENARIO, dtr_excerpt: TASKS.dtr_class_of_service.dtr_excerpt })
    expect(JSON.parse(sent[1]).decision).toEqual(TASKS.dtr_class_of_service.question)
    expect(JSON.stringify(json)).not.toContain("test-pat")
  })

  it("flags an uncertain yes/no for review", async () => {
    const answer = { answers: { decision: { type: "noul", noul: 0.55 } } }
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ data: [[0, { ANSWER_JSON: JSON.stringify(answer) }]] })),
    ))
    const json = await (await deciderPost(req({ taskId: "dtr_rental_use", scenario: "Rental car to dinner." }))).json()
    expect(json).toMatchObject({ decision: "abstain", top: "true", needsReview: true })
  })

  it("rejects unknown tasks and empty scenarios", async () => {
    expect((await deciderPost(req({ taskId: "drop table", scenario: "x" }))).status).toBe(400)
    expect((await deciderPost(req({ taskId: "dtr_rental_use", scenario: "   " }))).status).toBe(400)
  })

  it("returns 502 without leaking the error when the endpoint fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("no", { status: 500 })))
    const res = await deciderPost(req({ taskId: "dtr_rental_use", scenario: "x" }))
    expect(res.status).toBe(502)
    expect((await res.json()).error).toBe("decider call failed")
  })
})

describe("POST /api/decide/llm", () => {
  beforeEach(() => vi.clearAllMocks())

  it("binds the prompt and maps the structured output", async () => {
    vi.mocked(querySnowflake).mockResolvedValue([
      { R: JSON.stringify({ structured_output: [{ raw_message: { probability_true: 0.95 } }], usage: { prompt_tokens: 600, completion_tokens: 12 } }) },
    ])
    const res = await llmPost(req({ taskId: "dtr_foreign_carrier", scenario: SCENARIO }))
    const json = await res.json()
    expect(json).toMatchObject({ decision: "true", needsReview: false, inputTokens: 600, outputTokens: 12 })
    const [sql, opts] = vi.mocked(querySnowflake).mock.calls[0]
    expect(sql).not.toContain(SCENARIO)
    expect(opts?.binds?.[1]).toContain(SCENARIO)
  })

  it("rejects answers outside the option set", async () => {
    vi.mocked(querySnowflake).mockResolvedValue([
      { R: JSON.stringify({ structured_output: [{ raw_message: { answer: "teleport", confidence: 0.9 } }] }) },
    ])
    expect((await llmPost(req({ taskId: "dtr_travel_mode", scenario: SCENARIO }))).status).toBe(502)
  })
})

describe("/api/review", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(querySnowflake).mockResolvedValue([])
  })

  it("queues with binds only", async () => {
    const res = await reviewPost(req({ requestId: "r1", taskId: "dtr_ship_travel", scenario: "Ship to Guam'; DROP TABLE x; --", side: "decider", top: "true", confidence: 0.6 }))
    expect(res.status).toBe(200)
    const [sql, opts] = vi.mocked(querySnowflake).mock.calls[0]
    expect(sql).not.toContain("DROP")
    expect(opts?.binds).toContain("Ship to Guam'; DROP TABLE x; --")
  })

  it("only accepts a resolution from the task's options", async () => {
    expect((await reviewPatch(req({ reviewId: 1, taskId: "dtr_ship_travel", resolution: "maybe" }, "PATCH"))).status).toBe(400)
    expect((await reviewPatch(req({ reviewId: 1, taskId: "dtr_ship_travel", resolution: "false" }, "PATCH"))).status).toBe(200)
  })
})

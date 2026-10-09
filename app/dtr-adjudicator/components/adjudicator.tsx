"use client"

/**
 * DTR Travel Entitlement Adjudicator. One scenario goes to both sides with the same
 * question and the same governing DTR excerpt; each answer is shown with its calibrated
 * confidence and citation, and anything a side abstains on is queued for human review
 * instead of being presented as a determination. "Replay DTR-Bench" streams the
 * benchmark through both sides for live accuracy, review rate, decisions/sec and cost.
 */

import { useCallback, useEffect, useRef, useState } from "react"
import { Gavel, Pause, Play, RotateCcw, UserCheck } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { Determination } from "@/lib/dtr"

type SideKey = "decider" | "llm"
type Prices = {
  effectiveDate: string
  deciderUsdPerHour: number
  deciderHardware: string
  llmInputUsdPerM: number
  llmOutputUsdPerM: number
  llmWarehouseUsdPerHour: number
}
type TaskInfo = {
  id: string; label: string; kind: string; citation: string; instructions: string
  holdout: boolean; options: [string, string][]; provisions: { section: string; citation: string }[]
}
type Config = { llmModel: string; prices: Prices; tasks: TaskInfo[]; policy: { noulBand: number; choiceFloor: number } }
type Item = {
  id: string; taskId: string; split: string; section?: string; scenario: string; expected: string; smeReview: string
}
type ReviewRow = {
  reviewId: number; requestId: string; taskId: string; side: string; scenario: string
  citation: string; top: string; confidence: number; createdAt: string
}

interface SideStats {
  n: number; correct: number; review: number; reviewCorrect: number; errors: number
  latencies: number[]; inputTokens: number; outputTokens: number; activeMs: number
  activeSince: number | null; inflight: number; cursor: number
}
const newStats = (): SideStats => ({
  n: 0, correct: 0, review: 0, reviewCorrect: 0, errors: 0, latencies: [], inputTokens: 0,
  outputTokens: 0, activeMs: 0, activeSince: null, inflight: 0, cursor: 0,
})

const SIDES: SideKey[] = ["decider", "llm"]
const fmtMs = (v: number | null) => (v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`)
const fmtUsd = (v: number | null) => (v == null ? "–" : `$${v < 0.01 ? v.toFixed(5) : v.toFixed(3)}`)
const pctStr = (a: number, b: number) => (b ? `${((100 * a) / b).toFixed(1)}%` : "–")

function p50(xs: number[]): number | null {
  if (!xs.length) return null
  const s = [...xs].sort((a, b) => a - b)
  return s[Math.floor((s.length - 1) / 2)]
}
const active = (s: SideStats, now: number) => s.activeMs + (s.activeSince != null ? now - s.activeSince : 0)
function costUsd(side: SideKey, s: SideStats, p: Prices, now: number) {
  const h = active(s, now) / 3_600_000
  return side === "decider"
    ? p.deciderUsdPerHour * h
    : (s.inputTokens / 1e6) * p.llmInputUsdPerM + (s.outputTokens / 1e6) * p.llmOutputUsdPerM + p.llmWarehouseUsdPerHour * h
}

async function ask(side: SideKey, requestId: string, taskId: string, scenario: string, section?: string): Promise<Determination> {
  const r = await fetch(`/api/decide/${side}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ requestId, taskId, scenario, section }),
  })
  return r.json()
}

async function enqueue(side: SideKey, scenario: string, d: Determination, section?: string) {
  await fetch("/api/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ requestId: d.requestId, taskId: d.taskId, section, scenario, side, top: d.top, confidence: d.confidence }),
  })
}

export function Adjudicator() {
  const [config, setConfig] = useState<Config | null>(null)
  const [items, setItems] = useState<Item[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [taskId, setTaskId] = useState("")
  const [scenario, setScenario] = useState("")
  const [section, setSection] = useState("")
  const [busy, setBusy] = useState(false)
  const [single, setSingle] = useState<Partial<Record<SideKey, Determination>>>({})
  const [queue, setQueue] = useState<ReviewRow[]>([])
  const [replaying, setReplaying] = useState(false)
  const [, setFrame] = useState(0)
  const stats = useRef<Record<SideKey, SideStats>>({ decider: newStats(), llm: newStats() })
  const replayRef = useRef(false)
  const epoch = useRef(0)

  const loadQueue = useCallback(() => {
    fetch("/api/review").then((r) => r.json()).then((q) => Array.isArray(q) && setQueue(q)).catch(() => {})
  }, [])

  useEffect(() => {
    Promise.all([fetch("/api/config").then((r) => r.json()), fetch("/api/items").then((r) => r.json())])
      .then(([cfg, its]) => {
        if (its.error) throw new Error(its.error)
        setConfig(cfg)
        setItems(its)
        setTaskId(cfg.tasks[0]?.id ?? "")
      })
      .catch((e) => setLoadError(e instanceof Error ? e.message : String(e)))
    loadQueue()
  }, [loadQueue])

  useEffect(() => {
    if (!replaying) return
    const id = setInterval(() => setFrame((f) => f + 1), 200)
    return () => clearInterval(id)
  }, [replaying])

  const adjudicate = async () => {
    if (!scenario.trim() || !taskId) return
    setBusy(true)
    setSingle({})
    const requestId = crypto.randomUUID()
    await Promise.all(
      SIDES.map(async (side) => {
        const d = await ask(side, requestId, taskId, scenario, section || undefined).catch(() => ({ requestId, error: "request failed" }) as Determination)
        setSingle((s) => ({ ...s, [side]: d }))
        if (!d.error && d.needsReview) await enqueue(side, scenario, d, section || undefined)
      }),
    )
    setBusy(false)
    loadQueue()
  }

  const pump = useCallback(
    (side: SideKey) => {
      const s = stats.current[side]
      const my = epoch.current
      while (replayRef.current && s.inflight < 4 && s.cursor < items.length) {
        const it = items[s.cursor++]
        if (s.inflight === 0) s.activeSince = performance.now()
        s.inflight++
        ask(side, it.id, it.taskId, it.scenario, it.section)
          .then((d) => {
            if (my !== epoch.current) return
            if (d.error) return void s.errors++
            s.n++
            s.latencies.push(d.modelMs)
            s.inputTokens += d.inputTokens ?? 0
            s.outputTokens += d.outputTokens ?? 0
            if (it.expected === "abstain") {
              if (d.needsReview) s.reviewCorrect++
            } else if (d.decision === it.expected) s.correct++
            if (d.needsReview) s.review++
          })
          .catch(() => my === epoch.current && s.errors++)
          .finally(() => {
            if (my !== epoch.current) return
            s.inflight--
            if (s.inflight === 0 && s.activeSince != null) {
              s.activeMs += performance.now() - s.activeSince
              s.activeSince = null
            }
            const all = SIDES.every((k) => stats.current[k].cursor >= items.length && stats.current[k].inflight === 0)
            if (all) {
              replayRef.current = false
              setReplaying(false)
            }
            pump(side)
          })
      }
    },
    [items],
  )

  const startReplay = () => {
    replayRef.current = true
    setReplaying(true)
    SIDES.forEach(pump)
  }
  const pauseReplay = () => {
    replayRef.current = false
    setReplaying(false)
  }
  const resetReplay = () => {
    pauseReplay()
    epoch.current++
    stats.current = { decider: newStats(), llm: newStats() }
    setFrame((f) => f + 1)
  }

  const resolve = async (row: ReviewRow, resolution: string) => {
    await fetch("/api/review", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reviewId: row.reviewId, taskId: row.taskId, resolution }),
    })
    loadQueue()
  }

  if (loadError) return <div className="p-8 text-sm text-destructive">Could not load: {loadError}</div>
  if (!config) return <div className="p-8 text-sm text-muted-foreground">Loading…</div>

  const task = config.tasks.find((t) => t.id === taskId)
  const provision = task?.provisions.find((p) => p.section === section) ?? task?.provisions[0]
  const samples = items.filter((i) => i.taskId === taskId).slice(0, 8)
  const taskOf = (id: string) => config.tasks.find((t) => t.id === id)
  const now = performance.now()
  const titles: Record<SideKey, [string, string, string]> = {
    decider: ["strands-decider (DTR fine-tune)", `Decision model on ${config.prices.deciderHardware}, REST`, "var(--brand-primary)"],
    llm: [config.llmModel, "Frontier LLM through AI_COMPLETE", "var(--llm-accent)"],
  }

  return (
    <div className="flex flex-col gap-4 p-4 lg:p-6">
      <section className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4">
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-sm">
            <span className="text-muted-foreground">Determination</span>
            <select
              aria-label="Determination"
              value={taskId}
              onChange={(e) => { setTaskId(e.target.value); setSection(""); setSingle({}) }}
              className="rounded-md border border-border bg-background px-2 py-1"
            >
              {config.tasks.map((t) => (
                <option key={t.id} value={t.id}>{t.label}{t.holdout ? " (held out of training)" : ""}</option>
              ))}
            </select>
          </label>
          {task && task.provisions.length > 1 ? (
            <label className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground">Governing provision</span>
              <select
                aria-label="Governing provision"
                value={provision?.section ?? ""}
                onChange={(e) => { setSection(e.target.value); setSingle({}) }}
                className="max-w-md rounded-md border border-border bg-background px-2 py-1 text-xs"
              >
                {task.provisions.map((p) => (
                  <option key={p.section} value={p.section}>{p.citation}</option>
                ))}
              </select>
            </label>
          ) : (
            <span className="text-xs text-muted-foreground">{provision?.citation}</span>
          )}
        </div>
        <p className="text-sm font-medium">{task?.instructions}</p>
        <textarea
          aria-label="Travel scenario"
          value={scenario}
          maxLength={2000}
          onChange={(e) => setScenario(e.target.value)}
          placeholder="Describe the travel: traveler, route, purpose, what was booked or requested, and what approvals exist."
          className="min-h-24 rounded-md border border-border bg-background p-2 text-sm"
        />
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={adjudicate} disabled={busy || !scenario.trim()}>
            <Gavel className="size-4" /> {busy ? "Adjudicating…" : "Adjudicate"}
          </Button>
          <span className="text-xs text-muted-foreground">Samples from DTR-Bench:</span>
          {samples.map((s) => (
            <button
              key={s.id}
              onClick={() => { setScenario(s.scenario); setSection(s.section ?? ""); setSingle({}) }}
              className="max-w-56 truncate rounded border border-border px-2 py-0.5 text-xs hover:bg-muted"
              title={s.scenario}
            >
              {s.id.split("-").pop()} · {s.scenario}
            </button>
          ))}
        </div>
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        {SIDES.map((side) => (
          <ResultPanel key={side} title={titles[side][0]} subtitle={titles[side][1]} accent={titles[side][2]}
            d={single[side]} task={task} busy={busy && !single[side]} />
        ))}
      </div>

      <section className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4">
        <header className="flex flex-wrap items-center gap-3">
          <h2 className="text-lg font-semibold tracking-tight">Replay DTR-Bench ({items.length} items)</h2>
          {replaying ? (
            <Button onClick={pauseReplay} variant="secondary"><Pause className="size-4" /> Pause</Button>
          ) : (
            <Button onClick={startReplay} disabled={!items.length}><Play className="size-4" /> Run</Button>
          )}
          <Button onClick={resetReplay} variant="outline"><RotateCcw className="size-4" /> Reset</Button>
        </header>
        <div className="grid gap-4 lg:grid-cols-2">
          {SIDES.map((side) => {
            const s = stats.current[side]
            const secs = active(s, now) / 1000
            const spent = costUsd(side, s, config.prices, now)
            const definite = items.slice(0, s.cursor).filter((i) => i.expected !== "abstain").length
            return (
              <dl key={side} className="grid grid-cols-3 gap-2">
                <Stat label={`${titles[side][0]} · decisions`} value={`${s.n}/${items.length}`} sub={`${secs ? (s.n / secs).toFixed(2) : "–"} / s`} accent={titles[side][2]} />
                <Stat label="Correct (clear-cut)" value={pctStr(s.correct, definite)} sub={`review rate ${pctStr(s.review, s.n)} · ambiguous caught ${s.reviewCorrect}`} />
                <Stat label="Latency p50 · cost" value={fmtMs(p50(s.latencies))} sub={`${fmtUsd(spent)} · ${fmtUsd(s.n ? (spent / s.n) * 1000 : null)} / 1K`} note={s.errors ? `${s.errors} errors` : undefined} />
              </dl>
            )
          })}
        </div>
      </section>

      <section className="flex flex-col gap-2 rounded-xl border border-border bg-card p-4">
        <header className="flex items-center gap-2">
          <UserCheck className="size-4" />
          <h2 className="text-lg font-semibold tracking-tight">Human-review queue</h2>
          <span className="text-xs text-muted-foreground">
            Abstentions (yes/no within {config.policy.noulBand} of 50%, or top option below {Math.round(config.policy.choiceFloor * 100)}%) land here instead of being issued.
          </span>
        </header>
        {!queue.length && <p className="text-sm text-muted-foreground">Nothing waiting for review.</p>}
        <ul className="flex flex-col gap-2">
          {queue.map((row) => {
            const t = taskOf(row.taskId)
            return (
              <li key={row.reviewId} className="rounded-md border border-border p-2 text-sm">
                <div className="flex flex-wrap items-baseline gap-2 text-xs text-muted-foreground">
                  <span className="font-mono">#{row.reviewId}</span>
                  <span>{t?.label}</span>
                  <span>{row.side === "decider" ? "strands-decider" : config.llmModel} leaned “{t?.options.find(([k]) => k === row.top)?.[1] ?? row.top}” at {(row.confidence * 100).toFixed(0)}%</span>
                  <span>{row.citation}</span>
                </div>
                <p className="my-1">{row.scenario}</p>
                <div className="flex flex-wrap gap-1">
                  {t?.options.map(([k, label]) => (
                    <Button key={k} size="sm" variant="outline" onClick={() => resolve(row, k)} title={label}>
                      {label.length > 48 ? `${label.slice(0, 48)}…` : label}
                    </Button>
                  ))}
                </div>
              </li>
            )
          })}
        </ul>
      </section>

      <p className="text-xs text-muted-foreground">
        Prototype decision support, not an official determination. DTR-Bench labels are self-reviewed by the
        prototype author; DoD travel SME review is pending. Latency is timed on the app server around each model
        call. Costs are estimates at list price ({config.prices.effectiveDate} Service Consumption Table):
        strands-decider is GPU node time while requests were in flight; the LLM is tokens plus the XS warehouse.
      </p>
    </div>
  )
}

function ResultPanel(props: { title: string; subtitle: string; accent: string; d?: Determination; task?: TaskInfo; busy: boolean }) {
  const { title, subtitle, accent, d, task, busy } = props
  const label = (k: string) => task?.options.find(([o]) => o === k)?.[1] ?? k
  return (
    <section className="flex min-h-48 flex-col gap-3 rounded-xl border border-border bg-card p-4">
      <header>
        <h2 className="text-lg font-semibold tracking-tight" style={{ color: accent }}>{title}</h2>
        <p className="text-xs text-muted-foreground">{subtitle}</p>
      </header>
      {busy && <p className="text-sm text-muted-foreground">Waiting for the model…</p>}
      {d?.error && <p className="text-sm text-destructive">{d.error}</p>}
      {d && !d.error && (
        <>
          {d.needsReview ? (
            <div className="rounded-md border border-amber-500/60 bg-amber-500/10 p-2 text-sm">
              <span className="font-semibold">Routed to human review.</span> Leaning “{label(d.top)}” at{" "}
              {(d.confidence * 100).toFixed(0)}%, which is below the bar for an automated determination.
            </div>
          ) : (
            <div className="rounded-md p-2 text-sm" style={{ background: "color-mix(in oklab, currentColor 6%, transparent)" }}>
              <span className="font-semibold" style={{ color: accent }}>{label(d.decision)}</span>
              <span className="ml-2 font-mono text-xs">{(d.confidence * 100).toFixed(1)}% confidence</span>
            </div>
          )}
          <ul className="flex flex-col gap-1">
            {Object.entries(d.distribution)
              .sort((a, b) => b[1] - a[1])
              .map(([k, p]) => (
                <li key={k} className="flex items-center gap-2 text-xs">
                  <span className="w-1/2 truncate" title={label(k)}>{label(k)}</span>
                  <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                    <span className="block h-full" style={{ width: `${Math.round(p * 100)}%`, background: accent }} />
                  </span>
                  <span className="w-12 text-right font-mono tabular-nums">{(p * 100).toFixed(1)}%</span>
                </li>
              ))}
          </ul>
          <p className="text-xs text-muted-foreground">
            Cites {d.citation} · {fmtMs(d.modelMs)}
            {d.inputTokens ? ` · ${d.inputTokens} input tokens` : ""}
            {d.kind !== "noul" && Object.keys(d.distribution).length === 1 ? " · stated confidence in its own pick" : ""}
          </p>
        </>
      )}
    </section>
  )
}

function Stat(props: { label: string; value: string; accent?: string; sub?: string; note?: string }) {
  const { label, value, accent, sub, note } = props
  return (
    <div className="rounded-lg bg-muted/60 px-3 py-2">
      <dt className="truncate text-[11px] uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className="font-mono text-lg font-semibold tabular-nums" style={accent ? { color: accent } : undefined}>{value}</dd>
      {sub && <dd className="font-mono text-[11px] tabular-nums text-muted-foreground">{sub}</dd>}
      {note && <dd className="text-[11px] text-destructive">{note}</dd>}
    </div>
  )
}

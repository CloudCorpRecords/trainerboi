import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Calculator, Play, Square, MessageCircle, Check, Lightbulb } from 'lucide-react'
import { post, usePoll, toast, fmt } from '../api'
import { Button, Card, Chip, Field, PageHeader, Progress, Stat, Md } from '../ui'
import { Bars, LineChart } from '../charts'
import { Mascot } from '../mascot'

const MODELS = [['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen2.5 0.5B — fastest'], ['Qwen/Qwen2.5-1.5B-Instruct', 'Qwen2.5 1.5B'], ['Qwen/Qwen2.5-3B-Instruct', 'Qwen2.5 3B']]
const tone = { running: 'info', done: 'ok', error: 'bad', crashed: 'bad', stopped: 'neutral' }

export default function MathLab() {
  const { data: runs, reload: reloadRuns } = usePoll('/api/math/runs', 4000)
  const [sel, setSel] = useState(null)
  const [cfg, setCfg] = useState({ model: MODELS[0][0], rounds: 3, problems: 400, samples: 4, max_level: 3 })
  const [busy, setBusy] = useState(false)
  useEffect(() => { if (!sel && runs?.length) setSel(runs[0].name) }, [runs, sel])
  const { data: d } = usePoll(sel ? `/api/math/${encodeURIComponent(sel)}` : null, 2500)

  const start = async () => {
    setBusy(true)
    try { const r = await post('/api/math/start', cfg); setSel(r.name); reloadRuns() } catch (e) { toast(e.message, 'error') }
    setBusy(false)
  }
  const rounds = d?.rounds || []
  const base = rounds[0], now = rounds[rounds.length - 1]
  const running = d?.state === 'running'
  const skillRows = base ? Object.keys(base.skills).map((s) => ({ label: s, before: rounds.length > 1 ? 100 * base.skills[s] : null, value: 100 * ((rounds.length > 1 ? now : base).skills[s] ?? 0) })).sort((a, b) => a.value - b.value) : []
  const doneRounds = Math.max(0, rounds.length - 1)

  return (
    <div className="page">
      <PageHeader title="Math playground 🧮" sub="No homework to collect! The model practises endless problems that check themselves, keeps only the answers that are proven right, and learns from those — just like a student with a friendly answer key." />
      <div className="grid" style={{ gridTemplateColumns: '320px 1fr', alignItems: 'start' }}>
        <div className="stack">
          <Card title="A new practice session">
            <div className="stack">
              <Field label="Model to improve"><select value={cfg.model} onChange={(e) => setCfg({ ...cfg, model: e.target.value })}>{MODELS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></Field>
              <Field label={`Rounds of practice: ${cfg.rounds}`}><input type="range" min="1" max="8" value={cfg.rounds} onChange={(e) => setCfg({ ...cfg, rounds: +e.target.value })} /></Field>
              <Field label={`New problems per round: ${cfg.problems}`}><input type="range" min="100" max="3000" step="100" value={cfg.problems} onChange={(e) => setCfg({ ...cfg, problems: +e.target.value })} /></Field>
              <Field label={`Attempts per problem: ${cfg.samples}`} hint="More attempts find more correct solutions on hard problems."><input type="range" min="2" max="8" value={cfg.samples} onChange={(e) => setCfg({ ...cfg, samples: +e.target.value })} /></Field>
              <Field label={`Hardest level: ${cfg.max_level}`} hint="1 = easy arithmetic · 5 = multi-step, big numbers"><input type="range" min="1" max="5" value={cfg.max_level} onChange={(e) => setCfg({ ...cfg, max_level: +e.target.value })} /></Field>
              <Button variant="primary" icon={Play} onClick={start} disabled={busy || running}>{running ? 'Practising right now…' : 'Start practising 🌟'}</Button>
            </div>
          </Card>
          {runs?.length > 0 && (
            <Card title="Runs">
              {runs.map((r) => (
                <div key={r.name} className="list-row" onClick={() => setSel(r.name)} style={{ background: sel === r.name ? 'var(--surface-2)' : undefined }}>
                  <div style={{ flex: 1, minWidth: 0 }}><b style={{ fontWeight: 550 }}>{r.name}</b><div className="small muted">{r.model?.split('/').pop()}</div></div>
                  <Chip tone={tone[r.state] || 'neutral'}>{r.state}</Chip>
                </div>
              ))}
            </Card>
          )}
        </div>

        <div className="stack">
          {!d ? (
            <Card><div className="empty"><Mascot mood="happy" size={110} /><h3>Nothing practised yet</h3><p>Start a session on the left. A short one (1–2 rounds, small model) takes about ten minutes.</p></div></Card>
          ) : (
            <>
              <div className="row spread">
                <div className="row"><Chip tone={tone[d.state] || 'neutral'}>{d.state === 'crashed' ? 'paused unexpectedly' : d.state === 'done' ? 'all done!' : d.state === 'running' ? 'practising' : d.state}</Chip><span className="muted">{d.phase}</span></div>
                <div className="row">
                  {running && <Button variant="danger" size="sm" icon={Square} onClick={() => post(`/api/math/${encodeURIComponent(sel)}/stop`)}>Stop</Button>}
                  {(d.final_model || d.current_model) && <Link to={`/models/${encodeURIComponent(sel + '::math')}`}><Button size="sm" icon={MessageCircle}>Chat with it · Share</Button></Link>}
                </div>
              </div>
              {running && <Progress value={d.rounds_total ? doneRounds / d.rounds_total : null} />}
              {d.error && <div className="callout bad" style={{ flexDirection: 'column' }}><b>Oops — a small hiccup{d.help ? `: ${d.help.title.toLowerCase()}` : ''}. Your progress so far is safe.</b>{d.help && <div>{d.help.advice}</div>}<details><summary>Technical details</summary><pre className="log">{d.error.slice(-1500)}</pre></details></div>}
              <div className="grid g4">
                <Stat label="Exam score" value={now ? Math.round(100 * now.overall) : '–'} unit="%" sub={rounds.length > 1 ? `started at ${Math.round(100 * base.overall)}%` : 'before training'} tone={rounds.length > 1 && now.overall > base.overall ? 'ok' : undefined} />
                <Stat label="Rounds done" value={`${doneRounds}/${d.rounds_total || '?'}`} />
                <Stat label="Verified solutions learned" value={fmt.n(d.kept || 0)} sub="proven correct by a checker" />
                <Stat label="Answers in \boxed{}" value={now ? Math.round(100 * now.boxed_pct) : '–'} unit="%" />
              </div>
              <Card title="Exam score after each round 🌱">
                <LineChart xLabel="round" yLabel="" height={190} empty="The first score appears after the opening exam…"
                  series={[{ name: 'exam score (%)', color: 'var(--accent-2)', points: rounds.map((r) => ({ x: r.round, y: 100 * r.overall })) }]} />
              </Card>
              <Card title={rounds.length > 1 ? 'Accuracy by skill — before (grey) vs now' : 'Accuracy by skill (starting point)'}>
                {skillRows.length ? <Bars rows={skillRows} /> : <div className="muted">Waiting for the opening exam…</div>}
              </Card>
              {(d.latest || d.baseline)?.samples?.length > 0 && (
                <Card title="Example answers from the latest exam">
                  {(d.latest || d.baseline).samples.slice(0, 4).map((s, i) => (
                    <div className="qa" key={i}><div className="qa-q">{s.ok ? '✅' : '❌'} {s.q}</div><div className="small muted">expected {s.gold}</div><div className="qa-box" style={{ marginTop: 6 }}>{s.out}</div></div>
                  ))}
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

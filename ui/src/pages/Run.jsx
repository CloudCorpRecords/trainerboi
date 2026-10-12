import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Check, Circle, Loader2, X, Square, RotateCcw, MessageCircle, Share2, Snowflake, Lightbulb, AlertTriangle } from 'lucide-react'
import { post, usePoll, useHardware, fmt, toast } from '../api'
import { Button, Card, Chip, Progress, Stat, PageHeader, Md } from '../ui'
import { LineChart, smooth } from '../charts'

const stateTone = { running: 'info', done: 'ok', error: 'bad', crashed: 'bad', stopped: 'neutral' }

function Timeline({ ap, sub, live }) {
  return (
    <div className="timeline">
      {ap.steps.map((s) => {
        const running = s.state === 'running'
        let extra = null
        if (running && s.id === 'data' && sub) {
          const frac = sub.total ? sub.done / sub.total : null
          extra = <div style={{ marginTop: 8 }}><Progress value={frac} /><div className="small muted" style={{ marginTop: 4 }}>{sub.phase}{sub.kept ? ` · ${fmt.n(sub.kept)} examples so far` : ''}{sub.eta_s ? ` · ~${fmt.dur(sub.eta_s)}` : ''}</div></div>
        }
        if (running && s.id === 'train' && live?.max_steps) {
          extra = <div style={{ marginTop: 8 }}><Progress value={live.step / live.max_steps} />
            <div className="small muted" style={{ marginTop: 4 }}>{live.cooling ? '🧊 Cooling the GPU — pausing keeps results correct. ' : ''}Step {fmt.n(live.step)} of {fmt.n(live.max_steps)}</div></div>
        }
        return (
          <div key={s.id} className={`tl-item ${s.state}`}>
            <div className="tl-dot">{s.state === 'done' ? <Check size={14} /> : running ? <Loader2 size={14} className="spin" /> : s.state === 'error' ? <X size={14} /> : <Circle size={8} />}</div>
            <div><div className="tl-title">{s.title}</div>{s.detail && <div className="tl-detail"><Md text={s.detail} /></div>}{extra}</div>
          </div>
        )
      })}
    </div>
  )
}

export default function Run() {
  const { name } = useParams()
  const nav = useNavigate()
  const { data: r, error } = usePoll(`/api/run/${encodeURIComponent(name)}`, 1500)
  const { hw } = useHardware()
  const [showLog, setShowLog] = useState(false)
  const g = hw?.gpu

  if (error && !r) return <div className="page"><PageHeader title="Run not found" sub={error} /></div>
  if (!r) return <div className="page"><div className="muted">Loading…</div></div>

  const ap = r.autopilot, live = r.live
  const running = r.state === 'running'
  const lossPts = r.series.filter((s) => s.loss != null).map((s) => ({ x: s.step, y: s.loss }))
  const evalPts = r.series.filter((s) => s.eval_loss != null).map((s) => ({ x: s.step, y: s.eval_loss }))
  const latest = live?.latest || {}
  const sample = r.samples.length ? r.samples[r.samples.length - 1] : null
  const results = ap?.results?.pairs
  const err = ap?.error || live?.error
  const stop = async () => { await post(`/api/run/${encodeURIComponent(name)}/stop`); toast('Stopped.') }

  return (
    <div className="page">
      <PageHeader
        title={r.goal_title || name}
        sub={<span className="row wrap" style={{ gap: 8 }}><Chip tone={stateTone[r.state]}>{r.state === 'crashed' ? 'stopped unexpectedly' : r.state}</Chip><span className="faint">{name}</span>
          {ap && <span className="faint">· {fmt.dur(((running ? Date.now() / 1000 : ap.updated_at) - ap.started_at))}</span>}</span>}
        actions={<>
          {running && <Button variant="danger" icon={Square} onClick={stop}>Stop</Button>}
          {!running && ap && r.state !== 'done' && ap.steps.some((s) => s.id === 'train' && s.state === 'done') && <Button icon={RotateCcw} onClick={async () => { try { await post(`/api/run/${encodeURIComponent(name)}/retest`); toast('Re-running the test…', 'success') } catch (e) { toast(e.message, 'error') } }}>Retry the test</Button>}
          {r.state === 'done' && <><Button icon={MessageCircle} onClick={() => nav(`/models/${encodeURIComponent(r.model_id)}`)}>Chat with it</Button><Button variant="primary" icon={Share2} onClick={() => nav(`/models/${encodeURIComponent(r.model_id)}?tab=share`)}>Share</Button></>}
        </>}
      />

      {r.state === 'done' && <div className="callout ok" style={{ marginBottom: 16 }}><Check size={16} /> Your model is ready. It's saved in <b>My models</b> — chat with it or share it from there.</div>}
      {(r.state === 'error' || r.state === 'crashed') && (
        <div className="callout bad" style={{ marginBottom: 16, flexDirection: 'column' }}>
          <div className="row"><AlertTriangle size={16} /><b>{r.help ? r.help.title : 'Something went wrong'}</b></div>
          {r.help && <div>{r.help.advice}</div>}
          {err && <details><summary style={{ cursor: 'pointer' }}>Technical details</summary><pre className="log">{String(err).slice(-1800)}</pre></details>}
        </div>
      )}
      {ap?.data_quality && <div className="callout" style={{ marginBottom: 16 }}><Lightbulb size={16} /><span>Data quality <b>{ap.data_quality.grade}</b> ({Math.round(ap.data_quality.health)}/100){ap.data_quality.top_issue ? ` — ${ap.data_quality.top_issue}` : ''}</span></div>}

      <div className="grid" style={{ gridTemplateColumns: '300px 1fr', alignItems: 'start' }}>
        <Card title="Progress">
          {ap ? <Timeline ap={ap} sub={r.sub} live={live} /> : (
            <div className="stack"><div><b>{live?.phase || 'Working'}</b></div>{live?.max_steps ? <Progress value={live.step / live.max_steps} /> : <Progress indeterminate />}</div>
          )}
          {ap?.recoveries?.length > 0 && <div className="small muted" style={{ marginTop: 8 }}>Recovered automatically {ap.recoveries.length}×</div>}
        </Card>

        <div className="stack">
          <div className="grid g4">
            <Stat label="Step" value={live?.max_steps ? `${fmt.n(live.step)}` : '–'} sub={live?.max_steps ? `of ${fmt.n(live.max_steps)}` : ''} />
            <Stat label="Loss" value={latest.loss != null ? latest.loss.toFixed(3) : '–'} sub="lower = learning" />
            <Stat label="Speed" value={live?.steps_per_s ? live.steps_per_s.toFixed(2) : '–'} unit="steps/s" sub={latest.tokens_per_s ? `${fmt.n(latest.tokens_per_s)} tokens/s` : ''} />
            <Stat label="Time left" value={running && live?.eta_s ? fmt.dur(live.eta_s) : '–'} sub={live?.cooling ? 'GPU cooling' : ''} />
          </div>
          <Card title="Learning curve">
            <LineChart yLabel="loss " series={[
              { name: 'loss', color: 'var(--accent-2)', points: smooth(lossPts) },
              { name: 'raw', color: 'var(--accent-2)', points: lossPts, faint: true },
              ...(evalPts.length ? [{ name: 'test loss', color: 'var(--ok)', points: evalPts, dashed: true }] : []),
            ]} empty={running ? 'The curve appears once training starts…' : 'No training curve for this run.'} />
          </Card>
          <div className="grid g3">
            <Stat label="GPU now" value={g?.util != null ? Math.round(g.util) : '–'} unit="%" />
            <Stat label="Temperature" value={g?.temp != null ? Math.round(g.temp) : '–'} unit="°C" tone={g?.temp >= 85 ? 'bad' : g?.temp >= 78 ? 'warn' : 'ok'} />
            <Stat label="GPU memory" value={g?.vram_used_gb != null ? g.vram_used_gb.toFixed(1) : '–'} unit={`/ ${g?.vram_total_gb ? Math.round(g.vram_total_gb) : '–'} GB`} />
          </div>
          {sample && (
            <Card title={`What the model says now (step ${sample.step})`}>
              <div className="muted small" style={{ marginBottom: 6 }}>{sample.prompt}</div>
              <div style={{ whiteSpace: 'pre-wrap' }}>{sample.output}</div>
            </Card>
          )}
        </div>
      </div>

      {results?.length > 0 && (
        <Card title="Before vs. after training" className="" style={{ marginTop: 18 }}>
          <p className="muted small" style={{ marginBottom: 6 }}>Same questions to the original model and to yours.</p>
          {results.map((p, i) => (
            <div className="qa" key={i}>
              <div className="qa-q">{p.reference ? '📘 ' : ''}{p.prompt}</div>
              <div className="qa-pair">
                <div className="qa-box"><h5>Before</h5>{p.before}</div>
                <div className="qa-box after"><h5>After</h5>{p.after}</div>
              </div>
              {p.reference && <div className="small faint" style={{ marginTop: 6 }}>Reference answer: {p.reference.slice(0, 300)}</div>}
            </div>
          ))}
        </Card>
      )}

      <div style={{ marginTop: 18 }}>
        <button className="btn btn-ghost btn-sm" onClick={() => setShowLog(!showLog)}>{showLog ? 'Hide' : 'Show'} technical log</button>
        {showLog && <pre className="log" style={{ marginTop: 8 }}>{r.log}</pre>}
      </div>
    </div>
  )
}

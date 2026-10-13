import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Database, Stethoscope, Wand2, Wrench } from 'lucide-react'
import { j, post, usePoll, useJob, toast, fmt } from '../api'
import { Button, Card, Chip, Empty, Field, PageHeader, Progress } from '../ui'
import { Bars } from '../charts'
import { Mascot } from '../mascot'

const GRADE = { A: '#3ecf8e', B: '#9bd34a', C: '#f5a524', D: '#ff8a3d', F: '#f0556a' }
const SEV = { high: 'bad', medium: 'warn', low: 'neutral' }

function Grade({ g, size = 64 }) {
  if (!g) return <div className="grade" style={{ width: size, height: size, background: 'var(--surface-3)', color: 'var(--faint)', fontSize: size / 2 }}>?</div>
  return <div className="grade" style={{ width: size, height: size, fontSize: size / 1.9, background: `${GRADE[g]}22`, color: GRADE[g] }}>{g}</div>
}

function Detail({ ds, onChange }) {
  const [job, follow] = useJob()
  const { data: a, reload } = usePoll(`/api/datasets/${encodeURIComponent(ds.name)}/analysis`, 600000)
  const [drop, setDrop] = useState([])
  const [dedup, setDedup] = useState(false)
  const [minScore, setMinScore] = useState(0)
  const [out, setOut] = useState('')
  useEffect(() => {
    if (a) { setDrop(Object.entries(a.flags).filter(([, v]) => v.default_drop && v.pct >= 0.5).map(([k]) => k)); setDedup(a.recommendations.some((r) => r.action?.dedup_prompts)); setOut(`${ds.name}-clean`) }
  }, [a?.analyzed_at, ds.name])
  useEffect(() => { if (job?.state === 'done') { reload(); onChange() } if (job?.state === 'error') toast(job.error, 'error') }, [job?.state])

  const analyze = async () => { try { const r = await post(`/api/datasets/${encodeURIComponent(ds.name)}/analyze`); follow(r.job) } catch (e) { toast(e.message, 'error') } }
  const clean = async () => {
    try { const r = await post(`/api/datasets/${encodeURIComponent(ds.name)}/clean`, { out, drop, dedup, min_score: minScore }); toast(`Creating “${r.name}”… your original is untouched.`, 'success'); setTimeout(onChange, 4000) }
    catch (e) { toast(e.message, 'error') }
  }
  return (
    <div className="stack">
      <Card>
        <div className="row spread">
          <div className="row" style={{ gap: 16 }}><Grade g={a?.grade || ds.grade} /><div><h3 style={{ fontSize: 18 }}>{ds.name}</h3>
            <div className="muted small">{fmt.n(ds.train)} training · {fmt.n(ds.eval)} test examples · {ds.format}{ds.tokens ? ` · ${(ds.tokens / 1e6).toFixed(1)}M tokens` : ''}</div>
            {a && <div className="small faint">Health {Math.round(a.health)}/100 · checked {fmt.n(a.sampled)} examples</div>}</div></div>
          <Button icon={Stethoscope} variant="primary" onClick={analyze} disabled={job?.state === 'running'}>{job?.state === 'running' ? 'Taking a look…' : a ? 'Check again' : 'Check its health'}</Button>
        </div>
        {job?.state === 'running' && <div style={{ marginTop: 14 }}><Progress indeterminate /></div>}
      </Card>
      {!a ? <div className="muted">Press “Check its health” and we'll take a gentle look. 🔍</div> : (
        <>
          <Card title="Little things to tidy">
            <div className="stack" style={{ gap: 8 }}>{a.recommendations.map((r, i) => <div key={i} className="row" style={{ alignItems: 'flex-start' }}><Chip tone={SEV[r.severity]}>{r.severity}</Chip><span>{r.text}</span></div>)}</div>
          </Card>
          {Object.keys(a.flags).length > 0 && (
            <Card title="Problems found">
              <Bars max={Math.max(10, ...Object.values(a.flags).map((f) => f.pct))} unit="%" rows={Object.values(a.flags).map((f) => ({ label: f.label, value: f.pct, color: f.default_drop ? 'var(--bad)' : 'var(--warn)' }))} />
            </Card>
          )}
          <Card title="Make a tidied-up copy ✨" action={<span className="small faint">your original is never touched</span>}>
            <div className="stack">
              <div className="row wrap">{Object.entries(a.flags).map(([k, v]) => (
                <label key={k} className="src-chip" style={{ cursor: 'pointer' }}><input type="checkbox" checked={drop.includes(k)} onChange={(e) => setDrop(e.target.checked ? [...drop, k] : drop.filter((x) => x !== k))} /><span>{v.label} ({v.pct}%)</span></label>))}</div>
              <label className="row"><input type="checkbox" checked={dedup} onChange={(e) => setDedup(e.target.checked)} /> Remove repeated questions (also fixes test/train overlap)</label>
              <div className="grid g2"><Field label={`Also remove anything scoring below ${minScore}`}><input type="range" min="0" max="90" step="5" value={minScore} onChange={(e) => setMinScore(+e.target.value)} /></Field>
                <Field label="Name of the cleaned copy"><input type="text" value={out} onChange={(e) => setOut(e.target.value)} /></Field></div>
              <div><Button variant="primary" icon={Wand2} onClick={clean}>Make the tidy copy</Button></div>
            </div>
          </Card>
          {a.worst?.length > 0 && <Card title="Lowest-scoring examples">{a.worst.slice(0, 4).map((w, i) => <div className="qa" key={i}><div className="row"><Chip tone="bad">score {w.score}</Chip><span className="small muted">{w.flags.join(', ')}</span></div><div className="small" style={{ marginTop: 6 }}><b>{w.prompt}</b></div><div className="small muted">{w.response}</div></div>)}</Card>}
        </>
      )}
    </div>
  )
}

export default function Data() {
  const { data: list, reload } = usePoll('/api/datasets', 5000)
  const [sel, setSel] = useState(null)
  const items = (list || []).filter((d) => d.state === 'done' || d.state === 'running')
  useEffect(() => { if (!sel && items.length) setSel(items[0].name) }, [list])
  const cur = items.find((d) => d.name === sel)
  return (
    <div className="page">
      <PageHeader title="My data 🌱" sub="A little health check for your examples. Tidy them up now and training goes smoother later."
        actions={<Link to="/advanced"><Button icon={Wrench}>Open the toolbox</Button></Link>} />
      {!list ? <div className="row muted"><Mascot mood="working" size={48} /> Looking through your shelves…</div> : !items.length ? (
        <Empty icon={() => <Mascot mood="sleepy" size={110} />} title="The shelves are empty" text="Datasets appear here automatically when you teach a model. For full control (websites, YouTube, GitHub…) peek into the Toolbox." action={<Link to="/train"><Button variant="primary">Teach a model ✨</Button></Link>} />
      ) : (
        <div className="grid" style={{ gridTemplateColumns: '300px 1fr', alignItems: 'start' }}>
          <Card pad={false}><div style={{ padding: 6 }}>
            {items.map((d) => (
              <div key={d.name} className="list-row" onClick={() => setSel(d.name)} style={{ padding: '10px 12px', borderRadius: 8, background: sel === d.name ? 'var(--surface-2)' : undefined }}>
                <Grade g={d.grade} size={34} /><div style={{ flex: 1, minWidth: 0 }}><div style={{ fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{d.name}</div><div className="small muted">{fmt.n(d.train)} examples · {d.format}</div></div>
                {d.state === 'running' && <Chip tone="info">building</Chip>}
              </div>))}
          </div></Card>
          {cur && <Detail key={cur.name} ds={cur} onChange={reload} />}
        </div>
      )}
    </div>
  )
}

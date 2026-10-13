import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Soup, Stethoscope, Wrench as Spanner } from 'lucide-react'
import { post, usePoll, toast, fmt } from '../api'
import { Button, Card, Chip, Field, PageHeader, Progress, Stat } from '../ui'
import { Mascot } from '../mascot'
import { Ingredients, EMPTY_SRC } from '../ingredients'
import { cuteName } from '../recipes'
import { earn } from '../rewards'

const FORMATS = [
  { id: 'sft', emoji: '💬', title: 'Questions & answers', blurb: 'Chat-style examples. The usual choice for teaching a model something.' },
  { id: 'pretrain', emoji: '📖', title: 'Plain reading text', blurb: 'Lots of text to read, for building a tiny model from scratch.' },
  { id: 'dpo', emoji: '⚖️', title: 'Good vs. better answers', blurb: 'Pairs of answers, for preference tuning. Needs data that already has pairs.' },
]

function Cooking({ name, onDone }) {
  const { data: s } = usePoll(`/api/builder/${encodeURIComponent(name)}`, 1200)
  const state = s?.state
  useEffect(() => { if (state === 'done') { earn('gathered'); onDone?.() } }, [state])
  const pr = s?.progress
  const frac = pr?.total ? pr.done / pr.total : null
  const mood = state === 'done' ? 'cheer' : state === 'error' || state === 'crashed' ? 'oops' : 'working'
  return (
    <Card>
      <div className="kitchen">
        <div className="pot"><Mascot mood={mood} size={110} /><div className="pot-steam"><i /><i /><i /></div></div>
        <div className="stack" style={{ gap: 10, flex: 1 }}>
          <h3 style={{ fontSize: 20 }}>{state === 'done' ? 'Ready to serve! 🍲' : state === 'error' || state === 'crashed' ? 'Oops — the soup had a hiccup' : 'Stirring the soup…'}</h3>
          <div className="muted">{s?.phase || 'Getting the ingredients ready…'}{pr?.total ? ` · ${fmt.n(pr.done)} of ${fmt.n(pr.total)} ${pr.unit || ''}` : ''}{pr?.eta_s ? ` · about ${fmt.dur(pr.eta_s)} to go` : ''}</div>
          {state !== 'done' && state !== 'error' && state !== 'crashed' && <Progress value={frac} />}
          {s?.counters && Object.keys(s.counters).length > 0 && <div className="row wrap">{Object.entries(s.counters).slice(0, 5).map(([k, v]) => <Chip key={k} tone="accent">{k}: {fmt.n(v)}</Chip>)}</div>}
          {s?.warnings?.map((w, i) => <div key={i} className="small faint">• {w}</div>)}
          {(state === 'error' || state === 'crashed') && (
            <div className="callout warn" style={{ flexDirection: 'column' }}>
              {s.help ? <><b>{s.help.title}</b><div>{s.help.advice}</div></> : <div>Something didn't work out. Your files are safe.</div>}
              <details><summary style={{ cursor: 'pointer' }}>Technical details</summary><pre className="log">{(s.error || s.log || '').slice(-1500)}</pre></details>
            </div>
          )}
          {state === 'done' && s.stats && (
            <>
              <div className="grid g3">
                <Stat label="Training examples" value={fmt.n(s.stats.train_examples)} /><Stat label="Kept for testing" value={fmt.n(s.stats.eval_examples)} /><Stat label="Reading material" value={(s.stats.total_tokens / 1e6).toFixed(1)} unit="M tokens" />
              </div>
              {s.dropped && Object.keys(s.dropped).length > 0 && <div className="small muted">Tidied away: {Object.entries(s.dropped).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k, v]) => `${fmt.n(v)} ${k}`).join(' · ')}</div>}
              <div className="row wrap">
                <Link to="/data"><Button icon={Stethoscope}>Check its health</Button></Link>
                <Link to={`/workshop?dataset=${encodeURIComponent(s.stats.train_path || '')}`}><Button variant="primary" icon={Spanner}>Teach a model with it 🔧</Button></Link>
              </div>
            </>
          )}
        </div>
      </div>
    </Card>
  )
}

export default function Gather() {
  const [fmtId, setFmt] = useState('sft')
  const [files, setFiles] = useState([])
  const [src, setSrc] = useState(EMPTY_SRC)
  const [synth, setSynth] = useState(true)
  const [pii, setPii] = useState(false)
  const [name, setName] = useState('')
  const [persona, setPersona] = useState('')
  const [maxRows, setMaxRows] = useState(20000)
  const [started, setStarted] = useState(null)
  const [busy, setBusy] = useState(false)
  const hasAny = files.length > 0 || Object.values(src).some((a) => a.length)

  const start = async () => {
    setBusy(true)
    try {
      const nm = name.trim() || cuteName()
      const r = await post('/api/builder/start', { name: nm, format: fmtId, sources: { ...src, files: files.map((f) => f.path) }, synthetic: synth && fmtId === 'sft', redact_pii: pii,
        system_prompt: persona, max_rows: maxRows, max_synth_chunks: 300 })
      setStarted(r.name)
    } catch (e) { toast(e.message, 'error') }
    setBusy(false)
  }

  return (
    <div className="page" style={{ maxWidth: 980 }}>
      <PageHeader title="Gather data 🥣" sub="Toss in whatever you have — files, links, videos, repos, a topic — and we'll cook it into tidy examples a model can learn from." />
      {started ? (
        <div className="stack">
          <Cooking name={started} />
          <div><Button variant="ghost" onClick={() => setStarted(null)}>← Cook something else</Button></div>
        </div>
      ) : (
        <div className="stack">
          <Card title="1 · What kind of examples?">
            <div className="goal-grid">
              {FORMATS.map((f) => (
                <button key={f.id} className={`goal ${fmtId === f.id ? 'on' : ''}`} onClick={() => setFmt(f.id)}><div className="ico">{f.emoji}</div><h4>{f.title}</h4><p>{f.blurb}</p></button>
              ))}
            </div>
          </Card>
          <Card title="2 · Your ingredients"><Ingredients files={files} setFiles={setFiles} src={src} setSrc={setSrc} /></Card>
          <Card title="3 · A pinch of options">
            <div className="stack">
              {fmtId === 'sft' && (
                <label className="row" style={{ alignItems: 'flex-start' }}><input type="checkbox" checked={synth} onChange={(e) => setSynth(e.target.checked)} style={{ marginTop: 4 }} />
                  <span><b>Let a helper AI write questions and answers from my documents</b><div className="small muted">Best when you give it articles, notes or pages instead of ready-made Q&A. Skip it if your data already has questions and answers.</div></span></label>
              )}
              <label className="row"><input type="checkbox" checked={pii} onChange={(e) => setPii(e.target.checked)} /> <span>Hide private details like emails, phone numbers and keys <span className="small faint">(a nice habit before sharing)</span></span></label>
              {fmtId === 'sft' && <Field label="Give every example a personality (optional)" hint="Added to every example as a gentle instruction."><input type="text" value={persona} onChange={(e) => setPersona(e.target.value)} placeholder="You are Max, a cheerful helper…" /></Field>}
              <div className="grid g2">
                <Field label={`Take at most ${fmt.n(maxRows)} examples from big sources`}><input type="range" min="1000" max="100000" step="1000" value={maxRows} onChange={(e) => setMaxRows(+e.target.value)} /></Field>
                <Field label="Name this batch" hint="Leave empty for a surprise name."><input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. my-notes" /></Field>
              </div>
            </div>
          </Card>
          <div className="wiz-foot" style={{ justifyContent: 'flex-end' }}>
            <Button variant="primary" icon={Soup} onClick={start} disabled={!hasAny || busy}>{busy ? 'Lighting the stove…' : "Let's cook! 🍳"}</Button>
          </div>
        </div>
      )}
    </div>
  )
}

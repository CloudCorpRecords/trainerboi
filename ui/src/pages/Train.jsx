import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { UploadCloud, X, ArrowRight, ArrowLeft, Link2, Check, Lightbulb, ChevronDown } from 'lucide-react'
import { j, post, usePoll, toast } from '../api'
import { Button, Card, Chip, Field, PageHeader, Md } from '../ui'
import { Mascot } from '../mascot'
import { useCompanion } from '../companion'
import { RECIPES, cuteName } from '../recipes'
import { perkOn } from '../rewards'

const EMOJI = { docs: '📄', persona: '💬', instruct: '🧠', code: '💻', prefer: '⚖️', scratch: '🧱' }
const SRC_LABEL = { urls: 'Website', hf: 'Hugging Face', wikipedia: 'Wikipedia', arxiv: 'Papers', github: 'GitHub', youtube: 'YouTube', folders: 'On this PC' }
const EMPTY_SRC = { urls: [], hf: [], wikipedia: [], arxiv: [], github: [], youtube: [], folders: [] }

function Steps({ step }) {
  const items = ['What', 'Examples', 'Ready']
  return (
    <div className="steps">
      {items.map((t, i) => (
        <div key={t} className="row" style={{ gap: 10 }}>
          <div className={`step-dot ${i === step ? 'on' : i < step ? 'done' : ''}`}><i>{i < step ? <Check size={13} /> : i + 1}</i>{t}</div>
          {i < items.length - 1 && <div className="step-line" />}
        </div>
      ))}
    </div>
  )
}

export default function Train() {
  const nav = useNavigate()
  const [params] = useSearchParams()
  const { data: meta } = usePoll('/api/guided/meta', 600000)
  const { data: datasets } = usePoll('/api/datasets', 30000)
  const [step, setStep] = useState(0)
  const [goal, setGoal] = useState(params.get('goal') || 'instruct')
  const [files, setFiles] = useState([])
  const [src, setSrc] = useState(EMPTY_SRC)
  const [paste, setPaste] = useState('')
  const [persona, setPersona] = useState('')
  const [dataset, setDataset] = useState('')
  const [effort, setEffort] = useState('balanced')
  const [model, setModel] = useState('')
  const [name, setName] = useState('')
  const [hint, setHint] = useState(null)
  const [plan, setPlan] = useState(null)
  const [starting, setStarting] = useState(false)
  const [over, setOver] = useState(false)
  const [adv, setAdv] = useState(false)
  const [recipe, setRecipe] = useState(null)
  const fileRef = useRef(null)
  const { say } = useCompanion()
  useEffect(() => {
    const lines = ['What would you like to teach? Pick whatever sounds fun. 🌈', 'Show me some examples — files, links, or just a topic you\'re curious about. 📚', 'Almost ready! Take a peek at the plan, then press the big button. 🚂']
    say(lines[step], step === 1 ? 'think' : 'happy', 6500)
  }, [step])

  const g = meta?.goals.find((x) => x.key === goal)
  useEffect(() => {      // arriving from a recipe card: fill things in for the person
    const id = params.get('recipe')
    const r = RECIPES.find((x) => x.id === id)
    if (!r || r.link) return
    setRecipe(r); setGoal(r.goal); setEffort(r.effort || 'balanced'); if (r.persona) setPersona(r.persona)
    setStep(r.needs ? 1 : 2)
    say(`${r.emoji} ${r.title} — good choice!`, 'happy', 4500)
  }, [params.get('recipe')])
  const hasData = files.length > 0 || Object.values(src).some((a) => a.length) || !!dataset
  const needs = g?.needs === 'files' && !hasData

  useEffect(() => {
    if (step !== 2 || !g) return
    let stop = false
    post('/api/guided/plan', { goal, effort, has_data: hasData, persona, model }).then((p) => !stop && setPlan(p)).catch(() => {})
    return () => { stop = true }
  }, [step, goal, effort, hasData, model, persona])

  const upload = async (list) => {
    if (!list?.length) return
    const fd = new FormData()
    for (const f of list) fd.append('files', f)
    try {
      const out = await fetch('/api/upload', { method: 'POST', body: fd }).then((r) => r.json())
      const merged = [...files, ...out]
      setFiles(merged)
      const r = await post('/api/intake', { text: '', files: out.map((o) => o.path) })
      setHint(r)
    } catch (e) { toast(e.message, 'error') }
  }

  const addPaste = async () => {
    if (!paste.trim()) return
    try {
      const r = await post('/api/intake', { text: paste, files: [] })
      setSrc((s) => {
        const n = { ...s }
        const map = { folders: 'folders', urls: 'urls', hf: 'hf', wikipedia: 'wikipedia', arxiv: 'arxiv', github: 'github', youtube: 'youtube' }
        for (const k in map) n[k] = [...new Set([...s[k], ...(r.fields[k] || [])])]
        return n
      })
      setHint(r)
      setPaste('')
    } catch (e) { toast(e.message, 'error') }
  }

  const removeSrc = (k, v) => setSrc((s) => ({ ...s, [k]: s[k].filter((x) => x !== v) }))

  const start = async () => {
    setStarting(true)
    try {
      const r = await post('/api/guided/start', { goal, effort, files: files.map((f) => f.path), ...src, persona, model, name: name.trim() || cuteName(), dataset })
      nav(`/run/${encodeURIComponent(r.name)}`)
    } catch (e) { toast(e.message, 'error'); setStarting(false) }
  }

  const suggest = hint?.goal && hint.goal !== goal ? meta?.goals.find((x) => x.key === hint.goal) : null

  if (!meta) return <div className="page"><PageHeader title="Teach a model" /><div className="row muted"><Mascot mood="working" size={56} /> Warming up…</div></div>

  return (
    <div className="page" style={{ maxWidth: 980 }}>
      <PageHeader title="Teach a model" sub="Three gentle steps. We'll pick the model and every setting for you — and you can change anything you like." />
      <Steps step={step} />

      {recipe && step > 0 && <div className="callout" style={{ marginBottom: 16, alignItems: 'center' }}><Mascot mood="happy" size={42} /><div><b>{recipe.emoji} Recipe: {recipe.title}.</b> {recipe.needs || 'Everything is filled in — you can still change anything.'}</div></div>}
      {step === 0 && (
        <>
        <div className="recipe-row">
          <div className="small muted" style={{ marginBottom: 8, fontWeight: 800 }}>✨ Or start from a recipe — one click and it's filled in</div>
          <div className="recipes">{RECIPES.map((r) => (r.link ? <a key={r.id} className="recipe" href={`#${r.link}`}><span>{r.emoji}</span><b>{r.title}</b><small>{r.blurb}</small></a>
            : <button key={r.id} className="recipe" onClick={() => { setRecipe(r); setGoal(r.goal); setEffort(r.effort || 'balanced'); setPersona(r.persona || ''); setStep(r.needs ? 1 : 2) }}><span>{r.emoji}</span><b>{r.title}</b><small>{r.blurb}</small></button>))}</div>
        </div>
        <div className="goal-grid">
          {meta.goals.map((x) => {
            return (
              <button key={x.key} className={`goal ${goal === x.key ? 'on' : ''}`} onClick={() => setGoal(x.key)}>
                <div className="ico">{EMOJI[x.key] || '✨'}</div><h4>{x.title}</h4><p>{x.blurb}</p>
              </button>
            )
          })}
        </div>
        </>
      )}

      {step === 1 && (
        <div className="stack">
          {g?.key === 'persona' && (
            <Card title="Give it a personality 🎭">
              <Field label="Describe your assistant's name, tone and role" hint="Taught into every example. Leave empty for a neutral assistant.">
                <textarea value={persona} onChange={(e) => setPersona(e.target.value)} placeholder="You are Max, a cheerful support agent for AcmeBank. You answer briefly, warmly, and always suggest a next step." />
              </Field>
            </Card>
          )}
          <Card title={g?.needs === 'files' ? 'Your examples (this goal learns from them)' : 'Your examples (optional — we can bring some)'}>
            <p className="muted" style={{ marginBottom: 14 }}>{g?.data_hint}</p>
            <div className={`drop ${over ? 'over' : ''}`} onClick={() => fileRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); upload([...e.dataTransfer.files]) }}>
              <UploadCloud size={26} style={{ marginBottom: 6 }} />
              <div style={{ fontSize: 17 }}><b>Drop your files here</b> 🌱 or click to choose</div>
              <div className="small faint" style={{ marginTop: 4 }}>PDF, Word, text, web pages, CSV, JSON, code, zip, audio/video…</div>
              <input ref={fileRef} type="file" multiple hidden onChange={(e) => { upload([...e.target.files]); e.target.value = '' }} />
            </div>
            <div style={{ marginTop: 16 }}>
              <Field label="Or paste anything" hint="Links, YouTube videos, GitHub repos, Hugging Face datasets, folder paths — or just name a topic you're curious about. One per line.">
                <textarea value={paste} onChange={(e) => setPaste(e.target.value)} placeholder={'beekeeping basics\nhttps://docs.python.org/3/tutorial/\nhttps://github.com/owner/repo\nyahma/alpaca-cleaned'} />
              </Field>
              <div className="row" style={{ marginTop: 8 }}><Button icon={Link2} onClick={addPaste} disabled={!paste.trim()}>Add it</Button></div>
            </div>
            {(files.length > 0 || Object.values(src).some((a) => a.length)) && (
              <div className="row wrap" style={{ marginTop: 16 }}>
                {files.map((f) => <span key={f.path} className="src-chip"><Chip>File</Chip><span>{f.name}</span><button onClick={() => setFiles(files.filter((x) => x.path !== f.path))}><X size={13} /></button></span>)}
                {Object.entries(src).flatMap(([k, a]) => a.map((v) => (
                  <span key={k + v} className="src-chip"><Chip tone="accent">{SRC_LABEL[k]}</Chip><span>{v}</span><button onClick={() => removeSrc(k, v)}><X size={13} /></button></span>
                )))}
              </div>
            )}
            {suggest && (
              <div className="callout" style={{ marginTop: 16 }}>
                <Lightbulb size={16} style={{ flexShrink: 0, marginTop: 2 }} />
                <div style={{ flex: 1 }}>{hint.advice ? <Md text={hint.advice.replace(/###[^\n]*\n/, '').split('\n\n')[1] || ''} /> : 'This looks like a different goal.'} We'd suggest <b>{suggest.title}</b>.</div>
                <Button size="sm" onClick={() => { setGoal(suggest.key); setHint(null) }}>Switch</Button>
              </div>
            )}
            {datasets?.filter((d) => d.state === 'done').length > 0 && (
              <div style={{ marginTop: 16 }}>
                <Field label="Or reuse a dataset you already built">
                  <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
                    <option value="">None</option>
                    {datasets.filter((d) => d.state === 'done').map((d) => <option key={d.name} value={d.name}>{d.name} · {fmtN(d.train)} examples{d.grade ? ` · quality ${d.grade}` : ''}</option>)}
                  </select>
                </Field>
              </div>
            )}
          </Card>
          {needs && <div className="callout warn"><Mascot mood="happy" size={44} /><div>This one learns from <b>your own examples</b> — pop in a file, a link, a repo or a topic and we can keep going. 🌱</div></div>}
        </div>
      )}

      {step === 2 && (
        <div className="stack">
          <Card title="How much time would you like to spend?">
            <div className="effort">
              {meta.efforts.map((e) => (
                <button key={e.key} className={effort === e.key ? 'on' : ''} onClick={() => setEffort(e.key)}><b>{e.title}</b><span>{e.blurb}</span></button>
              ))}
            </div>
          </Card>
          <Card title="Here's the plan 🗺️">
            {!plan ? <div className="row muted"><Mascot mood="working" size={48} /> Working out the cosiest settings…</div> : (
              <>
                <div className="row wrap" style={{ marginBottom: 10 }}>
                  <Chip tone="accent">About {plan.minutes} min</Chip><Chip>~{plan.vram_gb} GB GPU memory</Chip>
                  {plan.downloads_model && <Chip>First run downloads the model</Chip>}
                </div>
                {plan.steps.map((s, i) => <div className="plan-step" key={i}><i>{i + 1}</i><div><b>{s.title}</b><div className="muted">{s.text}</div></div></div>)}
                <details style={{ marginTop: 8 }}><summary className="muted" style={{ cursor: 'pointer' }}>Why these settings? 🤔</summary>
                  <ul className="muted" style={{ paddingLeft: 18 }}>{plan.reasons.map((r, i) => <li key={i} style={{ margin: '6px 0' }}><Md text={r} /></li>)}</ul>
                </details>
              </>
            )}
          </Card>
          <div>
            <button className="btn btn-ghost btn-sm" onClick={() => setAdv(!adv)}><ChevronDown size={14} style={{ transform: adv ? 'rotate(180deg)' : '' }} />Advanced</button>
            {adv && (
              <div className="grid g2" style={{ marginTop: 10 }}>
                <Field label="Base model"><select value={model} onChange={(e) => setModel(e.target.value)}>{meta.models.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}</select></Field>
                <Field label="Run name" hint="Leave empty for a surprise name 🎲"><div className="row" style={{ gap: 8 }}><input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="a surprise…" style={{ flex: 1 }} />{perkOn('luckynames') && <button className="btn btn-ghost btn-sm" onClick={() => setName(cuteName())} title="Roll a fresh surprise name">🎲</button>}</div></Field>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="wiz-foot">
        <Button icon={ArrowLeft} variant="ghost" onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}>Back</Button>
        {step < 2
          ? <Button variant="primary" onClick={() => setStep(step + 1)} disabled={step === 1 && needs}>Next <ArrowRight size={16} /></Button>
          : <Button variant="primary" onClick={start} disabled={starting || needs}>{starting ? 'Hopping on…' : "Let's go! 🚂"}</Button>}
      </div>
    </div>
  )
}

const fmtN = (n) => (n == null ? '–' : Number(n).toLocaleString())

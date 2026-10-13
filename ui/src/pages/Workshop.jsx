import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams, Link } from 'react-router-dom'
import { Wand2, ChevronDown, Rocket } from 'lucide-react'
import { j, post, usePoll, toast } from '../api'
import { Button, Card, Chip, Field, PageHeader, Md } from '../ui'
import { Mascot } from '../mascot'
import { useCompanion } from '../companion'
import { cuteName } from '../recipes'
import { earn, perkOn } from '../rewards'

const NEEDS = { sft: ['sft'], dpo: ['dpo'], orpo: ['dpo'], kto: ['dpo'], grpo: ['sft'], pretrain: ['pretrain'] }
const FORMAT_WORD = { sft: 'question-and-answer', dpo: 'good-vs-better-answer', pretrain: 'plain-text' }
const MODELS = [['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen2.5 0.5B — fastest'], ['Qwen/Qwen2.5-1.5B-Instruct', 'Qwen2.5 1.5B'], ['Qwen/Qwen2.5-3B-Instruct', 'Qwen2.5 3B'],
  ['Qwen/Qwen2.5-7B-Instruct', 'Qwen2.5 7B — biggest'], ['unsloth/Llama-3.2-1B-Instruct', 'Llama 3.2 1B'], ['HuggingFaceTB/SmolLM2-360M-Instruct', 'SmolLM2 360M — tiny']]
const EFFORT = [['quick', '⚡ Quick'], ['balanced', '⚖️ Balanced'], ['best', '🏆 Best']]

export default function Workshop() {
  const nav = useNavigate()
  const [params] = useSearchParams()
  const { data: meta } = usePoll('/api/methods', 600000)
  const { data: datasets } = usePoll('/api/datasets', 20000)
  const { say } = useCompanion()
  const [method, setMethod] = useState(null)         // method card id
  const [have, setHave] = useState(''); const [want, setWant] = useState(''); const [advice, setAdvice] = useState(null)
  const [dataset, setDataset] = useState(params.get('dataset') || '')
  const [effort, setEffort] = useState('balanced')
  const [s, setS] = useState({ model: MODELS[0][0], mode: 'qlora', steps: 300, lr: 0.0002, batch: 4, grad_accum: 4, seq_len: 1024, lora_r: 16, size: 'tiny', reward: 'concise' })
  const [reasons, setReasons] = useState(null)
  const [minutes, setMinutes] = useState(null)
  const [adv, setAdv] = useState(false)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)

  const [lockMsg, setLockMsg] = useState(false)
  const isLocked = (m) => !!m && !m.redirect && m.level !== 'Start here' && m.level !== 'Unique' && !perkOn('advanced')
  const card = meta?.methods.find((m) => m.id === method)
  const run = card?.run
  const picked = datasets?.find((d) => d.train_path === dataset)
  const wrongFormat = picked && run && !NEEDS[run].includes(picked.format)

  const quiz = async (h, w) => {
    if (!h || !w) return
    try { const r = await post('/api/methods/recommend', { have: h, want: w }); setAdvice(r); const m = meta.methods.find((x) => x.id === r.id); if (isLocked(m)) { setMethod(null); setLockMsg(true) } else { setMethod(r.id); setLockMsg(false) }; say('How about this one? 💡', 'think', 4500) } catch (e) { toast(e.message, 'error') }
  }
  const recommend = async () => {
    if (!run) return
    try {
      const r = await post('/api/train/recommend', { method: run, dataset, effort })
      setS((x) => ({ ...x, model: r.model || x.model, mode: card.mode && card.mode !== 'qlora' && run === 'sft' ? card.mode : (r.mode || x.mode), steps: r.steps, lr: r.lr, batch: r.batch, grad_accum: r.grad_accum, seq_len: r.seq_len, lora_r: r.lora_r, size: r.size || x.size }))
      setReasons(r.reasons); setMinutes(r.minutes)
      say('Here are some cozy settings for you. ✨', 'happy', 4000)
    } catch (e) { toast(e.message, 'error') }
  }
  useEffect(() => { if (card) { setS((x) => ({ ...x, mode: card.mode || x.mode })); setReasons(null); setMinutes(null) } }, [method])

  const start = async () => {
    setBusy(true)
    try {
      const r = await post('/api/train/start', { method: run, dataset, ...s, name: name.trim() || cuteName() })
      earn('workshop'); nav(`/run/${encodeURIComponent(r.name)}`)
    } catch (e) { toast(e.message, 'error'); setBusy(false) }
  }
  const set = (k) => (e) => setS({ ...s, [k]: e.target.type === 'number' || e.target.type === 'range' ? +e.target.value : e.target.value })
  const ready = run && dataset

  if (!meta) return <div className="page"><PageHeader title="Workshop 🔧" /><div className="row muted"><Mascot mood="working" size={50} /> Sweeping the workshop…</div></div>

  return (
    <div className="page" style={{ maxWidth: 1040 }}>
      <PageHeader title="Workshop 🔧" sub="Every knob, with a friendly label. Pick a way of teaching, pick your examples, and tweak whatever you like — or let us suggest cozy settings." />
      <div className="stack">
        <Card title="1 · How shall we teach it?" action={<Chip tone="accent">not sure? ask below</Chip>}>
          <div className="quiz row wrap">
            <Field label="What do you have?"><select value={have} onChange={(e) => { setHave(e.target.value); quiz(e.target.value, want) }}><option value="">…</option>{meta.have.map((h) => <option key={h}>{h}</option>)}</select></Field>
            <Field label="What should it do?"><select value={want} onChange={(e) => { setWant(e.target.value); quiz(have, e.target.value) }}><option value="">…</option>{meta.want.map((h) => <option key={h}>{h}</option>)}</select></Field>
          </div>
          {lockMsg && <div className="callout warn" style={{ margin: '12px 0' }}><Mascot mood="sleepy" size={40} /><div>That one opens up after your first finished model (it's a little trickier). <Link to="/train" style={{ textDecoration: 'underline' }}>Teach your first one</Link> — or switch on “Open everything” in <Link to="/settings" style={{ textDecoration: 'underline' }}>Care &amp; comfort</Link>.</div></div>}
          {advice && <div className="callout" style={{ margin: '12px 0' }}><Mascot mood="think" size={44} /><div><b>Try “{meta.methods.find((m) => m.id === advice.id)?.name}”.</b> {advice.why}</div></div>}
          <div className="method-grid">
            {meta.methods.map((m) => (
              <button key={m.id} className={`method ${method === m.id ? 'on' : ''} ${isLocked(m) ? 'locked-method' : ''}`} onClick={() => { if (m.redirect) nav(m.redirect); else if (isLocked(m)) { setLockMsg(true); say('That one opens after your first finished model. 🌱', 'sleepy', 5000) } else { setLockMsg(false); setMethod(m.id) } }}>
                <div className="row spread"><span style={{ fontSize: 26 }}>{m.icon}</span><Chip tone={m.level === 'Start here' ? 'ok' : m.level === 'Unique' ? 'accent' : 'neutral'}>{m.redirect ? 'opens its own page' : isLocked(m) ? '🔒 after 1st model' : m.level}</Chip></div>
                <h4>{m.name}</h4><p>{m.tagline}</p>
                {method === m.id && <div className="small method-more"><div><b>Like:</b> {m.analogy}</div><div><b>Needs:</b> {m.data}</div><div><b>Memory:</b> {m.gpu} · <b>Time:</b> {m.time}</div><div><b>Great when:</b> {m.use_when}</div></div>}
              </button>
            ))}
          </div>
        </Card>

        {card && !card.redirect && (
          <>
            <Card title="2 · Which examples?">
              <div className="grid g2">
                <Field label="One of yours"><select value={dataset} onChange={(e) => setDataset(e.target.value)}>
                  <option value="">Choose…</option>
                  {(datasets || []).filter((d) => d.state === 'done' && d.train_path).map((d) => <option key={d.name} value={d.train_path}>{d.name} · {d.format} · {d.train?.toLocaleString()} examples{d.grade ? ` · quality ${d.grade}` : ''}</option>)}
                  <option value="math_gym">🧮 Endless maths problems (no data needed)</option>
                </select></Field>
                <Field label="…or a Hugging Face dataset / file path" hint="e.g. yahma/alpaca-cleaned"><input type="text" value={picked || dataset === 'math_gym' ? '' : dataset} onChange={(e) => setDataset(e.target.value)} placeholder="owner/name" /></Field>
              </div>
              {wrongFormat && <div className="callout warn" style={{ marginTop: 12 }}><Mascot mood="oops" size={40} /><div>“{card.name}” works best with <b>{FORMAT_WORD[NEEDS[run][0]]}</b> examples, but this dataset is <b>{FORMAT_WORD[picked.format] || picked.format}</b>. You can still try, or <Link to="/gather" style={{ textDecoration: 'underline' }}>gather some fitting data</Link>.</div></div>}
              {!datasets?.length && <div className="small muted" style={{ marginTop: 10 }}>No datasets yet — <Link to="/gather" style={{ textDecoration: 'underline' }}>gather some</Link> first, or type a Hugging Face dataset above.</div>}
            </Card>

            <Card title="3 · The knobs" action={<div className="row"><div className="seg">{EFFORT.map(([k, l]) => <button key={k} className={effort === k ? 'on' : ''} onClick={() => setEffort(k)}>{l}</button>)}</div><Button icon={Wand2} onClick={recommend} disabled={!dataset && run !== 'grpo'}>Suggest cozy settings</Button></div>}>
              {reasons && <div className="callout ok" style={{ marginBottom: 14, flexDirection: 'column' }}><b>About {minutes} minutes. Here's why these settings:</b><ul style={{ paddingLeft: 18, margin: 0 }}>{reasons.map((r, i) => <li key={i}><Md text={r} /></li>)}</ul></div>}
              <div className="grid g3">
                {run !== 'pretrain' ? <Field label="Base model"><select value={s.model} onChange={set('model')}>{MODELS.some(([m]) => m === s.model) ? null : <option>{s.model}</option>}{MODELS.map(([m, l]) => <option key={m} value={m}>{l}</option>)}</select></Field>
                  : <Field label="Size of the new model"><select value={s.size} onChange={set('size')}><option value="tiny">Tiny (~20M)</option><option value="small">Small (~60M)</option><option value="base">Base (~150M)</option></select></Field>}
                {run === 'sft' && <Field label="How much to change" hint="QLoRA uses the least memory."><select value={s.mode} onChange={set('mode')}><option value="qlora">QLoRA — smallest, 4-bit</option><option value="lora">LoRA — a bit bigger</option><option value="full">Everything (small models only)</option></select></Field>}
                {run === 'grpo' && <Field label="How are answers scored?"><select value={s.reward} onChange={set('reward')}><option value="concise">Short and to the point</option><option value="format">Thinks, then answers</option><option value="math">Right maths answer</option></select></Field>}
                <Field label={`Steps: ${s.steps.toLocaleString()}`} hint="One step = a small batch of examples."><input type="range" min="10" max="5000" step="10" value={s.steps} onChange={set('steps')} /></Field>
              </div>
              <button className="btn btn-ghost btn-sm" style={{ marginTop: 12 }} onClick={() => setAdv(!adv)}><ChevronDown size={14} style={{ transform: adv ? 'rotate(180deg)' : '' }} /> Even more knobs</button>
              {adv && (
                <div className="grid g3" style={{ marginTop: 10 }}>
                  <Field label="Learning rate" hint="How big each learning step is."><input type="number" step="any" value={s.lr} onChange={set('lr')} /></Field>
                  <Field label={`Batch size: ${s.batch}`}><input type="range" min="1" max="32" value={s.batch} onChange={set('batch')} /></Field>
                  <Field label={`Gather up: ${s.grad_accum}×`} hint="Effective batch = batch × this."><input type="range" min="1" max="32" value={s.grad_accum} onChange={set('grad_accum')} /></Field>
                  <Field label={`Longest example: ${s.seq_len} tokens`}><input type="range" min="128" max="4096" step="128" value={s.seq_len} onChange={set('seq_len')} /></Field>
                  <Field label={`LoRA size: ${s.lora_r}`} hint="Bigger = more capacity, more memory."><input type="range" min="4" max="128" step="4" value={s.lora_r} onChange={set('lora_r')} /></Field>
                  <Field label="Run name" hint="Empty = a surprise name."><input type="text" value={name} onChange={(e) => setName(e.target.value)} /></Field>
                </div>
              )}
            </Card>
            <div className="wiz-foot" style={{ justifyContent: 'flex-end' }}>
              <Button variant="primary" icon={Rocket} onClick={start} disabled={!ready || busy}>{busy ? 'Hopping on…' : "Start teaching! 🚂"}</Button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

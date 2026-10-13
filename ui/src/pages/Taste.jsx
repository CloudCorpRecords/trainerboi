import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Utensils, Heart } from 'lucide-react'
import { sse, usePoll, toast } from '../api'
import { Button, Card, Chip, PageHeader, Field } from '../ui'
import { Mascot, Confetti } from '../mascot'
import { useCompanion } from '../companion'
import { earn } from '../rewards'

const IDEAS = ['Introduce yourself in two sentences.', 'Give me three tips for better sleep.', 'Explain what an API is to a 10-year-old.', 'Write a haiku about autumn.', 'What is 17 × 6? Show your thinking.']

function Plate({ title, text, status, secs, mood, win, onVote, canVote }) {
  return (
    <div className={`plate ${win ? 'win' : ''}`}>
      <div className="row spread"><b style={{ fontFamily: 'var(--font-head)', fontSize: 17 }}>{title}</b>{secs != null && <Chip>{secs}s</Chip>}</div>
      <div className="plate-body">
        {status === 'loading' ? <div className="row muted"><Mascot mood="think" size={56} /> Warming up…</div> : text || <span className="faint">…waiting for the taste test…</span>}
        {status === 'streaming' && <span className="typing" />}
      </div>
      {canVote && <Button variant={win ? 'primary' : 'default'} icon={Heart} onClick={onVote}>{win ? 'You liked this one!' : 'I like this one'}</Button>}
    </div>
  )
}

export default function Taste() {
  const { data: opts } = usePoll('/api/compare/options', 20000)
  const { say } = useCompanion()
  const [a, setA] = useState(''); const [b, setB] = useState('')
  const [prompt, setPrompt] = useState('')
  const [out, setOut] = useState({ a: { text: '', status: '' }, b: { text: '', status: '' } })
  const [busy, setBusy] = useState(false)
  const [finished, setFinished] = useState(false)
  const [vote, setVote] = useState(null)
  const [party, setParty] = useState(false)
  const ctl = useRef(null)

  useEffect(() => {
    if (!opts?.length || a) return
    const mine = opts.find((o) => o.group === 'Your models')
    if (mine) { setB(mine.id); const base = opts.find((o) => o.id === 'base:' + mine.id); setA(base ? base.id : opts[0].id) }
  }, [opts])

  const go = async () => {
    if (!prompt.trim() || !a || !b) return
    setBusy(true); setFinished(false); setVote(null)
    setOut({ a: { text: '', status: 'loading' }, b: { text: '', status: '' } })
    ctl.current = new AbortController()
    try {
      await sse('/api/compare', { a, b, prompt }, (e) => {
        if (e.finished) { setFinished(true); return }
        if (!e.side) return
        setOut((o) => {
          const n = { ...o, [e.side]: { ...o[e.side] } }
          if (e.status === 'loading') n[e.side].status = 'loading'
          if (e.t) { n[e.side].text += e.t; n[e.side].status = 'streaming' }
          if (e.done) { n[e.side].status = 'done'; n[e.side].secs = e.seconds }
          if (e.error) { n[e.side].status = 'done'; n[e.side].text = `(${e.error})` }
          return n
        })
      }, ctl.current.signal)
    } catch (e) { if (e.name !== 'AbortError') toast(e.message, 'error') }
    setBusy(false)
  }
  const choose = (side) => { setVote(side); setParty(true); earn('taste'); say(side === 'tie' ? 'A tie! Both are tasty. 🍰' : 'Noted! Your taste is excellent. 📝✨', 'love', 5000); setTimeout(() => setParty(false), 100) }
  const label = (id) => opts?.find((o) => o.id === id)?.label || '…'

  return (
    <div className="page" style={{ maxWidth: 1080 }}>
      <Confetti play={party} />
      <PageHeader title="Taste test 🍰" sub="Ask two models the same question and see whose answer you like best. Try one against its own 'before training' self to see how much it grew." />
      {!opts ? <div className="muted">Loading…</div> : opts.length === 0 ? (
        <Card><div className="empty"><Mascot mood="sleepy" size={110} /><h3>Nothing to taste yet</h3><p>Teach a model first, then come back for a taste test.</p><Link to="/train" className="btn btn-primary">Teach a model ✨</Link></div></Card>
      ) : (
        <div className="stack">
          <Card>
            <div className="grid g2">
              <Field label="Model A"><select value={a} onChange={(e) => setA(e.target.value)}>{['Before training', 'Your models'].map((g) => <optgroup key={g} label={g}>{opts.filter((o) => o.group === g).map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}</optgroup>)}</select></Field>
              <Field label="Model B"><select value={b} onChange={(e) => setB(e.target.value)}>{['Your models', 'Before training'].map((g) => <optgroup key={g} label={g}>{opts.filter((o) => o.group === g).map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}</optgroup>)}</select></Field>
            </div>
            <div style={{ marginTop: 14 }}>
              <Field label="Your question"><input type="text" value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && !busy && go()} placeholder="Ask anything…" /></Field>
              <div className="row wrap" style={{ marginTop: 10 }}>{IDEAS.map((i) => <button key={i} className="src-chip" style={{ cursor: 'pointer' }} onClick={() => setPrompt(i)}><span>{i}</span></button>)}</div>
            </div>
            <div className="row" style={{ marginTop: 14 }}>
              <Button variant="primary" icon={Utensils} onClick={go} disabled={busy || !prompt.trim()}>{busy ? 'Tasting…' : 'Taste test! 🍴'}</Button>
              {busy && <Button variant="ghost" onClick={() => ctl.current?.abort()}>Stop</Button>}
              <span className="small faint">The models take turns, so only one is on the GPU at a time.</span>
            </div>
          </Card>
          <div className="grid g2">
            <Plate title={`A · ${label(a)}`} {...out.a} secs={out.a.secs} win={vote === 'a'} canVote={finished} onVote={() => choose('a')} />
            <Plate title={`B · ${label(b)}`} {...out.b} secs={out.b.secs} win={vote === 'b'} canVote={finished} onVote={() => choose('b')} />
          </div>
          {finished && !vote && <div className="row" style={{ justifyContent: 'center' }}><Button variant="ghost" onClick={() => choose('tie')}>They're both lovely (tie)</Button></div>}
        </div>
      )}
    </div>
  )
}

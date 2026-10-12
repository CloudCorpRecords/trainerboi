import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { MessageCircle, Share2, FileText, Settings2, Send, Square, Package, Globe, Link2, Download, FolderOpen, ArrowLeft, Trash2, ExternalLink } from 'lucide-react'
import { j, post, sse, usePoll, useJob, toast, fmt } from '../api'
import { Button, Card, Chip, Field, PageHeader, Tabs, Progress } from '../ui'
import { KIND } from './Models'

function Chat({ rec }) {
  const [msgs, setMsgs] = useState([])
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(false)
  const ctl = useRef(null)
  const end = useRef(null)
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  const send = async () => {
    const t = text.trim()
    if (!t || busy) return
    const history = [...msgs, { role: 'user', content: t }]
    setMsgs([...history, { role: 'assistant', content: '' }])
    setText(''); setBusy(true); setLoading(false)
    ctl.current = new AbortController()
    try {
      await sse('/api/chat', { model_id: rec.id, messages: history }, (e) => {
        if (e.status === 'loading') setLoading(true)
        if (e.t) { setLoading(false); setMsgs((m) => { const c = [...m]; c[c.length - 1] = { role: 'assistant', content: c[c.length - 1].content + e.t }; return c }) }
        if (e.error) toast(e.error, 'error')
      }, ctl.current.signal)
    } catch (e) { if (e.name !== 'AbortError') toast(e.message, 'error') }
    setBusy(false); setLoading(false)
  }
  const raw = rec.kind === 'pretrain'
  return (
    <Card pad={false}>
      <div className="card-body">
        <div className="chat">
          <div className="chat-scroll">
            {!msgs.length && <div className="muted" style={{ margin: 'auto', textAlign: 'center' }}>
              {raw ? 'This model was built from scratch to continue text. Type a few words (like “Once upon a time”) and it will continue.' : 'Say something to your model.'}
            </div>}
            {msgs.map((m, i) => <div key={i} className={`msg ${m.role === 'user' ? 'user' : 'bot'} ${busy && i === msgs.length - 1 ? 'typing' : ''}`}>{m.content || (loading ? 'Loading the model (first message only)…' : '')}</div>)}
            <div ref={end} />
          </div>
          <div className="chat-input">
            <input type="text" value={text} placeholder={raw ? 'Start of a sentence…' : 'Ask your model anything…'} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && send()} />
            {busy ? <Button icon={Square} onClick={() => ctl.current?.abort()}>Stop</Button> : <Button variant="primary" icon={Send} onClick={send}>Send</Button>}
          </div>
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <Button variant="ghost" size="sm" onClick={() => setMsgs([])}>Clear chat</Button>
          <Button variant="ghost" size="sm" onClick={async () => { await post('/api/chat/unload'); toast('GPU memory freed.', 'success') }}>Free GPU memory</Button>
        </div>
      </div>
    </Card>
  )
}

function ShareRow({ icon: I, title, text, children }) {
  return (
    <div className="share-opt" style={{ gridTemplateColumns: '40px 1fr' }}>
      <div className="ico"><I size={19} /></div>
      <div className="stack" style={{ gap: 10 }}><div><b>{title}</b><div className="muted small">{text}</div></div>{children}</div>
    </div>
  )
}

function Share({ rec }) {
  const [kind, setKind] = useState(rec.adapter ? 'adapter' : 'full')
  const [zipJob, followZip] = useJob()
  const [pubJob, followPub] = useJob()
  const [repo, setRepo] = useState('')
  const [priv, setPriv] = useState(true)
  const [token, setToken] = useState('')
  const { data: share, reload } = usePoll('/api/share', 2500)
  const mine = share && share.model_id === rec.id
  const id = encodeURIComponent(rec.id)

  const makeZip = async () => { try { const r = await post(`/api/models/${id}/zip`, { kind }); followZip(r.job) } catch (e) { toast(e.message, 'error') } }
  const publish = async () => { try { const r = await post(`/api/models/${id}/publish`, { kind, repo, private: priv, token }); followPub(r.job) } catch (e) { toast(e.message, 'error') } }
  const startShare = async () => { try { await post(`/api/models/${id}/share/start`); reload() } catch (e) { toast(e.message, 'error') } }
  const stopShare = async () => { await post('/api/share/stop'); reload() }

  return (
    <div className="stack">
      {rec.adapter && (
        <div className="row"><span className="muted small">What to send:</span>
          <div className="seg"><button className={kind === 'full' ? 'on' : ''} onClick={() => setKind('full')}>Full model <span className="faint">(works anywhere)</span></button>
            <button className={kind === 'adapter' ? 'on' : ''} onClick={() => setKind('adapter')}>Small adapter <span className="faint">(needs the base model)</span></button></div></div>
      )}
      <ShareRow icon={Package} title="Send as a file" text="One .zip with the model, an auto-written model card and a chat script. Works offline — share it any way you like.">
        <div className="row wrap">
          <Button variant="primary" icon={Package} onClick={makeZip} disabled={zipJob?.state === 'running'}>{zipJob?.state === 'running' ? 'Preparing…' : 'Create zip'}</Button>
          {zipJob?.state === 'running' && <span className="muted small">{zipJob.log?.slice(-1)[0] || 'Working…'}</span>}
          {zipJob?.state === 'done' && <>
            <a href={`/api/download?path=${encodeURIComponent(zipJob.result.path)}`} download><Button icon={Download}>Save {zipJob.result.name} ({zipJob.result.size_mb} MB)</Button></a>
            <Button variant="ghost" icon={FolderOpen} onClick={() => post('/api/reveal', { path: zipJob.result.path })}>Show in folder</Button></>}
          {zipJob?.state === 'error' && <Chip tone="bad">{zipJob.error}</Chip>}
        </div>
      </ShareRow>

      <ShareRow icon={Globe} title="Publish on Hugging Face" text="Free hosting with a link you can send. Needs a Hugging Face account and an access token with write permission (or sign in once with the Hugging Face CLI and leave the token empty). Check the base model's licence on the Card tab first.">
        <div className="grid g2">
          <Field label="Name on the Hub"><input type="text" value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="my-model  or  username/my-model" /></Field>
          <Field label="Access token (optional)"><input type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder="hf_…" /></Field>
        </div>
        <label className="row small"><input type="checkbox" checked={priv} onChange={(e) => setPriv(e.target.checked)} /> Private — only you can see it until you change that on the Hub</label>
        <div className="row wrap">
          <Button icon={Globe} onClick={publish} disabled={pubJob?.state === 'running'}>{pubJob?.state === 'running' ? 'Uploading…' : 'Publish'}</Button>
          {pubJob?.state === 'running' && <span className="muted small">{pubJob.log?.slice(-1)[0]}</span>}
          {pubJob?.state === 'done' && <a href={pubJob.result.url} target="_blank" rel="noreferrer" className="row" style={{ color: 'var(--accent-2)' }}>{pubJob.result.url} <ExternalLink size={14} /></a>}
          {pubJob?.state === 'error' && <Chip tone="bad">{pubJob.error}</Chip>}
        </div>
      </ShareRow>

      <ShareRow icon={Link2} title="Temporary public chat link" text="Let friends chat with the model in their browser — no install. It runs on your computer through Gradio's relay, so it only works while TrainerBoi is open, expires in about a week, and anyone with the link can use it.">
        <div className="row wrap">
          {!mine || share.state === 'off' || share.state === 'stopped'
            ? <Button icon={Link2} onClick={startShare}>Create link</Button>
            : <Button variant="danger" icon={Square} onClick={stopShare}>Stop sharing</Button>}
          {mine && share.state === 'starting' && <span className="muted small">Loading the model and creating the link (about a minute)…</span>}
          {mine && share.state === 'live' && <a href={share.url} target="_blank" rel="noreferrer" className="row" style={{ color: 'var(--ok)' }}>{share.url} <ExternalLink size={14} /></a>}
          {mine && share.state === 'stopped' && <Chip tone="warn">The shared page stopped</Chip>}
        </div>
      </ShareRow>
    </div>
  )
}

function Card_({ rec }) {
  const { data } = usePoll(`/api/models/${encodeURIComponent(rec.id)}/card`, 600000)
  return <Card title="Model card (README.md shipped with every share)"><pre className="log" style={{ maxHeight: 460 }}>{data?.markdown || 'Loading…'}</pre></Card>
}

function Manage({ rec, onSaved }) {
  const nav = useNavigate()
  const [title, setTitle] = useState(rec.title)
  const [notes, setNotes] = useState(rec.notes || '')
  const [sure, setSure] = useState(false)
  const id = encodeURIComponent(rec.id)
  return (
    <div className="stack" style={{ maxWidth: 640 }}>
      <Card title="Name & notes">
        <div className="stack">
          <Field label="Name"><input type="text" value={title} onChange={(e) => setTitle(e.target.value)} /></Field>
          <Field label="Notes (appear on the model card)"><textarea value={notes} onChange={(e) => setNotes(e.target.value)} /></Field>
          <div><Button variant="primary" onClick={async () => { await post(`/api/models/${id}/meta`, { title, notes }); toast('Saved.', 'success'); onSaved() }}>Save</Button></div>
        </div>
      </Card>
      <Card title="Location"><div className="row spread"><code style={{ wordBreak: 'break-all' }}>{rec.path}</code><Button size="sm" icon={FolderOpen} onClick={() => post('/api/reveal', { path: rec.path })}>Show</Button></div></Card>
      <Card title="Delete">
        <p className="muted small" style={{ marginBottom: 10 }}>Deletes this run's folder: the model, checkpoints and logs. This can't be undone.</p>
        <label className="row small" style={{ marginBottom: 10 }}><input type="checkbox" checked={sure} onChange={(e) => setSure(e.target.checked)} /> Yes, delete this model permanently</label>
        <Button variant="danger" icon={Trash2} disabled={!sure} onClick={async () => { await post(`/api/models/${id}/delete`); toast('Deleted.'); nav('/models') }}>Delete</Button>
      </Card>
    </div>
  )
}

export default function ModelDetail() {
  const { id: raw } = useParams()
  const id = decodeURIComponent(raw)
  const [params, setParams] = useSearchParams()
  const { data: models, reload } = usePoll('/api/models', 8000)
  const tab = params.get('tab') || 'chat'
  const rec = models?.find((m) => m.id === id)
  if (!models) return <div className="page muted">Loading…</div>
  if (!rec) return <div className="page"><PageHeader title="Model not found" sub="It may have been deleted." actions={<Link to="/models"><Button icon={ArrowLeft}>All models</Button></Link>} /></div>
  return (
    <div className="page">
      <PageHeader title={rec.title}
        sub={<span className="row wrap"><Chip tone="accent">{KIND[rec.kind] || rec.kind}{rec.adapter ? ' · LoRA adapter' : ''}</Chip>{rec.base && <Chip>{rec.base.split('/').pop()}</Chip>}{rec.metric && <Chip tone="ok">{rec.metric}</Chip>}<span className="faint">{fmt.mb(rec.size_mb)}</span></span>}
        actions={<Link to="/models"><Button variant="ghost" icon={ArrowLeft}>All models</Button></Link>} />
      <Tabs value={tab} onChange={(t) => setParams({ tab: t })} tabs={[
        { id: 'chat', label: 'Chat', icon: MessageCircle }, { id: 'share', label: 'Share', icon: Share2 },
        { id: 'card', label: 'Model card', icon: FileText }, { id: 'manage', label: 'Manage', icon: Settings2 }]} />
      {tab === 'chat' && (rec.chat_ok === false ? <div className="callout warn">This is an embedding model — it finds similar text instead of chatting. Use it from Advanced tools → RAG.</div> : <Chat rec={rec} />)}
      {tab === 'share' && <Share rec={rec} />}
      {tab === 'card' && <Card_ rec={rec} />}
      {tab === 'manage' && <Manage rec={rec} onSaved={reload} />}
    </div>
  )
}

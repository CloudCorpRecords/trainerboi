import { useRef, useState } from 'react'
import { UploadCloud, X, Link2 } from 'lucide-react'
import { post, toast } from './api'
import { Button, Chip, Field } from './ui'

export const SRC_LABEL = { urls: 'Website', hf: 'Hugging Face', wikipedia: 'Wikipedia', arxiv: 'Papers', github: 'GitHub', youtube: 'YouTube', folders: 'On this PC' }
export const EMPTY_SRC = { urls: [], hf: [], wikipedia: [], arxiv: [], github: [], youtube: [], folders: [] }

/** Drop files, or paste links / repos / datasets / a topic. Used by the data kitchen. */
export function Ingredients({ files, setFiles, src, setSrc, onHint }) {
  const [paste, setPaste] = useState('')
  const [over, setOver] = useState(false)
  const ref = useRef(null)

  const upload = async (list) => {
    if (!list?.length) return
    const fd = new FormData()
    for (const f of list) fd.append('files', f)
    try {
      const out = await fetch('/api/upload', { method: 'POST', body: fd }).then((r) => r.json())
      setFiles([...files, ...out])
      onHint?.(await post('/api/intake', { text: '', files: out.map((o) => o.path) }))
    } catch (e) { toast(e.message, 'error') }
  }
  const add = async () => {
    if (!paste.trim()) return
    try {
      const r = await post('/api/intake', { text: paste, files: [] })
      setSrc((s) => { const n = { ...s }; for (const k in EMPTY_SRC) n[k] = [...new Set([...s[k], ...(r.fields[k] || [])])]; return n })
      onHint?.(r); setPaste('')
    } catch (e) { toast(e.message, 'error') }
  }
  const chips = Object.entries(src).flatMap(([k, a]) => a.map((v) => [k, v]))
  return (
    <div className="stack" style={{ gap: 14 }}>
      <div className={`drop ${over ? 'over' : ''}`} onClick={() => ref.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); upload([...e.dataTransfer.files]) }}>
        <UploadCloud size={26} style={{ marginBottom: 6 }} />
        <div style={{ fontSize: 17 }}><b>Drop your files here</b> 🌱 or click to choose</div>
        <div className="small faint" style={{ marginTop: 4 }}>PDF, Word, text, web pages, CSV, JSON, code, zip, audio/video…</div>
        <input ref={ref} type="file" multiple hidden onChange={(e) => { upload([...e.target.files]); e.target.value = '' }} />
      </div>
      <Field label="Or paste anything" hint="Links, YouTube videos, GitHub repos, Hugging Face datasets, folder paths — or just a topic you're curious about. One per line.">
        <textarea value={paste} onChange={(e) => setPaste(e.target.value)} placeholder={'beekeeping basics\nhttps://docs.python.org/3/tutorial/\nhttps://github.com/owner/repo\nyahma/alpaca-cleaned'} />
      </Field>
      <div><Button icon={Link2} onClick={add} disabled={!paste.trim()}>Add it</Button></div>
      {(files.length > 0 || chips.length > 0) && (
        <div className="row wrap">
          {files.map((f) => <span key={f.path} className="src-chip"><Chip>File</Chip><span>{f.name}</span><button onClick={() => setFiles(files.filter((x) => x.path !== f.path))}><X size={13} /></button></span>)}
          {chips.map(([k, v]) => <span key={k + v} className="src-chip"><Chip tone="accent">{SRC_LABEL[k]}</Chip><span>{v}</span><button onClick={() => setSrc((s) => ({ ...s, [k]: s[k].filter((x) => x !== v) }))}><X size={13} /></button></span>)}
        </div>
      )}
    </div>
  )
}

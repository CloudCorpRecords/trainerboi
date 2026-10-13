import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Search, Trophy, Sparkles } from 'lucide-react'
import { usePoll, fmt } from '../api'
import { Chip, Empty, PageHeader, Button } from '../ui'
import { Mascot } from '../mascot'

export const KIND = { sft: 'Fine-tuned', dpo: 'Preference-tuned', orpo: 'ORPO', kto: 'KTO', grpo: 'Reinforcement', pretrain: 'From scratch', math: 'Math', exported: 'Exported', embedding: 'Embedding' }

export default function Models() {
  const { data: models } = usePoll('/api/models', 6000)
  const [q, setQ] = useState('')
  const list = useMemo(() => (models || []).filter((m) => (m.title + m.kind + (m.base || '')).toLowerCase().includes(q.toLowerCase())), [models, q])
  return (
    <div className="page">
      <PageHeader title="My little models" sub="Everything you've taught, all in one cozy place. Open one to chat with it or share it."
        actions={<Link to="/train"><Button variant="primary" icon={Sparkles}>Teach another</Button></Link>} />
      {models && models.length > 4 && (
        <div style={{ position: 'relative', maxWidth: 360, marginBottom: 16 }}>
          <Search size={15} style={{ position: 'absolute', left: 12, top: 12, color: 'var(--faint)' }} />
          <input type="text" style={{ paddingLeft: 34 }} placeholder="Search models" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
      )}
      {!models ? <div className="row muted"><Mascot mood="working" size={48} /> Gathering your models…</div> : !list.length ? (
        <Empty icon={() => <Mascot mood="sleepy" size={110} />} title="It's quiet in here" text="No models yet — let's teach your first one together. It takes about ten minutes." action={<Link to="/train"><Button variant="primary">Let's begin ✨</Button></Link>} />
      ) : (
        <div className="grid g3">
          {list.map((m) => (
            <Link key={m.id} to={`/models/${encodeURIComponent(m.id)}`} className="card model-card">
              <div className="row spread"><Chip tone="accent">{KIND[m.kind] || m.kind}{m.adapter ? ' · adapter' : ''}</Chip><span className="small faint">{fmt.ago(m.created)}</span></div>
              <h4>{m.title}</h4>
              <div className="small muted" style={{ minHeight: 36 }}>{m.goal}{m.base ? ` · ${m.base.split('/').pop()}` : ''}</div>
              <div className="row spread small">
                <span className="muted">{m.metric || '—'}</span>
                <span className="row" style={{ gap: 8 }}>{m.status && m.status !== 'done' && <Chip tone="warn">{m.status}</Chip>}<span className="faint">{fmt.mb(m.size_mb)}</span></span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}

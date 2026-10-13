import { Link, useNavigate } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'
import { useHardware, usePoll, fmt } from '../api'
import { Card, Chip, Stat, Empty } from '../ui'
import { Sparkline } from '../charts'
import { Mascot, Sky, Track, tempMood } from '../mascot'
import { jobLink } from '../App'

const KIND = { sft: 'Fine-tuned', dpo: 'Preference', orpo: 'ORPO', kto: 'KTO', grpo: 'Reinforcement', pretrain: 'From scratch', math: 'Maths', exported: 'Exported', embedding: 'Embedding' }

export default function Home() {
  const { hw, history } = useHardware()
  const nav = useNavigate()
  const { data: models } = usePoll('/api/models', 8000)
  const { data: runs } = usePoll('/api/runs', 5000)
  const g = hw?.gpu
  const job = hw?.jobs?.[0]
  const hour = new Date().getHours()
  const greet = hour < 5 ? 'Still up? Cozy vibes' : hour < 12 ? 'Good morning' : hour < 18 ? 'Hello there' : 'Good evening'
  const m = tempMood(g?.temp)
  const mood = job ? (job.kind === 'cooling' ? 'toasty' : 'working') : hour >= 23 || hour < 5 ? 'sleepy' : 'happy'

  return (
    <div className="page">
      <div className="hero">
        <Sky />
        <div>
          <h1>{greet} 👋</h1>
          <p>{job
            ? <>Boi is busy teaching <b>{job.name}</b>. You can relax — it'll tell you when it's done.</>
            : <>Your {g?.name || 'GPU'} is resting and ready. What shall we teach a little model today?</>}</p>
          <div className="row" style={{ marginTop: 16 }}>
            <Link to="/train" className="btn btn-primary">Teach a model ✨</Link>
            <Link to="/math" className="btn">Math playground 🧮</Link>
          </div>
        </div>
        <Mascot mood={mood} size={168} />
      </div>

      {job && (
        <div className="live-card" onClick={() => nav(jobLink(job))}>
          <div className="stack" style={{ gap: 6 }}>
            <div className="row"><Chip tone={job.kind === 'cooling' ? 'warn' : 'info'}>{job.kind === 'cooling' ? 'Taking a breather' : 'Chugging along'}</Chip><b>{job.name}</b></div>
            <div className="muted">{job.phase}{job.max_steps ? ` · step ${fmt.n(job.step)} of ${fmt.n(job.max_steps)}` : ''}{job.loss != null ? ` · loss ${job.loss.toFixed(3)}` : ''}{job.eta_s ? ` · about ${fmt.dur(job.eta_s)} to go` : ''}</div>
            <Track value={job.progress} />
          </div>
          <span className="btn btn-sm">Peek in <ArrowRight size={14} /></span>
        </div>
      )}

      <div className="grid g4">
        <Stat label="How hard it's working" value={g?.util != null ? Math.round(g.util) : '–'} unit="%"><Sparkline data={history.util} min={0} max={100} color="var(--accent)" /></Stat>
        <Stat label="Memory in use" value={g?.vram_used_gb != null ? g.vram_used_gb.toFixed(1) : '–'} unit={`/ ${g?.vram_total_gb ? Math.round(g.vram_total_gb) : '–'} GB`}>
          <Sparkline data={history.vram} min={0} max={g?.vram_total_gb || 16} color="var(--sky)" />
        </Stat>
        <Stat label="Temperature" value={g?.temp != null ? Math.round(g.temp) : '–'} unit="°C" tone={m.tone} sub={g?.temp != null ? `Feeling ${m.word}${g.temp >= 80 ? ' — it will pause to cool down' : ''}` : ''}>
          <Sparkline data={history.temp} color="var(--peach)" />
        </Stat>
        <Stat label="Energy" value={g?.power != null ? Math.round(g.power) : '–'} unit={g?.power_limit ? `/ ${Math.round(g.power_limit)} W` : 'W'}>
          <Sparkline data={history.power} min={0} max={g?.power_limit || 150} color="var(--mint)" />
        </Stat>
      </div>

      <div className="action-grid">
        <Link to="/train" className="action primary"><div className="ico">✨</div><h4>Teach a model</h4><p>Tell us what you'd like. We'll pick everything else, gently.</p></Link>
        <Link to="/math" className="action"><div className="ico">🧮</div><h4>Math playground</h4><p>Practice problems that check themselves — no data needed.</p></Link>
        <Link to="/data" className="action"><div className="ico">🌱</div><h4>My data</h4><p>See how healthy your examples are and tidy them up.</p></Link>
        <Link to="/models" className="action"><div className="ico">💬</div><h4>My little models</h4><p>Say hello to what you've made, then share it with friends.</p></Link>
      </div>

      <div className="grid g2" style={{ gridTemplateColumns: '1.4fr 1fr' }}>
        <Card title="Your little models" action={<Link to="/models" className="small muted">See all</Link>}>
          {!models?.length ? (
            <Empty title="No models yet" text="Let's make your first one together — it only takes about ten minutes!"
              icon={() => <Mascot mood="happy" size={90} />} action={<Link to="/train" className="btn btn-primary">Let's begin ✨</Link>} />
          ) : (
            <div className="grid g2">
              {models.slice(0, 4).map((x) => (
                <Link key={x.id} to={`/models/${encodeURIComponent(x.id)}`} className="card model-card">
                  <div className="row spread"><Chip tone="accent">{KIND[x.kind] || x.kind}</Chip><span className="small faint">{fmt.ago(x.created)}</span></div>
                  <h4>{x.title}</h4>
                  <div className="small muted">{x.metric || x.goal} · {fmt.mb(x.size_mb)}</div>
                </Link>
              ))}
            </div>
          )}
        </Card>
        <Card title="What's been happening">
          {!runs?.length ? <div className="muted">It's quiet and peaceful here. 🌿</div> : runs.slice(0, 6).map((r) => (
            <Link key={r.name} className="list-row" to={r.kind === 'math' ? '/math' : `/run/${encodeURIComponent(r.name)}`}>
              <span style={{ fontSize: 18 }}>{r.state === 'done' ? '🌸' : r.state === 'running' ? '🚂' : r.state === 'crashed' || r.state === 'error' ? '🫧' : '💤'}</span>
              <div style={{ flex: 1, minWidth: 0 }}><div style={{ fontWeight: 800, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.name}</div><div className="small muted">{r.title} · {fmt.ago(r.updated)}</div></div>
              <Chip tone={r.state === 'done' ? 'ok' : r.state === 'running' ? 'info' : r.state === 'crashed' || r.state === 'error' ? 'warn' : 'neutral'}>{r.state === 'done' ? 'all done' : r.state === 'running' ? 'busy' : r.state === 'crashed' || r.state === 'error' ? 'hiccup' : r.state}</Chip>
            </Link>
          ))}
        </Card>
      </div>
    </div>
  )
}

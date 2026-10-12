import { Link, useNavigate } from 'react-router-dom'
import { Sparkles, Calculator, Database, Trophy, Thermometer, Zap, MemoryStick, Activity, ArrowRight } from 'lucide-react'
import { useHardware, usePoll, fmt } from '../api'
import { Card, Chip, Progress, Stat, Empty } from '../ui'
import { Sparkline } from '../charts'
import { jobLink } from '../App'

const tempTone = (t) => (t == null ? undefined : t >= 85 ? 'bad' : t >= 75 ? 'warn' : 'ok')

export default function Home() {
  const { hw, history } = useHardware()
  const nav = useNavigate()
  const { data: models } = usePoll('/api/models', 8000)
  const { data: runs } = usePoll('/api/runs', 5000)
  const g = hw?.gpu
  const job = hw?.jobs?.[0]
  const hour = new Date().getHours()
  const greet = hour < 5 ? 'Working late' : hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'

  return (
    <div className="page">
      <div className="hero">
        <h1>{greet}.</h1>
        <p>{job ? <>Your {g?.name || 'GPU'} is busy with <b>{job.name}</b>.</> : <>Your {g?.name || 'GPU'} is idle and ready. Train a model on your own computer — no code, no cloud.</>}</p>
      </div>

      {job && (
        <div className="live-card" onClick={() => nav(jobLink(job))}>
          <div className="stack" style={{ gap: 8 }}>
            <div className="row"><Chip tone={job.kind === 'cooling' ? 'warn' : 'info'}>{job.kind === 'cooling' ? 'Cooling down' : 'Running now'}</Chip><b>{job.name}</b></div>
            <div className="muted">{job.phase}{job.max_steps ? ` · step ${fmt.n(job.step)} of ${fmt.n(job.max_steps)}` : ''}{job.loss != null ? ` · loss ${job.loss.toFixed(3)}` : ''}{job.eta_s ? ` · about ${fmt.dur(job.eta_s)} left` : ''}</div>
            <Progress value={job.progress} />
          </div>
          <span className="btn btn-sm">Open <ArrowRight size={14} /></span>
        </div>
      )}

      <div className="grid g4">
        <Stat label="GPU activity" value={g?.util != null ? Math.round(g.util) : '–'} unit="%"><Sparkline data={history.util} min={0} max={100} /></Stat>
        <Stat label="GPU memory" value={g?.vram_used_gb != null ? g.vram_used_gb.toFixed(1) : '–'} unit={`/ ${g?.vram_total_gb ? Math.round(g.vram_total_gb) : '–'} GB`}>
          <Sparkline data={history.vram} min={0} max={g?.vram_total_gb || 16} color="var(--info)" />
        </Stat>
        <Stat label="Temperature" value={g?.temp != null ? Math.round(g.temp) : '–'} unit="°C" tone={tempTone(g?.temp)} sub={g?.temp >= 80 ? 'Hot — training will pause to cool' : 'Training pauses at 82°C'}>
          <Sparkline data={history.temp} color="var(--warn)" />
        </Stat>
        <Stat label="Power" value={g?.power != null ? Math.round(g.power) : '–'} unit={g?.power_limit ? `/ ${Math.round(g.power_limit)} W` : 'W'}>
          <Sparkline data={history.power} min={0} max={g?.power_limit || 150} color="var(--ok)" />
        </Stat>
      </div>

      <div className="action-grid">
        <Link to="/train" className="action primary"><div className="ico"><Sparkles size={18} /></div><h4>New training</h4><p>Pick a goal. We choose the model, method and settings for you.</p></Link>
        <Link to="/math" className="action"><div className="ico"><Calculator size={18} /></div><h4>Math Lab</h4><p>Teach a model maths with no data — it learns from its own verified answers.</p></Link>
        <Link to="/data" className="action"><div className="ico"><Database size={18} /></div><h4>Data</h4><p>Check your datasets for problems and fix them in one click.</p></Link>
        <Link to="/models" className="action"><div className="ico"><Trophy size={18} /></div><h4>My models</h4><p>Chat with what you trained, then share it with anyone.</p></Link>
      </div>

      <div className="grid g2" style={{ gridTemplateColumns: '1.4fr 1fr' }}>
        <Card title="Recent models" action={<Link to="/models" className="small muted">See all</Link>}>
          {!models?.length ? <Empty icon={Trophy} title="No models yet" text="Your first one takes about 10 minutes." /> : (
            <div className="grid g2">
              {models.slice(0, 4).map((m) => (
                <Link key={m.id} to={`/models/${encodeURIComponent(m.id)}`} className="card model-card">
                  <div className="row spread"><Chip tone="accent">{m.kind}</Chip><span className="small faint">{fmt.ago(m.created)}</span></div>
                  <h4>{m.title}</h4>
                  <div className="small muted">{m.metric || m.goal} · {fmt.mb(m.size_mb)}</div>
                </Link>
              ))}
            </div>
          )}
        </Card>
        <Card title="Recent activity">
          {!runs?.length ? <div className="muted">Nothing yet.</div> : runs.slice(0, 6).map((r) => (
            <Link key={r.name} className="list-row" to={r.kind === 'math' ? '/math' : `/run/${encodeURIComponent(r.name)}`}>
              <Activity size={15} className="faint" />
              <div style={{ flex: 1, minWidth: 0 }}><div style={{ fontWeight: 550, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.name}</div><div className="small muted">{r.title} · {fmt.ago(r.updated)}</div></div>
              <Chip tone={r.state === 'done' ? 'ok' : r.state === 'running' ? 'info' : r.state === 'crashed' || r.state === 'error' ? 'bad' : 'neutral'}>{r.state}</Chip>
            </Link>
          ))}
        </Card>
      </div>
    </div>
  )
}

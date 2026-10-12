import { NavLink, Route, Routes, useNavigate } from 'react-router-dom'
import { Home as HomeIcon, Sparkles, Calculator, Database, Trophy, Wrench, Settings as Cog, Cpu } from 'lucide-react'
import { useHardware } from './api'
import { Chip, Toasts, Progress } from './ui'
import { Sparkline } from './charts'
import Home from './pages/Home'
import Train from './pages/Train'
import Run from './pages/Run'
import Models from './pages/Models'
import ModelDetail from './pages/ModelDetail'
import MathLab from './pages/MathLab'
import Data from './pages/Data'
import Advanced from './pages/Advanced'
import Settings from './pages/Settings'

const NAV = [
  { to: '/', label: 'Home', icon: HomeIcon, end: true },
  { section: 'Create' },
  { to: '/train', label: 'New training', icon: Sparkles },
  { to: '/math', label: 'Math Lab', icon: Calculator },
  { to: '/data', label: 'Data', icon: Database },
  { section: 'Use' },
  { to: '/models', label: 'My models', icon: Trophy },
  { section: 'More' },
  { to: '/advanced', label: 'Advanced tools', icon: Wrench },
  { to: '/settings', label: 'Settings & health', icon: Cog },
]

export const jobLink = (j) => (j.kind === 'math' ? '/math' : j.kind === 'data' ? '/data' : `/run/${encodeURIComponent(j.name)}`)

function JobPill() {
  const { hw } = useHardware()
  const nav = useNavigate()
  const job = hw?.jobs?.[0]
  if (!job) return <span className="job-pill"><Chip tone="ok">Idle</Chip></span>
  const label = job.kind === 'cooling' ? 'Cooling GPU' : job.kind === 'math' ? 'Math Lab' : job.kind === 'data' ? 'Preparing data' : 'Training'
  return (
    <span className="job-pill" onClick={() => nav(jobLink(job))}>
      <Chip tone={job.kind === 'cooling' ? 'warn' : 'info'}>{label} · {job.name}{hw.jobs.length > 1 ? ` +${hw.jobs.length - 1}` : ''}</Chip>
    </span>
  )
}

function SideGpu() {
  const { hw, history } = useHardware()
  const nav = useNavigate()
  const g = hw?.gpu
  const hot = g?.temp != null && g.temp >= 80
  return (
    <div className="side-gpu" onClick={() => nav('/settings')} title="Live GPU readings from the NVIDIA driver">
      <div className="row spread" style={{ marginBottom: 6 }}>
        <span className="row" style={{ gap: 6 }}><Cpu size={14} /> <b>{g?.name || 'GPU'}</b></span>
        <b style={{ color: hot ? 'var(--bad)' : undefined }}>{g?.temp != null ? `${Math.round(g.temp)}°C` : 'n/a'}</b>
      </div>
      <Progress value={g?.vram_used_gb != null && g?.vram_total_gb ? g.vram_used_gb / g.vram_total_gb : 0} />
      <div className="row spread small muted" style={{ marginTop: 6 }}>
        <span>VRAM {g?.vram_used_gb != null ? g.vram_used_gb.toFixed(1) : '–'} / {g?.vram_total_gb ? Math.round(g.vram_total_gb) : '–'} GB</span>
        <span>{g?.util != null ? `${Math.round(g.util)}%` : '–'}</span>
      </div>
      <Sparkline data={history.util} min={0} max={100} height={28} />
    </div>
  )
}

export default function App() {
  const { hw } = useHardware()
  const busy = !!hw?.jobs?.some((j) => j.kind !== 'data')
  return (
    <div className="app">
      <div className="titlebar">
        <div className="brand"><span className="logo-mark">T</span>TrainerBoi</div>
        <JobPill />
      </div>
      <div className="body">
        <aside className="sidebar">
          <nav className="nav">
            {NAV.map((n, i) => n.section
              ? <div className="nav-section" key={i}>{n.section}</div>
              : <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => (isActive ? 'active' : '')}>
                <n.icon size={17} />{n.label}{n.to === '/train' && busy && <span className="dot" />}
              </NavLink>)}
          </nav>
          <SideGpu />
        </aside>
        <main className="main">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/train" element={<Train />} />
            <Route path="/run/:name" element={<Run />} />
            <Route path="/models" element={<Models />} />
            <Route path="/models/:id" element={<ModelDetail />} />
            <Route path="/math" element={<MathLab />} />
            <Route path="/data" element={<Data />} />
            <Route path="/advanced" element={<Advanced />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Home />} />
          </Routes>
        </main>
      </div>
      <Toasts />
    </div>
  )
}

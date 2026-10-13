import { useEffect, useState } from 'react'
import { NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { Home as HomeIcon, Sparkles, Calculator, Database, Trophy, Wrench, HeartPulse, Lock, Moon, Sun, Bell, BellOff } from 'lucide-react'
import { useHardware, getTheme, applyTheme } from './api'
import { Chip, Toasts, Progress } from './ui'
import { Mascot, tempMood } from './mascot'
import { Sparkline } from './charts'
import { CompanionProvider } from './companion'
import { installSparkles } from './effects'
import { soundOn, setSound, pop } from './sound'
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
  { section: 'Make something' },
  { to: '/train', label: 'Teach a model', icon: Sparkles },
  { to: '/math', label: 'Math playground', icon: Calculator },
  { to: '/data', label: 'My data', icon: Database },
  { section: 'Play' },
  { to: '/models', label: 'My little models', icon: Trophy },
  { section: 'Cozy corner' },
  { to: '/advanced', label: 'Toolbox', icon: Wrench },
  { to: '/settings', label: 'Care & comfort', icon: HeartPulse },
]

export const jobLink = (j) => (j.kind === 'math' ? '/math' : j.kind === 'data' ? '/data' : `/run/${encodeURIComponent(j.name)}`)
const JOB_WORD = { train: 'Chugging along', cooling: 'Taking a breather', math: 'Practising maths', data: 'Gathering data', guided: 'Getting ready' }

function JobPill() {
  const { hw } = useHardware()
  const nav = useNavigate()
  const job = hw?.jobs?.[0]
  if (!job) return <span className="job-pill"><Chip tone="ok">Resting · all quiet</Chip></span>
  return (
    <span className="job-pill" onClick={() => nav(jobLink(job))}>
      <Chip tone={job.kind === 'cooling' ? 'warn' : 'info'}>{JOB_WORD[job.kind] || 'Working'} · {job.name}{hw.jobs.length > 1 ? ` +${hw.jobs.length - 1}` : ''}</Chip>
    </span>
  )
}

function SideGpu() {
  const { hw, history } = useHardware()
  const nav = useNavigate()
  const g = hw?.gpu
  const busy = hw?.jobs?.length > 0
  const m = tempMood(g?.temp)
  return (
    <div className="side-gpu" onClick={() => nav('/settings')} title="Live readings straight from your graphics card">
      <div className="row" style={{ gap: 10, alignItems: 'center', marginBottom: 6 }}>
        <Mascot mood={m.tone === 'ok' ? (busy ? 'working' : 'happy') : 'toasty'} size={46} />
        <div style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 800, fontSize: 14, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{g?.name || 'Your GPU'}</div>
          <div className={`side-mood tone-text-${m.tone}`} style={{ margin: 0 }}>{g?.temp != null ? `${Math.round(g.temp)}°C · ${m.word}` : 'resting'}</div>
        </div>
      </div>
      <Progress value={g?.vram_used_gb != null && g?.vram_total_gb ? g.vram_used_gb / g.vram_total_gb : 0} />
      <div className="row spread small muted" style={{ marginTop: 6, fontWeight: 700 }}>
        <span>Memory {g?.vram_used_gb != null ? g.vram_used_gb.toFixed(1) : '–'} / {g?.vram_total_gb ? Math.round(g.vram_total_gb) : '–'} GB</span>
        <span>{g?.util != null ? `${Math.round(g.util)}%` : '–'}</span>
      </div>
      <Sparkline data={history.util} min={0} max={100} height={26} color="var(--accent)" />
    </div>
  )
}

function ThemeToggle() {
  const [t, setT] = useState(getTheme())
  const [snd, setSnd] = useState(soundOn())
  const set = (v) => { applyTheme(v); setT(v) }
  return (
    <div className="theme-toggle">
      <button className={`bell ${snd ? 'on' : ''}`} title={snd ? 'Little sounds are on' : 'Turn on little sounds'} onClick={() => { const v = !snd; setSound(v); setSnd(v); if (v) pop() }}>{snd ? <Bell size={13} /> : <BellOff size={13} />}</button>
      <button className={t === 'dusk' ? 'on' : ''} onClick={() => set('dusk')}><Moon size={12} style={{ verticalAlign: -1 }} /> Night</button>
      <button className={t === 'dawn' ? 'on' : ''} onClick={() => set('dawn')}><Sun size={12} style={{ verticalAlign: -1 }} /> Day</button>
    </div>
  )
}

export default function App() {
  const { hw } = useHardware()
  const loc = useLocation()
  const busy = !!hw?.jobs?.some((j) => j.kind !== 'data')
  useEffect(() => installSparkles(), [])
  return (
    <CompanionProvider>
    <div className="app">
      <div className="titlebar">
        <div className="brand" onClick={() => document.querySelector('.boi-btn')?.click()} title="Say hi to Boi"><Mascot mood="happy" size={30} />TrainerBoi</div>
        <JobPill />
      </div>
      <div className="body">
        <aside className="sidebar">
          <nav className="nav">
            {NAV.map((n, i) => n.section
              ? <div className="nav-section" key={i}>{n.section}</div>
              : <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => (isActive ? 'active' : '')}>
                <n.icon size={18} />{n.label}{n.to === '/train' && busy && <span className="dot" />}
              </NavLink>)}
          </nav>
          <div style={{ marginTop: 'auto' }} />
          <SideGpu />
          <div className="privacy"><Lock size={14} /> Everything stays on this computer</div>
          <ThemeToggle />
        </aside>
        <main className="main">
          <div className="page-anim" key={loc.pathname}>
          <Routes location={loc}>
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
          </div>
        </main>
      </div>
      <Toasts />
    </div>
    </CompanionProvider>
  )
}

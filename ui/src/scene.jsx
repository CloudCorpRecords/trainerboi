import { useMemo } from 'react'
import { Mascot } from './mascot'

// A storybook window onto the journey: scenery scrolls past while Boi chugs along, stations mark the steps, a cottage waits at the end.
const STATION = { check: ['🔭', 'Look around'], data: ['🥣', 'Gather data'], plan: ['🗺️', 'Make a plan'], train: ['🔥', 'Learn'], test: ['🎓', 'Test it'] }

function Hills({ d, className }) {
  const one = <path d={d} />
  return (
    <div className={`sc-strip ${className}`}>
      <div className="sc-strip-in">
        {[0, 1].map((k) => <svg key={k} viewBox="0 0 400 120" preserveAspectRatio="none">{one}</svg>)}
      </div>
    </div>
  )
}

function Trees() {
  const row = useMemo(() => [20, 70, 110, 190, 240, 310, 350].map((x, i) => ({ x, s: 0.8 + ((i * 7) % 5) / 10, c: i % 2 })), [])
  return (
    <div className="sc-strip trees">
      <div className="sc-strip-in">
        {[0, 1].map((k) => (
          <svg key={k} viewBox="0 0 400 80" preserveAspectRatio="none">
            {row.map((t, i) => (
              <g key={i} transform={`translate(${t.x} ${64 - 28 * t.s}) scale(${t.s})`}>
                <rect x="-2" y="26" width="4" height="12" rx="2" fill="var(--sc-trunk)" />
                <circle cx="0" cy="18" r="13" fill={t.c ? 'var(--sc-tree2)' : 'var(--sc-tree1)'} />
                <circle cx="-7" cy="24" r="8" fill={t.c ? 'var(--sc-tree2)' : 'var(--sc-tree1)'} /><circle cx="7" cy="24" r="8" fill={t.c ? 'var(--sc-tree2)' : 'var(--sc-tree1)'} />
              </g>
            ))}
          </svg>
        ))}
      </div>
    </div>
  )
}

export function Journey({ steps = [], progress = 0, state = 'running', hot = false, label }) {
  const running = state === 'running'
  const done = state === 'done'
  const bad = state === 'error' || state === 'crashed'
  const p = done ? 1 : Math.max(0, Math.min(1, progress))
  const mood = done ? 'cheer' : bad ? 'oops' : hot ? 'toasty' : running ? 'working' : 'sleepy'
  const n = steps.length || 5
  const stations = steps.map((s, i) => ({ id: s.id, frac: (i + 0.5) / (n + 0.5), state: s.state }))
  const WORLD = 150                                             // how far (in % of the window) a full journey spans
  const xOf = (frac) => 24 + (frac - p) * WORLD
  return (
    <div className={`scene ${running ? 'go' : ''} ${bad ? 'rainy' : ''} ${done ? 'arrived' : ''}`} aria-hidden="false">
      <div className="sc-sky" />
      <span className="sc-orb" />
      <span className="sc-cloud a" /><span className="sc-cloud b" /><span className="sc-cloud c" />
      {[...Array(10)].map((_, i) => <span key={i} className={`sc-fly f${i}`} />)}
      <Hills className="far" d="M0 90 Q60 30 130 74 T260 66 T400 84 V120 H0Z" />
      <Hills className="near" d="M0 100 Q70 56 150 92 T300 84 T400 98 V120 H0Z" />
      <Trees />
      <div className="sc-ground"><div className="sc-ties" /></div>

      {stations.map((s) => {
        const x = xOf(s.frac)
        const [emo, name] = STATION[s.id] || ['📍', s.id]
        return (
          <div key={s.id} className={`sc-station ${s.state}`} style={{ left: `${x}%` }} hidden={x < -12 || x > 112}>
            <div className="sc-sign"><span>{emo}</span></div><div className="sc-pole" /><div className="sc-name">{name}{s.state === 'done' ? ' ✓' : ''}</div>
          </div>
        )
      })}
      <div className="sc-home" style={{ left: `${xOf(1)}%` }} hidden={xOf(1) > 112}>
        <div className="sc-house"><div className="roof" /><div className="wall"><i className="door" /><i className="win" /></div><span className="chim" /><span className="smoke" /><span className="smoke s2" /></div>
      </div>

      <div className="sc-boi"><Mascot mood={mood} size={74} />{bad && <span className="sc-umbrella">☂️</span>}</div>
      {bad && <div className="sc-rain">{[...Array(26)].map((_, i) => <i key={i} style={{ left: `${(i * 37) % 100}%`, animationDelay: `${(i % 7) * 0.13}s` }} />)}</div>}
      {done && <div className="sc-fireworks">{['🎆', '✨', '🎇', '🌟', '🎉'].map((g, i) => <span key={i} style={{ left: `${12 + i * 19}%`, animationDelay: `${i * 0.5}s` }}>{g}</span>)}</div>}
      {label && <div className="sc-label">{label}</div>}
      <div className="sc-pct">{Math.round(p * 100)}%</div>
    </div>
  )
}

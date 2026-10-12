import { useMemo, useRef, useState } from 'react'

const nice = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2))

export function Sparkline({ data, color = 'var(--accent)', height = 34, min, max, fill = true }) {
  const pts = (data || []).map((v) => (v == null ? null : Number(v)))
  const vals = pts.filter((v) => v != null)
  if (vals.length < 2) return <div style={{ height }} />
  const lo = min ?? Math.min(...vals), hi = max ?? Math.max(...vals)
  const span = hi - lo || 1
  const W = 100
  const step = W / Math.max(pts.length - 1, 1)
  let d = '', first = null, last = null
  pts.forEach((v, i) => {
    if (v == null) return
    const x = i * step, y = height - 3 - ((v - lo) / span) * (height - 6)
    d += `${d ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`
    if (first == null) first = x
    last = x
  })
  return (
    <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" width="100%" height={height} className="spark">
      {fill && <path d={`${d}L${last},${height}L${first},${height}Z`} fill={color} opacity="0.12" />}
      <path d={d} fill="none" stroke={color} strokeWidth="1.6" vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
    </svg>
  )
}

/** Multi-series line chart with hover read-out. series: [{name,color,points:[{x,y}],dashed,faint}] */
export function LineChart({ series, height = 230, yLabel = '', xLabel = 'step', logY = false, empty = 'Waiting for data…' }) {
  const ref = useRef(null)
  const [hover, setHover] = useState(null)
  const all = series.flatMap((s) => s.points.filter((p) => p.y != null && isFinite(p.y)))
  const m = { l: 44, r: 12, t: 10, b: 24 }
  const W = 640, H = height
  const view = useMemo(() => {
    if (all.length < 2) return null
    const xs = all.map((p) => p.x), ys = all.map((p) => p.y)
    const x0 = Math.min(...xs), x1 = Math.max(...xs)
    let y0 = Math.min(...ys), y1 = Math.max(...ys)
    const pad = (y1 - y0) * 0.08 || 0.1
    y0 = Math.max(0, y0 - pad); y1 += pad
    return { x0, x1: x1 === x0 ? x0 + 1 : x1, y0, y1 }
  }, [series])
  if (!view) return <div className="chart-empty" style={{ height }}>{empty}</div>
  const X = (x) => m.l + ((x - view.x0) / (view.x1 - view.x0)) * (W - m.l - m.r)
  const Y = (y) => H - m.b - ((y - view.y0) / (view.y1 - view.y0)) * (H - m.t - m.b)
  const ticks = 4
  const path = (pts) => pts.filter((p) => p.y != null && isFinite(p.y)).map((p, i) => `${i ? 'L' : 'M'}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join('')

  const onMove = (e) => {
    const r = ref.current.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * W
    const x = view.x0 + ((px - m.l) / (W - m.l - m.r)) * (view.x1 - view.x0)
    const main = series.find((s) => s.points.length) || series[0]
    let best = null
    for (const p of main.points) if (p.y != null && (best == null || Math.abs(p.x - x) < Math.abs(best.x - x))) best = p
    setHover(best)
  }
  return (
    <div className="chart">
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} width="100%" height={H} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {Array.from({ length: ticks + 1 }).map((_, i) => {
          const v = view.y0 + ((view.y1 - view.y0) * i) / ticks
          return <g key={i}><line x1={m.l} x2={W - m.r} y1={Y(v)} y2={Y(v)} className="grid" /><text x={m.l - 6} y={Y(v) + 3} textAnchor="end" className="axis">{nice(v)}</text></g>
        })}
        {[0, 0.5, 1].map((f, i) => {
          const v = view.x0 + (view.x1 - view.x0) * f
          return <text key={i} x={X(v)} y={H - 6} textAnchor={i === 0 ? 'start' : i === 2 ? 'end' : 'middle'} className="axis">{Math.round(v).toLocaleString()}</text>
        })}
        {series.map((s) => (
          <path key={s.name} d={path(s.points)} fill="none" stroke={s.color} strokeWidth={s.faint ? 1 : 2} opacity={s.faint ? 0.35 : 1}
            strokeDasharray={s.dashed ? '5 4' : undefined} strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {hover && <g><line x1={X(hover.x)} x2={X(hover.x)} y1={m.t} y2={H - m.b} className="cursor" /><circle cx={X(hover.x)} cy={Y(hover.y)} r="4" fill={series[0].color} /></g>}
      </svg>
      <div className="chart-legend">
        {series.map((s) => <span key={s.name}><i style={{ background: s.color, opacity: s.faint ? 0.4 : 1 }} />{s.name}</span>)}
        {hover && <b>{xLabel} {hover.x.toLocaleString()} · {yLabel}{nice(hover.y)}</b>}
      </div>
    </div>
  )
}

export function Bars({ rows, max = 100, unit = '%' }) {
  return (
    <div className="bars">
      {rows.map((r) => (
        <div key={r.label} className="bar-row">
          <span className="bar-label">{r.label}</span>
          <div className="bar-track">
            {r.before != null && <div className="bar-before" style={{ width: `${(r.before / max) * 100}%` }} />}
            <div className="bar-fill" style={{ width: `${(r.value / max) * 100}%`, background: r.color || 'var(--accent)' }} />
          </div>
          <span className="bar-val">{r.value < 10 && r.value > 0 ? r.value.toFixed(1) : Math.round(r.value)}{unit}</span>
        </div>
      ))}
    </div>
  )
}

/** Exponential moving average for noisy loss curves. */
export function smooth(points, a = 0.15) {
  let s = null
  return points.map((p) => {
    if (p.y == null) return p
    s = s == null ? p.y : a * p.y + (1 - a) * s
    return { x: p.x, y: s }
  })
}

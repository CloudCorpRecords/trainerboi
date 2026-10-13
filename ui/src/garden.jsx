import { Link } from 'react-router-dom'
import { burst } from './effects'

// petal colour per kind of model
const PETAL = { sft: '#ff9ecb', pretrain: '#ffd36b', math: '#8ee6c4', dpo: '#b9a5ff', orpo: '#b9a5ff', kto: '#b9a5ff', grpo: '#ffab7d', exported: '#8fd3ff', embedding: '#ffab7d' }

function Flower({ color, delay, tall }) {
  const h = tall ? 78 : 66
  return (
    <svg className="flower" viewBox="0 0 48 80" width="46" height="76" style={{ animationDelay: `${delay}s` }}>
      <path d={`M24 78 C24 ${h - 14} 21 ${h - 28} 24 30`} stroke="#6fcf97" strokeWidth="3.2" fill="none" strokeLinecap="round" />
      <ellipse cx="16" cy="58" rx="8" ry="3.8" fill="#6fcf97" transform="rotate(-28 16 58)" />
      <ellipse cx="32" cy="48" rx="8" ry="3.8" fill="#8fe0a8" transform="rotate(28 32 48)" />
      <g className="bloom">
        {[0, 72, 144, 216, 288].map((a) => (
          <circle key={a} cx={24 + 9 * Math.cos((a * Math.PI) / 180)} cy={24 + 9 * Math.sin((a * Math.PI) / 180)} r="7.2" fill={color} />
        ))}
        <circle cx="24" cy="24" r="6" fill="#ffe29a" />
        <circle cx="22.4" cy="22.6" r="1.1" fill="#3a2b63" opacity=".7" /><circle cx="25.8" cy="22.6" r="1.1" fill="#3a2b63" opacity=".7" /><path d="M22 26.2 Q24 28 26 26.2" stroke="#3a2b63" strokeWidth="1" fill="none" strokeLinecap="round" opacity=".7" />
      </g>
    </svg>
  )
}

/** One flower for every model you've trained. */
export function Garden({ models }) {
  const list = (models || []).slice(0, 16)
  return (
    <div className="garden">
      <span className="g-sun" aria-hidden="true" />
      <span className="g-butterfly" aria-hidden="true">🦋</span>
      <div className="g-row">
        {list.map((m, i) => (
          <Link key={m.id} to={`/models/${encodeURIComponent(m.id)}`} className="g-spot" title={`${m.title} — ${m.metric || m.goal || ''}`}
            onMouseEnter={(e) => { const r = e.currentTarget.getBoundingClientRect(); burst(r.left + r.width / 2, r.top + 24, ['🌼', '✨', '🌸'], 5) }}>
            <Flower color={PETAL[m.kind] || '#ff9ecb'} delay={(i % 5) * 0.35} tall={i % 3 === 0} />
            <span className="g-name">{m.title}</span>
          </Link>
        ))}
        {!list.length && (
          <div className="g-seed">
            <svg viewBox="0 0 48 80" width="46" height="76"><g className="sprout"><path d="M24 78 C24 66 24 58 24 52" stroke="#6fcf97" strokeWidth="3.2" fill="none" strokeLinecap="round" /><ellipse cx="17" cy="50" rx="9" ry="4.5" fill="#8fe0a8" transform="rotate(-30 17 50)" /><ellipse cx="31" cy="48" rx="9" ry="4.5" fill="#6fcf97" transform="rotate(30 31 48)" /></g></svg>
            <span>Your garden is waiting for its first flower 🌱</span>
          </div>
        )}
      </div>
      <div className="g-ground" />
    </div>
  )
}

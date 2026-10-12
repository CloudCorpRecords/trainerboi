import { useEffect, useState } from 'react'
import { toast as _t } from './api'
import { X, Check, AlertTriangle, Info } from 'lucide-react'

export const Button = ({ variant = 'default', size, icon: Icon, children, className = '', ...p }) => (
  <button className={`btn btn-${variant} ${size === 'sm' ? 'btn-sm' : ''} ${className}`} {...p}>
    {Icon && <Icon size={size === 'sm' ? 14 : 16} />}{children}
  </button>
)

export const Card = ({ title, action, children, className = '', pad = true, ...p }) => (
  <section className={`card ${className}`} {...p}>
    {(title || action) && <header className="card-head"><h3>{title}</h3><div>{action}</div></header>}
    <div className={pad ? 'card-body' : ''}>{children}</div>
  </section>
)

export const Chip = ({ tone = 'neutral', icon: Icon, children }) => (
  <span className={`chip chip-${tone}`}>{Icon && <Icon size={12} />}{children}</span>
)

export const Progress = ({ value, indeterminate, tone = 'accent' }) => (
  <div className={`progress ${indeterminate || value == null ? 'indet' : ''} tone-${tone}`}>
    <div style={{ width: value == null ? undefined : `${Math.max(2, Math.min(100, value * 100))}%` }} />
  </div>
)

export const Stat = ({ label, value, unit, sub, tone, children }) => (
  <div className="stat">
    <div className="stat-label">{label}</div>
    <div className={`stat-value ${tone ? `tone-text-${tone}` : ''}`}>{value}{unit && <span>{unit}</span>}</div>
    {sub && <div className="stat-sub">{sub}</div>}
    {children}
  </div>
)

export const PageHeader = ({ title, sub, actions }) => (
  <div className="page-head">
    <div><h1>{title}</h1>{sub && <p>{sub}</p>}</div>
    <div className="page-actions">{actions}</div>
  </div>
)

export const Empty = ({ icon: Icon, title, text, action }) => (
  <div className="empty">{Icon && <Icon size={30} />}<h3>{title}</h3><p>{text}</p>{action}</div>
)

export const Field = ({ label, hint, children }) => (
  <label className="field"><span className="field-label">{label}</span>{children}{hint && <span className="field-hint">{hint}</span>}</label>
)

export function Tabs({ tabs, value, onChange }) {
  return <div className="tabs">{tabs.map((t) => <button key={t.id} className={value === t.id ? 'on' : ''} onClick={() => onChange(t.id)}>{t.icon && <t.icon size={15} />}{t.label}</button>)}</div>
}

export function Toasts() {
  const [items, setItems] = useState([])
  useEffect(() => {
    const h = (e) => {
      setItems((x) => [...x, e.detail])
      setTimeout(() => setItems((x) => x.filter((i) => i.id !== e.detail.id)), 5200)
    }
    window.addEventListener('tb-toast', h)
    return () => window.removeEventListener('tb-toast', h)
  }, [])
  return (
    <div className="toasts">
      {items.map((t) => (
        <div key={t.id} className={`toast toast-${t.kind}`}>
          {t.kind === 'error' ? <AlertTriangle size={16} /> : t.kind === 'success' ? <Check size={16} /> : <Info size={16} />}
          <span>{t.msg}</span>
          <button onClick={() => setItems((x) => x.filter((i) => i.id !== t.id))}><X size={14} /></button>
        </div>
      ))}
    </div>
  )
}

export const toast = _t

/** Render the small bit of markdown our server text uses: **bold**, *italic*, `code`. */
export function Md({ text }) {
  const parts = String(text || '').split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g)
  return <>{parts.map((p, i) => p.startsWith('**') ? <b key={i}>{p.slice(2, -2)}</b> : p.startsWith('`') ? <code key={i}>{p.slice(1, -1)}</code>
    : p.startsWith('*') && p.length > 2 ? <i key={i}>{p.slice(1, -1)}</i> : <span key={i}>{p}</span>)}</>
}

import { useEffect, useMemo, useRef, useState } from 'react'
import { getHat } from './rewards'
import { LOADING, pick } from './phrases'

// ---------------------------------------------------------------- eyes that follow the pointer (one shared listener)
const watchers = new Set()
let listening = false
const pointer = { x: -1, y: -1 }
function startListening() {
  if (listening) return
  listening = true
  let raf = 0
  window.addEventListener('mousemove', (e) => {
    pointer.x = e.clientX; pointer.y = e.clientY
    if (raf) return
    raf = requestAnimationFrame(() => { raf = 0; watchers.forEach((fn) => fn()) })
  }, { passive: true })
}

/** "Boi" - the little train. Moods: happy, working, sleepy, toasty, cheer, oops, think, love. Pure SVG + CSS, themed by --m-* variables. */
export function Mascot({ mood = 'happy', size = 96, className = '', title, follow = true, hat }) {
  const ref = useRef(null)
  const [worn, setWorn] = useState(getHat())
  useEffect(() => { const h = (e) => setWorn(e.detail); window.addEventListener('tb-hat', h); return () => window.removeEventListener('tb-hat', h) }, [])
  const wearing = hat || worn
  useEffect(() => {
    if (!follow) return
    startListening()
    const fn = () => {
      const el = ref.current
      if (!el || pointer.x < 0) return
      const r = el.getBoundingClientRect()
      const dx = pointer.x - (r.left + r.width / 2), dy = pointer.y - (r.top + r.height / 2)
      const k = Math.max(1, Math.hypot(dx, dy) / 120)
      el.style.setProperty('--lx', `${(dx / k / 120) * 3.4}px`)
      el.style.setProperty('--ly', `${(dy / k / 120) * 2.6}px`)
    }
    watchers.add(fn)
    return () => watchers.delete(fn)
  }, [follow])

  const eyes = {
    happy: [<Eye key="l" x={47} />, <Eye key="r" x={73} />],
    working: [<Eye key="l" x={47} look />, <Eye key="r" x={73} look />],
    toasty: [<Eye key="l" x={47} />, <Eye key="r" x={73} />],
    oops: [<Eye key="l" x={47} big />, <Eye key="r" x={73} big />],
    think: [<Eye key="l" x={47} up />, <Eye key="r" x={73} up />],
    sleepy: [<path key="l" d="M40 63 Q47 69 54 63" className="m-line" />, <path key="r" d="M66 63 Q73 69 80 63" className="m-line" />],
    cheer: [<path key="l" d="M40 65 Q47 56 54 65" className="m-line" />, <path key="r" d="M66 65 Q73 56 80 65" className="m-line" />],
    love: [<path key="l" className="m-heart-eye" d="M47 68 C37 60 40 52 47 58 C54 52 57 60 47 68Z" />, <path key="r" className="m-heart-eye" d="M73 68 C63 60 66 52 73 58 C80 52 83 60 73 68Z" />],
  }[mood] || null
  const mouth = {
    happy: <path d="M52 73 Q60 81 68 73" className="m-line" />,
    working: <path d="M53 74 Q60 79 67 74" className="m-line" />,
    sleepy: <path d="M56 75 Q60 77.5 64 75" className="m-line" />,
    toasty: <path d="M51 75 Q55.5 71 60 75 T69 75" className="m-line" />,
    cheer: <path d="M50 70 Q60 88 70 70 Z" fill="#ff7fa8" stroke="var(--m-ink)" strokeWidth="2.5" strokeLinejoin="round" />,
    oops: <path d="M53 79 Q60 72 67 79" className="m-line" />,
    think: <ellipse cx="60" cy="76" rx="3.4" ry="4" fill="var(--m-ink)" />,
    love: <path d="M52 73 Q60 82 68 73" className="m-line" />,
  }[mood]
  return (
    <svg ref={ref} className={`mascot mood-${mood} ${className}`} width={size} height={size} viewBox="0 0 120 120" role="img" aria-label={title || 'Boi the little train'}>
      {title && <title>{title}</title>}
      <g className="m-puffs">
        <circle cx="88" cy="16" r="5" /><circle cx="95" cy="9" r="4" /><circle cx="84" cy="7" r="3" />
      </g>
      <g className="m-bob">
        <rect x="78" y="21" width="14" height="20" rx="4" fill="var(--m-accent2)" />
        <rect x="74" y="17" width="22" height="9" rx="4.5" fill="var(--m-accent)" />
        <rect x="22" y="22" width="34" height="22" rx="9" fill="var(--m-roof)" />
        <rect x="14" y="34" width="92" height="58" rx="28" fill="var(--m-body)" />
        <ellipse cx="38" cy="46" rx="14" ry="5" fill="#fff" opacity=".22" />
        <ellipse cx="42" cy="73" rx="6.5" ry="4" fill="var(--m-cheek)" opacity=".75" />
        <ellipse cx="78" cy="73" rx="6.5" ry="4" fill="var(--m-cheek)" opacity=".75" />
        {(mood === 'toasty' || mood === 'love') && <><ellipse cx="42" cy="73" rx="7.5" ry="4.8" fill="#ff7fa8" opacity=".5" /><ellipse cx="78" cy="73" rx="7.5" ry="4.8" fill="#ff7fa8" opacity=".5" /></>}
        <g className="m-eyes">{eyes}</g>
        {mood === 'oops' && <><path d="M41 50 L55 54" className="m-line thin" /><path d="M79 50 L65 54" className="m-line thin" /></>}
        {mood === 'toasty' && <path className="m-drop" d="M92 46 Q96 53 92 56 Q88 53 92 46Z" fill="#8fd3ff" />}
        {mouth}
        {mood === 'sleepy' && <g className="m-zzz"><text x="92" y="38" fontSize="12">z</text><text x="100" y="28" fontSize="9">z</text></g>}
        {mood === 'think' && <g className="m-thought"><circle cx="98" cy="34" r="2.2" /><circle cx="104" cy="25" r="3.2" /><circle cx="112" cy="14" r="4.4" /></g>}
        {mood === 'love' && <g className="m-hearts">
          <path d="M20 40 C13 34 15 28 20 32 C25 28 27 34 20 40Z" /><path d="M100 46 C95 41 96 36 100 39 C104 36 105 41 100 46Z" /><path d="M64 14 C57 8 59 2 64 6 C69 2 71 8 64 14Z" />
        </g>}
        {mood === 'cheer' && <g className="m-sparks">
          <path d="M14 30 l2 5 5 2 -5 2 -2 5 -2 -5 -5 -2 5 -2z" /><path d="M104 38 l1.5 4 4 1.5 -4 1.5 -1.5 4 -1.5 -4 -4 -1.5 4 -1.5z" /><path d="M60 8 l1.5 4 4 1.5 -4 1.5 -1.5 4 -1.5 -4 -4 -1.5 4 -1.5z" />
        </g>}
        {wearing && wearing !== 'none' && <Hat kind={wearing} />}        <g className="m-wheels">
          {[34, 60, 86].map((x) => (
            <g key={x}><circle cx={x} cy="98" r="11" fill="var(--m-wheel)" /><g className="m-wheel"><circle cx={x} cy="98" r="5" fill="var(--m-accent)" /><rect x={x - 1} y="88" width="2" height="6" rx="1" fill="var(--m-ink)" opacity=".35" /></g></g>
          ))}
        </g>
      </g>
      <rect x="6" y="110" width="108" height="3.5" rx="1.75" fill="var(--m-track)" />
    </svg>
  )
}

function Hat({ kind }) {
  const INK = 'var(--m-ink)'
  switch (kind) {
    case 'party': return <g className="m-hat"><path d="M28 25 L50 25 L39 4Z" fill="#ff9ecb" /><path d="M31 21 L47 21" stroke="#fff" strokeWidth="2.4" opacity=".7" /><path d="M34 14 L44 14" stroke="#ffe29a" strokeWidth="2.4" /><circle cx="39" cy="4" r="3.6" fill="#ffe29a" /></g>
    case 'flowers': return <g className="m-hat">{[[26, 35, '#ff9ecb'], [41, 32, '#ffe29a'], [57, 33, '#8ee6c4'], [73, 33, '#ff9ecb'], [90, 36, '#ffe29a']].map(([x, y, c], i) => <g key={i}>{[0, 72, 144, 216, 288].map((a) => <circle key={a} cx={x + 3.4 * Math.cos(a * Math.PI / 180)} cy={y + 3.4 * Math.sin(a * Math.PI / 180)} r="2.7" fill={c} />)}<circle cx={x} cy={y} r="2.2" fill="#fff6c9" /></g>)}<path d="M20 38 Q58 28 98 40" stroke="#6fcf97" strokeWidth="2" fill="none" opacity=".8" /></g>
    case 'glasses': return <g className="m-hat" fill="#fff" fillOpacity=".14" stroke={INK} strokeWidth="2.6"><circle cx="47" cy="62" r="12" /><circle cx="73" cy="62" r="12" /><path d="M59 62 Q60 60 61 62" fill="none" /><path d="M35 60 L28 56 M85 60 L92 56" fill="none" strokeLinecap="round" /></g>
    case 'headphones': return <g className="m-hat"><path d="M20 58 Q20 26 60 26 Q100 26 100 58" stroke="#ffab7d" strokeWidth="5" fill="none" strokeLinecap="round" /><rect x="11" y="52" width="12" height="22" rx="6" fill="#ff9ecb" /><rect x="97" y="52" width="12" height="22" rx="6" fill="#ff9ecb" /></g>
    case 'crown': return <g className="m-hat"><path d="M26 24 L26 9 L32.5 17 L39 6 L45.5 17 L52 9 L52 24Z" fill="#ffd36b" stroke="#ffab7d" strokeWidth="2" strokeLinejoin="round" /><circle cx="39" cy="14" r="2.3" fill="#ff9ecb" /></g>
    case 'sparkle': return <g className="m-hat"><path d="M24 30 Q39 12 54 30" stroke="#d3c6ff" strokeWidth="3" fill="none" strokeLinecap="round" /><path d="M39 4 l2.6 6.4 6.4 2.6 -6.4 2.6 -2.6 6.4 -2.6 -6.4 -6.4 -2.6 6.4 -2.6z" fill="#ffe29a" /></g>
    case 'bow': return <g className="m-hat"><path d="M30 30 L14 20 L14 40Z" fill="#ff9ecb" /><path d="M30 30 L46 20 L46 40Z" fill="#ff7fb5" /><circle cx="30" cy="30" r="5" fill="#ffc0dd" /></g>
    case 'chef': return <g className="m-hat"><rect x="26" y="26" width="26" height="10" rx="3" fill="#fff" /><circle cx="29" cy="20" r="8" fill="#fff" /><circle cx="39" cy="15" r="9.5" fill="#fff" /><circle cx="50" cy="20" r="8" fill="#fff" /><path d="M26 31 L52 31" stroke="#e6dcff" strokeWidth="2" /></g>
    case 'hardhat': return <g className="m-hat"><path d="M24 36 Q24 14 39 14 Q54 14 54 36Z" fill="#ffd36b" stroke="#ffab7d" strokeWidth="2" /><rect x="20" y="34" width="38" height="5" rx="2.5" fill="#ffab7d" /><rect x="35" y="14" width="8" height="12" rx="2" fill="#ffc15a" /></g>
    case 'tophat': return <g className="m-hat"><rect x="26" y="6" width="26" height="26" rx="3" fill="#3a2b63" /><rect x="22" y="30" width="34" height="6" rx="3" fill="#3a2b63" /><rect x="26" y="22" width="26" height="5" fill="#ff9ecb" /></g>
    case 'nightcap': return <g className="m-hat"><path d="M22 36 Q30 -2 72 14 Q50 14 56 36Z" fill="#8fd3ff" /><path d="M22 36 L58 36" stroke="#fff" strokeWidth="5" strokeLinecap="round" opacity=".9" /><circle cx="73" cy="14" r="5" fill="#fff" /></g>
    default: return null
  }
}

function Eye({ x, big, look, up }) {
  const r = big ? [7.5, 9] : [6, 8]
  const ox = look ? 1.2 : up ? 2 : 0
  const oy = up ? -3.5 : 0
  return (
    <g transform={`translate(${x} 62)`}>
      <g className="m-look">
        <g className="m-eye">
          <ellipse cx={ox} cy={oy} rx={r[0]} ry={r[1]} fill="var(--m-ink)" />
          <circle cx={ox + 2.2} cy={oy - 3} r="2.3" fill="#fff" />
          <circle cx={ox - 2} cy={oy + 3} r="1.1" fill="#fff" opacity=".8" />
        </g>
      </g>
    </g>
  )
}

const FLAG = (v) => (v == null ? '🌱' : v >= 1 ? '🌸' : v > 0.66 ? '🌷' : v > 0.33 ? '🌿' : '🌱')

/** A little track Boi rides along; the seedling at the end grows as he gets closer. value 0..1, or null for "unknown length". */
export function Track({ value, label, mood }) {
  const v = value == null ? null : Math.max(0, Math.min(1, value))
  const m = mood || (v === 1 ? 'cheer' : 'working')
  return (
    <div className="track-wrap">
      <div className="track">
        <div className="track-fill" style={{ width: v == null ? '0%' : `${v * 100}%` }} />
        <div className="track-ties" />
        <div className={`track-boi ${v == null ? 'roam' : ''}`} style={v == null ? undefined : { left: `calc(${v * 100}% - ${v * 44}px)` }}>
          <Mascot mood={m} size={52} />
        </div>
        <div className="track-flag" aria-hidden="true" key={FLAG(v)}>{FLAG(v)}</div>
      </div>
      {label && <div className="track-label">{label}</div>}
    </div>
  )
}

/** Soft floating clouds / twinkling stars / a moon or sun behind a hero. Decorative only. */
export function Sky() {
  return (
    <div className="sky" aria-hidden="true">
      <span className="orb" />
      <span className="rainbow" />
      <span className="cloud c1" /><span className="cloud c2" /><span className="cloud c3" />
      <span className="balloon b1">🎈</span><span className="balloon b2">🎈</span>
      <span className="bird bd1">🕊️</span><span className="bird bd2">🕊️</span>
      {[...Array(8)].map((_, i) => <span key={i} className={`firefly ff${i}`} />)}
      <svg className="sky-hills" viewBox="0 0 400 60" preserveAspectRatio="none"><path d="M0 38 Q70 6 150 34 T300 28 T400 36 V60 H0Z" fill="var(--hill-a)" /><path d="M0 50 Q90 26 190 46 T400 42 V60 H0Z" fill="var(--hill-b)" /></svg>
      {[...Array(9)].map((_, i) => <span key={i} className={`star s${i}`}>✦</span>)}
    </div>
  )
}

/** One gentle burst of petals and sparkles. Plays once. */
export function Confetti({ play }) {
  const [on, setOn] = useState(false)
  useEffect(() => { if (play) { setOn(true); const t = setTimeout(() => setOn(false), 5200); return () => clearTimeout(t) } }, [play])
  const bits = useMemo(() => [...Array(34)].map((_, i) => ({
    left: (i * 29) % 100, delay: ((i * 37) % 100) / 100 * 1.4, dur: 2.8 + ((i * 13) % 20) / 10,
    glyph: ['🌸', '✨', '💜', '⭐', '🎈', '🌼'][i % 6], size: 14 + ((i * 7) % 12),
  })), [])
  if (!on) return null
  return (
    <div className="confetti" aria-hidden="true">
      {bits.map((b, i) => <span key={i} style={{ left: `${b.left}%`, animationDelay: `${b.delay}s`, animationDuration: `${b.dur}s`, fontSize: b.size }}>{b.glyph}</span>)}
    </div>
  )
}

/** How warm the GPU feels, in words (the number is always shown next to it). */
export function tempMood(t) {
  if (t == null) return { word: 'resting', tone: 'ok', mood: 'happy' }
  if (t < 55) return { word: 'cool and comfy', tone: 'ok', mood: 'happy' }
  if (t < 70) return { word: 'nice and warm', tone: 'ok', mood: 'happy' }
  if (t < 79) return { word: 'getting toasty', tone: 'warn', mood: 'toasty' }
  return { word: 'needs a little breather', tone: 'bad', mood: 'toasty' }
}

/** Mascot + three dots while something loads. */
export function Loading({ text, mood = 'working' }) {
  const line = useMemo(() => text || pick(LOADING), [text])
  return <div className="row muted loading" style={{ fontWeight: 700 }}><Mascot mood={mood} size={54} /> <span>{line}</span><span className="dots"><i /><i /><i /></span></div>
}

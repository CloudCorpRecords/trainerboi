import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search } from 'lucide-react'
import { usePoll, applyTheme, getTheme } from './api'
import { Mascot } from './mascot'
import { useCompanion } from './companion'
import { RECIPES } from './recipes'
import { setDust, dustOn } from './pixie'
import { burst } from './effects'
import { toot } from './sound'

const PAGES = [
  ['🏠', 'Home', '/', 'start'], ['✨', 'Teach a model', '/train', 'train guided new wizard'], ['🧮', 'Math playground', '/math', 'maths numbers practise'],
  ['🥣', 'Gather data', '/gather', 'data builder kitchen files links'], ['🔧', 'Workshop', '/workshop', 'manual training dpo orpo kto grpo settings knobs'],
  ['🌱', 'My data', '/data', 'datasets quality health clean'], ['💬', 'My little models', '/models', 'chat share export zip'], ['🍰', 'Taste test', '/taste', 'compare arena battle'],
  ['🌟', "Boi's trail", '/stickers', 'stickers hats closet rewards tricks rank unlock'], ['🧰', 'Toolbox', '/advanced', 'rag audio autotrain tune'], ['💜', 'Care & comfort', '/settings', 'gpu health temperature settings'],
]
const JOKES = [
  'Why did the model go to school? To get a little more training! 🎓', 'What do you call a sleepy neural network? A snooze-ral net. 😴', 'I told my GPU a joke. It said it was too hot to laugh. 🔥',
  'Why was the loss so calm? It had hit rock-bottom… in a good way. 📉', 'How does a train-er say hello? Chugga-chugga-choo! 🚂', 'My favourite number is 0.0001. It\'s small but it learns a lot. 🌱',
]

export function Palette({ open, setOpen }) {
  const nav = useNavigate()
  const { say } = useCompanion()
  const [q, setQ] = useState('')
  const [i, setI] = useState(0)
  const inp = useRef(null)
  const { data: models } = usePoll(open ? '/api/models' : null, 60000)
  const { data: ds } = usePoll(open ? '/api/datasets' : null, 60000)
  useEffect(() => { if (open) { setQ(''); setI(0); setTimeout(() => inp.current?.focus(), 30) } }, [open])

  const items = useMemo(() => {
    const go = (p) => () => nav(p)
    const list = [
      ...PAGES.map(([e, l, p, k]) => ({ e, l, k, hint: 'Go to', run: go(p) })),
      ...RECIPES.map((r) => ({ e: r.emoji, l: r.title, k: 'recipe ' + r.blurb, hint: 'Recipe', run: go(r.link || `/train?recipe=${r.id}`) })),
      ...(models || []).map((m) => ({ e: '💬', l: m.title, k: 'model chat ' + m.kind, hint: 'Open model', run: go(`/models/${encodeURIComponent(m.id)}`) })),
      ...(ds || []).filter((d) => d.state === 'done').map((d) => ({ e: '🌱', l: d.name, k: 'dataset data', hint: 'Dataset', run: go('/data') })),
      { e: getTheme() === 'dusk' ? '☀️' : '🌙', l: getTheme() === 'dusk' ? 'Switch to soft day' : 'Switch to cozy night', k: 'theme dark light', hint: 'Do', run: () => applyTheme(getTheme() === 'dusk' ? 'dawn' : 'dusk') },
      { e: '✨', l: dustOn() ? 'Turn pixie dust off' : 'Sprinkle pixie dust', k: 'sparkle cursor', hint: 'Do', run: () => setDust(!dustOn()) },
      { e: '🎉', l: 'Tell me a joke', k: 'funny laugh', hint: 'Boi', run: () => say(JOKES[Math.floor(Math.random() * JOKES.length)], 'cheer', 9000) },
      { e: '🙌', l: 'Give Boi a high five', k: 'hello hi', hint: 'Boi', run: () => { toot(); burst(window.innerWidth - 80, window.innerHeight - 90, ['✨', '💜', '🌟'], 12); say('High five! 🙌 You\'re wonderful.', 'love', 5000) } },
    ]
    const words = q.toLowerCase().split(/\s+/).filter(Boolean)
    return list.filter((x) => words.every((w) => (x.l + ' ' + (x.k || '')).toLowerCase().includes(w))).slice(0, 9)
  }, [q, models, ds, open])
  useEffect(() => setI(0), [q])

  if (!open) return null
  const choose = (x) => { setOpen(false); x?.run() }
  return (
    <div className="palette-back" onMouseDown={() => setOpen(false)}>
      <div className="palette" onMouseDown={(e) => e.stopPropagation()}>
        <div className="palette-head"><Mascot mood="think" size={46} /><div><b>What would you like to do?</b><div className="small muted">Type anything — “math”, “pirate”, “share”, “joke”…</div></div></div>
        <div className="palette-input"><Search size={16} /><input ref={inp} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask Boi…"
          onKeyDown={(e) => { if (e.key === 'ArrowDown') { e.preventDefault(); setI((v) => Math.min(items.length - 1, v + 1)) } else if (e.key === 'ArrowUp') { e.preventDefault(); setI((v) => Math.max(0, v - 1)) } else if (e.key === 'Enter') choose(items[i]); else if (e.key === 'Escape') setOpen(false) }} /></div>
        <div className="palette-list">
          {items.map((x, k) => <button key={x.l + k} className={k === i ? 'on' : ''} onMouseEnter={() => setI(k)} onClick={() => choose(x)}><span className="pe">{x.e}</span><span className="pl">{x.l}</span><small>{x.hint}</small></button>)}
          {!items.length && <div className="empty" style={{ padding: 22 }}>Hmm, I couldn't find that. Try another word? 🔍</div>}
        </div>
        <div className="palette-foot small faint">↑ ↓ to choose · Enter to go · Esc to close</div>
      </div>
    </div>
  )
}

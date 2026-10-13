import { useEffect, useState } from 'react'
import { Lock } from 'lucide-react'
import { Card, PageHeader } from '../ui'
import { Mascot } from '../mascot'
import { STICKERS, HATS, earned, hasHat, getHat, setHat } from '../rewards'
import { useCompanion } from '../companion'

export default function Stickers() {
  const [got, setGot] = useState(earned())
  const [hat, setH] = useState(getHat())
  const { say } = useCompanion()
  useEffect(() => { const h = () => setGot(earned()); window.addEventListener('tb-earned', h); return () => window.removeEventListener('tb-earned', h) }, [])
  const count = STICKERS.filter((s) => got[s.id]).length
  const pick = (id) => {
    if (!hasHat(id)) return
    setHat(id); setH(id)
    say(id === 'none' ? 'Back to my plain old self! 🚂' : 'Ooh, how do I look? 💜', id === 'none' ? 'happy' : 'love', 4500)
  }
  return (
    <div className="page">
      <PageHeader title="Sticker book 🌟" sub="Little rewards for real things you do — and some of them unlock hats for Boi. No pressure, just fun." />
      <div className="grid" style={{ gridTemplateColumns: '1.6fr 1fr', alignItems: 'start' }}>
        <Card title={`Your stickers · ${count} of ${STICKERS.length}`}>
          <div className="sticker-sheet">
            {STICKERS.map((s, i) => {
              const ok = !!got[s.id]
              return (
                <div key={s.id} className={`sticker ${ok ? 'got' : 'locked'}`} style={{ '--tilt': `${((i * 37) % 9) - 4}deg` }} title={ok ? s.title : s.how}>
                  <div className="sticker-pic">{ok ? s.emoji : '❔'}</div>
                  <b>{ok ? s.title : '???'}</b>
                  <span>{ok ? (s.hat ? `Unlocked: ${HATS.find((h) => h.id === s.hat)?.label}` : new Date(got[s.id]).toLocaleDateString()) : s.how}</span>
                </div>
              )
            })}
          </div>
        </Card>
        <Card title="Boi's closet 🎩">
          <div className="closet-preview"><Mascot mood="happy" size={150} /></div>
          <div className="closet">
            {HATS.map((h) => {
              const ok = hasHat(h.id)
              return (
                <button key={h.id} className={`hat-btn ${hat === h.id ? 'on' : ''}`} disabled={!ok} onClick={() => pick(h.id)} title={ok ? h.label : 'Earn a sticker to unlock'}>
                  <span>{ok ? h.emoji : <Lock size={15} />}</span><small>{ok ? h.label : 'locked'}</small>
                </button>
              )
            })}
          </div>
        </Card>
      </div>
    </div>
  )
}

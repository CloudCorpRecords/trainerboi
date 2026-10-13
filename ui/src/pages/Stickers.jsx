import { useEffect, useState } from 'react'
import { Lock } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Card, PageHeader, Progress } from '../ui'
import { Mascot } from '../mascot'
import { STICKERS, HATS, PERKS, RANKS, earned, hasHat, getHat, setHat, rank, perkOn, openAll, cozyDays } from '../rewards'
import { useCompanion } from '../companion'

export default function Trail() {
  const [got, setGot] = useState(earned())
  const [hat, setH] = useState(getHat())
  const { say } = useCompanion()
  useEffect(() => { const h = () => setGot(earned()); window.addEventListener('tb-earned', h); return () => window.removeEventListener('tb-earned', h) }, [])
  const count = STICKERS.filter((s) => got[s.id]).length
  const r = rank(count)
  const days = cozyDays()
  const pick = (id) => {
    if (!hasHat(id)) return
    setHat(id); setH(id)
    say(id === 'none' ? 'Back to my plain old self! 🚂' : 'Ooh, how do I look? 💜', id === 'none' ? 'happy' : 'love', 4500)
  }
  return (
    <div className="page">
      <PageHeader title="Boi's trail 🌟" sub="Little rewards for real things you do. Stickers are for celebrating; tricks are real new abilities that open up as you explore. No money, no pressure — just a path to wander." />

      <Card className="trail-rank">
        <div className="row" style={{ gap: 18, alignItems: 'center', flexWrap: 'wrap' }}>
          <div className="rank-badge">{r.emoji}</div>
          <div style={{ flex: 1, minWidth: 220 }}>
            <div className="rank-name">{r.name}</div>
            <div className="small muted">{r.next ? `${r.toNext} more sticker${r.toNext === 1 ? '' : 's'} until you're a ${r.next.name} ${r.next.emoji}` : 'You\'ve reached the top of the trail — thank you for being here. 💜'}</div>
            <div style={{ marginTop: 8 }}><Progress value={r.next ? (count - r.at) / (r.next.at - r.at) : 1} /></div>
            <div className="rank-steps">{RANKS.map((x) => <span key={x.name} className={count >= x.at ? 'on' : ''} title={`${x.name} · ${x.at} stickers`}>{x.emoji}</span>)}</div>
          </div>
          <div className="cozy-days"><b>{days}</b><span>cozy day{days === 1 ? '' : 's'} together</span><small>It only goes up. 🌿</small></div>
        </div>
      </Card>

      <Card title="Boi's tricks ✨" className="trail-perks" action={openAll() ? <span className="small muted">Everything is open</span> : null}>
        <div className="perk-list">
          {PERKS.map((p) => {
            const ok = perkOn(p.id)
            const need = STICKERS.find((s) => s.id === p.needs)
            return (
              <div key={p.id} className={`perk ${ok ? 'on' : 'locked'}`}>
                <div className="perk-ico">{ok ? p.emoji : <Lock size={18} />}</div>
                <div>
                  <b>{p.title}</b> <span className="faint small">· {p.where}</span>
                  <div className="small">{p.what}</div>
                  {!ok && <div className="small perk-need">Unlocks with the {need?.emoji} <b>{need?.title}</b> sticker — {need?.how.toLowerCase()}.</div>}
                </div>
              </div>
            )
          })}
        </div>
        <div className="small faint" style={{ marginTop: 12 }}>Already know your way around? Switch on <Link to="/settings" style={{ textDecoration: 'underline' }}>“Open everything”</Link> in Care &amp; comfort and every trick is yours straight away.</div>
      </Card>

      <div className="grid" style={{ gridTemplateColumns: '1.6fr 1fr', alignItems: 'start' }}>
        <Card title={`Your stickers · ${count} of ${STICKERS.length}`}>
          <div className="sticker-sheet">
            {STICKERS.map((s, i) => {
              const ok = !!got[s.id]
              const unlocks = PERKS.filter((p) => p.needs === s.id)
              return (
                <div key={s.id} className={`sticker ${ok ? 'got' : 'locked'}`} style={{ '--tilt': `${((i * 37) % 9) - 4}deg` }} title={ok ? s.title : s.how}>
                  <div className="sticker-pic">{ok ? s.emoji : '❔'}</div>
                  <b>{ok ? s.title : '???'}</b>
                  <span>{ok ? (unlocks.length ? `Unlocked: ${unlocks.map((p) => p.title).join(', ')}` : s.hat ? `Unlocked: ${HATS.find((h) => h.id === s.hat)?.label}` : new Date(got[s.id]).toLocaleDateString()) : s.how}</span>
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

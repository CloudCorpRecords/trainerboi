import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { useHardware } from './api'
import { Mascot } from './mascot'
import { burst } from './effects'
import { toot, chime } from './sound'
import { pokeCount, earn, HATS } from './rewards'
import { X } from 'lucide-react'

const Ctx = createContext({ say: () => {}, celebrate: () => {} })
export const useCompanion = () => useContext(Ctx)

const TIPS = [
  'Tip: the temperature card tells you how I\'m feeling — I\'ll pause myself if I get too toasty. 🧊',
  'Little models learn best from lots of different examples. 🌈',
  'Finished a model? Open it and try the Share tab — a zip makes a lovely gift. 🎁',
  'The Math playground needs no data at all — just a little time. 🧮',
  'Everything here stays on this computer. Nothing is sent anywhere unless you press Share. 🔒',
  'You can switch between Cozy night and Soft day in the sidebar. 🌙☀️',
  'Click me! I like it. 💜',
  'A falling line in "How it\'s learning" means your model is getting better. 🌱',
]
const POKES = ['Toot toot! 🚂', 'Hehe, that tickles!', 'I\'m right here if you need me 💜', 'Chugga chugga! ✨', 'Want to teach something new?', 'You\'re doing great!']

export function CompanionProvider({ children }) {
  const { hw } = useHardware()
  const [bubble, setBubble] = useState(null)
  const [asleep, setAsleep] = useState(() => { try { return localStorage.getItem('tb-boi-sleep') === '1' } catch { return false } })
  const [anim, setAnim] = useState('')
  const timer = useRef(null)
  const bubbleRef = useRef(null)
  bubbleRef.current = bubble
  const asleepRef = useRef(asleep)
  asleepRef.current = asleep

  const say = useCallback((text, mood, ms = 7500) => {
    if (asleepRef.current) return
    clearTimeout(timer.current)
    setBubble({ text, mood, id: Math.random() })
    timer.current = setTimeout(() => setBubble(null), ms)
  }, [])
  const celebrate = useCallback((text = 'We did it! 🎉') => {
    chime(); say(text, 'cheer', 9000)
    setAnim('hop'); setTimeout(() => setAnim(''), 1600)
  }, [say])

  // new stickers
  useEffect(() => {
    const h = (e) => {
      const s = e.detail
      if (!s) return
      const hat = s.hat && HATS.find((x) => x.id === s.hat)
      chime(); burst(window.innerWidth - 90, window.innerHeight - 100, ['⭐', '✨', '🌟'], 10)
      say(`New sticker: ${s.emoji} ${s.title}!${hat ? ` I unlocked the ${hat.label.toLowerCase()} ${hat.emoji} — try it in the Sticker book.` : ''}`, 'cheer', 11000)
      setAnim('hop'); setTimeout(() => setAnim(''), 1200)
    }
    window.addEventListener('tb-earned', h)
    return () => window.removeEventListener('tb-earned', h)
  }, [say])

  // gentle tips now and then when nothing else is happening
  useEffect(() => {
    const t = setInterval(() => { if (!bubbleRef.current && !asleepRef.current) say(TIPS[Math.floor(Math.random() * TIPS.length)], 'happy', 9000) }, 55000)
    const first = setTimeout(() => say('Hi! I\'m Boi. I\'ll cheer you on while we teach little models. 💜', 'happy', 9000), 2500)
    return () => { clearInterval(t); clearTimeout(first) }
  }, [say])

  // react to what the computer is doing
  const job = hw?.jobs?.[0]
  const temp = hw?.gpu?.temp
  const prevJobs = useRef(0), wasHot = useRef(false)
  useEffect(() => {
    const n = hw?.jobs?.filter((j) => j.kind !== 'data').length ?? 0
    if (hw && n > prevJobs.current) say('Off we go! 🚂 I\'ll keep you posted.', 'working')
    if (hw && n < prevJobs.current) celebrate('All done! That was fun 🎉')
    if (hw) prevJobs.current = n
  }, [hw?.jobs?.length])
  useEffect(() => {
    const hot = temp != null && temp >= 80
    if (hot && !wasHot.current) say('Phew, getting toasty! 🔥 I\'ll take a little breather if it gets too warm.', 'toasty')
    if (!hot && wasHot.current && temp != null && temp < 72) say('Ahh, all cooled down. Back to work! 🧊', 'happy', 5000)
    wasHot.current = hot
  }, [temp != null && temp >= 80, temp != null && temp < 72])

  const hour = new Date().getHours()
  const base = job ? (job.kind === 'cooling' || temp >= 80 ? 'toasty' : 'working') : (hour >= 23 || hour < 5 ? 'sleepy' : 'happy')
  const mood = asleep ? 'sleepy' : bubble?.mood || base

  const poke = (e) => {
    if (asleep) { setAsleep(false); try { localStorage.setItem('tb-boi-sleep', '0') } catch { /* ignore */ } say('Mmm… oh! Good to see you. 💜', 'happy'); return }
    toot(); burst(e.clientX, e.clientY)
    if (pokeCount() >= 10) earn('poked')
    setAnim('hop'); setTimeout(() => setAnim(''), 800)
    say(POKES[Math.floor(Math.random() * POKES.length)], 'happy', 4000)
  }
  const spin = () => { setAnim('spin'); setTimeout(() => setAnim(''), 900); burst(window.innerWidth - 80, window.innerHeight - 90, ['🌸', '✨'], 10) }
  const sleep = (e) => {
    e.stopPropagation()
    const v = !asleep
    setAsleep(v); try { localStorage.setItem('tb-boi-sleep', v ? '1' : '0') } catch { /* ignore */ }
    setBubble(null)
  }

  return (
    <Ctx.Provider value={{ say, celebrate }}>
      {children}
      <div className={`companion ${asleep ? 'asleep' : ''}`}>
        {bubble && !asleep && (
          <div className="speech" key={bubble.id}>
            {bubble.text}
            <button onClick={() => setBubble(null)} aria-label="Dismiss"><X size={12} /></button>
          </div>
        )}
        <button className={`boi-btn ${anim}`} onClick={poke} onDoubleClick={spin} title={asleep ? 'Wake Boi up' : 'Click me!'} aria-label="Boi, your companion">
          <Mascot mood={mood} size={74} />
        </button>
        <button className="boi-sleep" onClick={sleep} title={asleep ? 'Wake Boi' : 'Let Boi nap (hides tips)'}>{asleep ? '☀️' : '💤'}</button>
      </div>
    </Ctx.Provider>
  )
}

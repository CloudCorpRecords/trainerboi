import { useEffect, useRef, useState, useCallback, createContext, useContext } from 'react'

export async function j(url, opts) {
  const r = await fetch(url, opts)
  if (!r.ok) {
    let m
    try { m = (await r.json()).detail } catch { /* ignore */ }
    throw new Error(m || r.statusText || 'Request failed')
  }
  return r.json()
}
export const post = (u, b) => j(u, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b || {}) })

/** Poll a URL every `ms`. Keeps the last good data if a request fails. */
export function usePoll(url, ms = 2000, enabled = true) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const alive = useRef(true)
  const load = useCallback(async () => {
    if (!url) return
    try { const d = await j(url); if (alive.current) { setData(d); setError(null) } }
    catch (e) { if (alive.current) setError(e.message) }
  }, [url])
  useEffect(() => {
    alive.current = true
    if (!enabled || !url) return
    load()
    const t = setInterval(load, ms)
    return () => { alive.current = false; clearInterval(t) }
  }, [url, ms, enabled, load])
  return { data, error, reload: load }
}

/** Run a background job on the server and follow it until it finishes. */
export function useJob() {
  const [job, setJob] = useState(null)
  const timer = useRef(null)
  const follow = useCallback((jid) => {
    clearInterval(timer.current)
    setJob({ id: jid, state: 'running', log: [] })
    timer.current = setInterval(async () => {
      try {
        const s = await j(`/api/jobs/${jid}`)
        setJob({ id: jid, ...s })
        if (s.state !== 'running') clearInterval(timer.current)
      } catch { /* keep trying */ }
    }, 1000)
  }, [])
  useEffect(() => () => clearInterval(timer.current), [])
  return [job, follow, () => setJob(null)]
}

/** POST and read a server-sent-event stream, calling onEvent for every JSON message. */
export async function sse(url, body, onEvent, signal) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal })
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText)
  const reader = r.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let i
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const line = buf.slice(0, i).trim()
      buf = buf.slice(i + 2)
      if (line.startsWith('data:')) { try { onEvent(JSON.parse(line.slice(5))) } catch { /* ignore */ } }
    }
  }
}

// ---------------------------------------------------------------- live hardware (shared by every page)
const HwContext = createContext({ hw: null, history: { util: [], temp: [], vram: [], power: [] } })
export const useHardware = () => useContext(HwContext)

export function HardwareProvider({ children }) {
  const [hw, setHw] = useState(null)
  const [history, setHistory] = useState({ util: [], temp: [], vram: [], power: [] })
  useEffect(() => {
    let stop = false
    const tick = async () => {
      try {
        const d = await j('/api/hardware')
        if (stop) return
        setHw(d)
        setHistory((h) => {
          const push = (arr, v) => [...arr, v ?? null].slice(-90)
          return { util: push(h.util, d.gpu.util), temp: push(h.temp, d.gpu.temp), vram: push(h.vram, d.gpu.vram_used_gb), power: push(h.power, d.gpu.power) }
        })
      } catch { /* server restarting */ }
    }
    tick()
    const t = setInterval(tick, 1500)
    return () => { stop = true; clearInterval(t) }
  }, [])
  return <HwContext.Provider value={{ hw, history }}>{children}</HwContext.Provider>
}

// ---------------------------------------------------------------- toasts
export const toast = (msg, kind = 'info') => window.dispatchEvent(new CustomEvent('tb-toast', { detail: { msg, kind, id: Math.random() } }))

export const fmt = {
  dur(s) {
    if (s == null || !isFinite(s)) return '–'
    s = Math.round(s)
    if (s < 90) return `${s}s`
    if (s < 5400) return `${Math.round(s / 60)} min`
    return `${(s / 3600).toFixed(1)} h`
  },
  ago(ts) {
    const s = Date.now() / 1000 - ts
    if (s < 90) return 'just now'
    if (s < 5400) return `${Math.round(s / 60)} min ago`
    if (s < 172800) return `${Math.round(s / 3600)} h ago`
    return new Date(ts * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  },
  mb(m) { return m >= 1024 ? `${(m / 1024).toFixed(1)} GB` : `${Math.round(m)} MB` },
  n(x, d = 0) { return x == null ? '–' : Number(x).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }) },
}

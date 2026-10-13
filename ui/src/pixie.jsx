import { useEffect, useRef } from 'react'

export const dustOn = () => { try { return localStorage.getItem('tb-dust') !== '0' } catch { return true } }
export const setDust = (v) => { try { localStorage.setItem('tb-dust', v ? '1' : '0') } catch { /* ignore */ } }

/** A faint trail of pastel sparkles that follows the pointer. Only draws while there is something to draw. */
export function PixieDust() {
  const ref = useRef(null)
  useEffect(() => {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return
    const c = ref.current
    const ctx = c.getContext('2d')
    let w = 0, h = 0, raf = 0, last = 0
    const parts = []
    const colors = ['#ffc0dd', '#d3c6ff', '#ffe29a', '#b8f0dc', '#bfe6ff']
    const resize = () => {
      const dpr = window.devicePixelRatio || 1
      w = window.innerWidth; h = window.innerHeight
      c.width = w * dpr; c.height = h * dpr; c.style.width = `${w}px`; c.style.height = `${h}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    const star = (x, y, r) => {
      ctx.beginPath()
      for (let i = 0; i < 8; i++) { const a = (Math.PI / 4) * i, rr = i % 2 ? r * 0.4 : r; ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr) }
      ctx.closePath(); ctx.fill()
    }
    const draw = () => {
      ctx.clearRect(0, 0, w, h)
      for (let i = parts.length - 1; i >= 0; i--) {
        const p = parts[i]
        p.life -= 0.024
        if (p.life <= 0) { parts.splice(i, 1); continue }
        p.x += p.vx; p.y += p.vy; p.vy += 0.014; p.rot += 0.04
        ctx.globalAlpha = p.life * 0.85; ctx.fillStyle = p.c
        if (p.star) star(p.x, p.y, p.s * 1.6); else { ctx.beginPath(); ctx.arc(p.x, p.y, p.s * 0.7, 0, 7); ctx.fill() }
      }
      ctx.globalAlpha = 1
      raf = parts.length ? requestAnimationFrame(draw) : 0
      if (!parts.length) ctx.clearRect(0, 0, w, h)
    }
    const move = (e) => {
      if (!dustOn()) return
      const t = performance.now()
      if (t - last < 30) return
      last = t
      parts.push({ x: e.clientX + (Math.random() - 0.5) * 8, y: e.clientY + (Math.random() - 0.5) * 8, vx: (Math.random() - 0.5) * 0.9, vy: Math.random() * 0.5 + 0.15,
        life: 1, s: 2 + Math.random() * 3, c: colors[(Math.random() * colors.length) | 0], star: Math.random() < 0.45, rot: 0 })
      if (parts.length > 70) parts.shift()
      if (!raf) raf = requestAnimationFrame(draw)
    }
    resize()
    window.addEventListener('resize', resize)
    window.addEventListener('mousemove', move, { passive: true })
    return () => { window.removeEventListener('resize', resize); window.removeEventListener('mousemove', move); cancelAnimationFrame(raf) }
  }, [])
  return <canvas ref={ref} className="pixie" aria-hidden="true" />
}

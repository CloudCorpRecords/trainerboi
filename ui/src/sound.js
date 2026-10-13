// Tiny, gentle sounds made with the browser's built-in audio (no files). Off by default; the sidebar bell turns them on.
let ctx
const enabled = () => { try { return localStorage.getItem('tb-sound') === '1' } catch { return false } }
export const soundOn = enabled
export const setSound = (v) => { try { localStorage.setItem('tb-sound', v ? '1' : '0') } catch { /* ignore */ } }

function tone(freq, start, dur, vol = 0.05, type = 'sine') {
  try {
    ctx = ctx || new (window.AudioContext || window.webkitAudioContext)()
    const t0 = ctx.currentTime + start
    const o = ctx.createOscillator(), g = ctx.createGain()
    o.type = type; o.frequency.setValueAtTime(freq, t0)
    g.gain.setValueAtTime(0.0001, t0); g.gain.exponentialRampToValueAtTime(vol, t0 + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur)
    o.connect(g); g.connect(ctx.destination); o.start(t0); o.stop(t0 + dur + 0.05)
  } catch { /* audio not available */ }
}

export const toot = () => { if (!enabled()) return; tone(523, 0, 0.13, 0.05, 'triangle'); tone(784, 0.11, 0.2, 0.05, 'triangle') }
export const chime = () => { if (!enabled()) return; [523, 659, 784, 1047].forEach((f, i) => tone(f, i * 0.13, 0.4, 0.045)) }
export const pop = () => { if (!enabled()) return; tone(880, 0, 0.08, 0.03) }

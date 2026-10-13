// Little decorative effects drawn straight into the page (no React state, auto-cleaned).
export function burst(x, y, glyphs = ['✨', '💜', '🌸', '⭐'], n = 7) {
  for (let i = 0; i < n; i++) {
    const s = document.createElement('span')
    s.className = 'burst-bit'
    s.textContent = glyphs[i % glyphs.length]
    const a = (Math.PI * 2 * i) / n + Math.random() * 0.6
    const d = 36 + Math.random() * 38
    s.style.cssText = `left:${x}px;top:${y}px;--dx:${Math.cos(a) * d}px;--dy:${Math.sin(a) * d - 18}px;font-size:${12 + Math.random() * 9}px`
    document.body.appendChild(s)
    setTimeout(() => s.remove(), 1000)
  }
}

// Make every primary button and big call-to-action sparkle when clicked.
export function installSparkles() {
  const h = (e) => { if (e.target.closest?.('.btn-primary, .action.primary')) burst(e.clientX, e.clientY) }
  document.addEventListener('click', h)
  return () => document.removeEventListener('click', h)
}

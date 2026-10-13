// A cute one-page "birth certificate" for a model - a single self-contained .html file (nothing is uploaded anywhere).
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))

const KIND = { sft: 'a chat model', dpo: 'a polite chat model', orpo: 'a polite chat model', kto: 'a polite chat model', grpo: 'a careful thinker', pretrain: 'a model grown from scratch', math: 'a maths whiz', embedding: 'a text finder' }

export function certificateHtml(rec, parent = '') {
  const ts = rec.modified || rec.updated || rec.mtime
  const born = ts ? new Date(ts > 1e12 ? ts : ts * 1000) : new Date()
  const date = born.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })
  const base = rec.base ? rec.base.split('/').pop() : null
  const rows = [
    ['Kind', KIND[rec.kind] || rec.kind], base && ['Grew up from', base], rec.adapter && ['Learned with', 'a small LoRA adapter'],
    rec.metric && ['Report card', rec.metric], rec.size_mb != null && ['Weighs', `${Math.round(rec.size_mb)} MB`],
  ].filter(Boolean)
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Birth certificate - ${esc(rec.title)}</title>
<style>
  *{box-sizing:border-box} body{margin:0;min-height:100vh;display:grid;place-items:center;padding:28px;background:#f6f0ff;font-family:'Nunito','Segoe UI',system-ui,sans-serif;color:#33295f}
  .paper{position:relative;max-width:680px;width:100%;padding:44px 40px 36px;text-align:center;background:#fffdf8;border-radius:28px;border:3px dashed #b9a5ff;box-shadow:0 18px 50px rgba(120,90,200,.22)}
  .paper:before,.paper:after{content:'\\2728';position:absolute;font-size:28px} .paper:before{top:16px;left:20px} .paper:after{bottom:16px;right:20px}
  .train{font-size:64px;line-height:1} h1{margin:6px 0 0;font-size:15px;letter-spacing:.2em;text-transform:uppercase;color:#8d6df2}
  h2{margin:10px 0 4px;font-size:38px;line-height:1.1} .sub{color:#7a70a8;font-size:15px}
  table{margin:24px auto 8px;border-collapse:collapse;text-align:left;font-size:15px} td{padding:6px 14px;border-bottom:1px dotted #d6c8ff} td:first-child{color:#8a80b8;font-weight:700}
  .foot{margin-top:22px;font-size:14px;color:#7a70a8} .sign{margin-top:18px;font-family:'Segoe Script','Comic Sans MS',cursive;font-size:22px;color:#8d6df2}
  @media print{body{background:#fff;padding:0}.paper{box-shadow:none}}
</style></head><body><div class="paper">
<div class="train">🚂</div><h1>Certificate of Birth</h1>
<h2>${esc(rec.title)}</h2>
<div class="sub">was taught on ${esc(date)}${parent ? ` by ${esc(parent)}` : ''}</div>
<table>${rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join('')}</table>
${rec.notes ? `<div class="foot">“${esc(rec.notes)}”</div>` : ''}
<div class="sign">${parent ? esc(parent) + ' &amp; ' : ''}TrainerBoi 💜</div>
<div class="foot">Made on a home computer with TrainerBoi - free, open-source, and kind to your GPU.</div>
</div></body></html>`
}

export function downloadCertificate(rec, parent = '') {
  const blob = new Blob([certificateHtml(rec, parent)], { type: 'text/html' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `${(rec.title || 'model').replace(/[^\w.-]+/g, '-')}-birth-certificate.html`
  document.body.appendChild(a); a.click(); a.remove()
  setTimeout(() => URL.revokeObjectURL(a.href), 2000)
}

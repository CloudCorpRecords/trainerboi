const $ = (id) => document.getElementById(id);
let steps = [];
function render(active, done) {
  $('steps').innerHTML = steps.map((s, i) => {
    const cls = i < active || done ? 'done' : i === active ? 'active' : '';
    const dot = i < active || done ? '✓' : i === active ? '⏳' : '○';
    return `<li class="${cls}"><span class="dot">${dot}</span>${s}</li>`;
  }).join('');
  $('bar').style.width = (done ? 100 : Math.round((active / steps.length) * 100)) + '%';
}
window.tb.on('setup-gpu', (t) => { $('gpu').textContent = '🖥️ ' + t; });
window.tb.on('setup-steps', (s) => { steps = s; render(0, false); });
window.tb.on('setup-step', (i) => render(i, false));
window.tb.on('setup-done', () => render(steps.length, true));
window.tb.on('setup-log', (t) => { const l = $('log'); l.textContent = (l.textContent + t).slice(-20000); l.scrollTop = l.scrollHeight; });
$('logLink').onclick = () => window.tb.openLog();

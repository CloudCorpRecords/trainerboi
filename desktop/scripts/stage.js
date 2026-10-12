// Copies the Python app + uv into desktop/stage so electron-builder can ship them as extraResources.
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const root = path.resolve(__dirname, '..', '..');
const stage = path.resolve(__dirname, '..', 'stage');
fs.rmSync(stage, { recursive: true, force: true });
fs.mkdirSync(path.join(stage, 'app-src'), { recursive: true });
fs.mkdirSync(path.join(stage, 'bin'), { recursive: true });

const keepExt = new Set(['.py', '.css', '.txt', '.md', '.json']);
const skipFiles = new Set(['gpu_health.json']);
for (const f of fs.readdirSync(root)) {
  const p = path.join(root, f);
  if (fs.statSync(p).isFile() && keepExt.has(path.extname(f)) && !skipFiles.has(f) && !(f.startsWith('model_') && ['.pt', '.json'].includes(path.extname(f)))) {
    fs.copyFileSync(p, path.join(stage, 'app-src', f));
  }
}
// small sample data used by the tabular demo
const samples = path.join(root, 'data', 'samples');
if (fs.existsSync(samples)) fs.cpSync(samples, path.join(stage, 'app-src', 'data', 'samples'), { recursive: true });

// the React UI (build it first: cd ui && npm run build)
const uiDist = path.join(root, 'ui', 'dist');
if (!fs.existsSync(path.join(uiDist, 'index.html'))) { console.error('ui/dist is missing. Run: cd ui && npm install && npm run build'); process.exit(1); }
fs.cpSync(uiDist, path.join(stage, 'app-src', 'ui', 'dist'), { recursive: true });

let uv = process.env.UV_PATH;
if (!uv) {
  try { uv = execSync('where uv', { encoding: 'utf8' }).split(/\r?\n/)[0].trim(); } catch (e) { /* not on PATH */ }
}
if (!uv || !fs.existsSync(uv)) {
  console.error('uv.exe not found. Install uv (https://docs.astral.sh/uv/) or set UV_PATH.');
  process.exit(1);
}
fs.copyFileSync(uv, path.join(stage, 'bin', 'uv.exe'));
console.log('staged app-src and uv from', uv);

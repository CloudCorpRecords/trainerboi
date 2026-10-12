// TrainerBoi desktop shell: first-run setup (Python + PyTorch for your GPU), then runs the Studio in a window.
const { app, BrowserWindow, Menu, shell, dialog, ipcMain, session } = require('electron');
const { spawn, execFile, execFileSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');
const net = require('net');

const DEV_ROOT = path.resolve(__dirname, '..');                       // repo checkout (dev mode)
const IS_DEV = !app.isPackaged;
const SRC_DIR = IS_DEV ? DEV_ROOT : path.join(process.resourcesPath, 'app-src');
const UV = IS_DEV && !fs.existsSync(path.join(DEV_ROOT, 'desktop', 'stage', 'bin', 'uv.exe'))
  ? 'uv' : path.join(IS_DEV ? path.join(DEV_ROOT, 'desktop', 'stage') : process.resourcesPath, 'bin', 'uv.exe');
const DATA = app.getPath('userData');                                 // %APPDATA%/TrainerBoi
const ENV_DIR = path.join(DATA, 'env');
const WORK_DIR = path.join(DATA, 'workspace');                        // code copy + runs/ + data/ live here
const READY = path.join(ENV_DIR, '.ready.json');
const SETUP_VERSION = 4;                                              // bump when requirements change
const LOG = path.join(DATA, 'desktop.log');

let win = null, setupWin = null, backend = null, port = 0, quitting = false;

function log(...a) {
  const line = `[${new Date().toISOString()}] ${a.join(' ')}\n`;
  try { fs.mkdirSync(DATA, { recursive: true }); fs.appendFileSync(LOG, line); } catch (e) { /* ignore */ }
  console.log(...a);
}

const pyExe = () => path.join(ENV_DIR, 'Scripts', 'python.exe');
const devVenvPy = () => path.join(DEV_ROOT, '.venv', 'Scripts', 'python.exe');
// In dev we reuse the project's existing .venv so nothing has to be reinstalled.
const useDevVenv = () => IS_DEV && fs.existsSync(devVenvPy()) && !process.env.TB_FORCE_SETUP;
const pythonPath = () => (useDevVenv() ? devVenvPy() : pyExe());
const appDir = () => (IS_DEV ? DEV_ROOT : WORK_DIR);

// ---------------------------------------------------------------- helpers
function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => resolve(p)); });
    s.on('error', reject);
  });
}

function waitForServer(p, timeoutMs = 180000) {
  const t0 = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get({ host: '127.0.0.1', port: p, path: '/', timeout: 2000 }, (res) => { res.resume(); resolve(); });
      req.on('error', () => (Date.now() - t0 > timeoutMs ? reject(new Error('Studio did not start in time')) : setTimeout(tick, 700)));
      req.on('timeout', () => req.destroy());
    };
    tick();
  });
}

function detectGpu() {
  // Returns {name, driver, cuda} or null. Uses nvidia-smi, which ships with the NVIDIA driver.
  try {
    const out = execFileSync('nvidia-smi', ['--query-gpu=name,driver_version', '--format=csv,noheader'],
      { encoding: 'utf8', timeout: 8000, windowsHide: true }).trim().split(/\r?\n/)[0];
    const [name, driver] = out.split(',').map((s) => s.trim());
    const major = parseInt(driver, 10);
    // PyTorch wheels: cu130 needs driver >= 580, cu128 >= 570, cu126 >= 560.
    const cuda = major >= 580 ? 'cu130' : major >= 570 ? 'cu128' : major >= 560 ? 'cu126' : null;
    return { name, driver, cuda };
  } catch (e) { return null; }
}

function send(channel, payload) { if (setupWin && !setupWin.isDestroyed()) setupWin.webContents.send(channel, payload); }

function run(cmd, args, label, opts = {}) {
  return new Promise((resolve, reject) => {
    log('RUN', cmd, args.join(' '));
    const child = spawn(cmd, args, { cwd: opts.cwd || DATA, windowsHide: true,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', UV_LINK_MODE: 'copy', UV_PYTHON_INSTALL_DIR: path.join(DATA, 'python') } });
    const onData = (d) => { const s = d.toString(); log(s.trimEnd()); send('setup-log', s); };
    child.stdout.on('data', onData); child.stderr.on('data', onData);
    child.on('error', reject);
    child.on('close', (code) => (code === 0 || opts.allowFail ? resolve(code) : reject(new Error(`${label} failed (exit ${code}). See ${LOG}`))));
  });
}

function syncApp() {
  // Packaged mode: copy the Python sources into a writable workspace (runs/ and data/ are created next to them).
  if (IS_DEV) return;
  fs.mkdirSync(WORK_DIR, { recursive: true });
  fs.cpSync(SRC_DIR, WORK_DIR, { recursive: true, force: true });
}

// ---------------------------------------------------------------- first-run setup
function isReady() {
  if (useDevVenv()) return true;
  try { return JSON.parse(fs.readFileSync(READY, 'utf8')).version === SETUP_VERSION && fs.existsSync(pyExe()); } catch (e) { return false; }
}

async function setup() {
  setupWin = new BrowserWindow({ width: 760, height: 620, resizable: false, title: 'TrainerBoi setup', autoHideMenuBar: true,
    backgroundColor: '#0b0f19', webPreferences: { preload: path.join(__dirname, 'preload.js') } });
  setupWin.loadFile(path.join(__dirname, 'setup.html'));
  await new Promise((r) => setupWin.webContents.once('did-finish-load', r));

  const gpu = detectGpu();
  send('setup-gpu', gpu ? `${gpu.name} (driver ${gpu.driver})${gpu.cuda ? '' : ' - driver is old, using CPU build'}` : 'No NVIDIA GPU found - using the CPU build (small models only)');
  const steps = ['Prepare Python', 'Install PyTorch', 'Install TrainerBoi libraries', 'GPU speed-ups', 'Finish'];
  send('setup-steps', steps);

  fs.mkdirSync(DATA, { recursive: true });
  const step = (i) => send('setup-step', i);

  step(0);
  syncApp();
  await run(UV, ['venv', ENV_DIR, '--python', '3.11', '--clear'], 'Python setup');
  const py = pyExe();

  step(1);
  const index = gpu && gpu.cuda ? `https://download.pytorch.org/whl/${gpu.cuda}` : 'https://download.pytorch.org/whl/cpu';
  await run(UV, ['pip', 'install', '--python', py, 'torch', 'torchvision', 'torchaudio', '--index-url', index], 'PyTorch install');

  step(2);
  await run(UV, ['pip', 'install', '--python', py, '-r', path.join(SRC_DIR, 'requirements-app.txt')], 'Library install');

  step(3);
  if (gpu && gpu.cuda && process.platform === 'win32') {
    // Optional: failure here must not block the app (Unsloth/bitsandbytes are only speed-ups).
    await run(UV, ['pip', 'install', '--python', py, '-r', path.join(SRC_DIR, 'requirements-nvidia.txt')], 'GPU extras', { allowFail: true });
    await run(UV, ['pip', 'install', '--python', py, '--no-deps', 'liger-kernel==0.8.4'], 'Liger kernel', { allowFail: true });
  }

  step(4);
  fs.writeFileSync(READY, JSON.stringify({ version: SETUP_VERSION, gpu, date: new Date().toISOString() }));
  send('setup-done', true);
  await new Promise((r) => setTimeout(r, 800));
  setupWin.close(); setupWin = null;
}

// ---------------------------------------------------------------- backend
async function startBackend() {
  port = await freePort();
  const env = { ...process.env, TB_PORT: String(port), TB_NO_BROWSER: '1', PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1',
    PYTHONWARNINGS: 'ignore', HF_HUB_DISABLE_TELEMETRY: '1', GRADIO_ANALYTICS_ENABLED: 'False',
    HF_HOME: process.env.HF_HOME || path.join(DATA, 'hf-cache') };
  log('Starting Studio on port', port, 'python', pythonPath());
  const out = fs.openSync(path.join(DATA, 'studio.log'), 'w');
  backend = spawn(pythonPath(), ['-u', 'studio_server.py', '--port', String(port)], { cwd: appDir(), env, windowsHide: true, stdio: ['ignore', out, out] });
  backend.on('exit', (code) => { log('Studio exited', code); backend = null; if (!quitting && win) showCrash(code); });
  await waitForServer(port);
}

function stopBackend() {
  if (!backend) return;
  try { execFile('taskkill', ['/PID', String(backend.pid), '/T', '/F'], { windowsHide: true }); } catch (e) { /* ignore */ }
  backend = null;
}

function showCrash(code) {
  const r = dialog.showMessageBoxSync(win, { type: 'error', buttons: ['Restart', 'Open log', 'Quit'], defaultId: 0,
    message: 'The TrainerBoi engine stopped unexpectedly.', detail: `Exit code ${code}. Your runs and datasets are safe.` });
  if (r === 0) restart(); else if (r === 1) { shell.showItemInFolder(path.join(DATA, 'studio.log')); } else app.quit();
}

async function restart() {
  stopBackend();
  await startBackend();
  if (win) win.loadURL(`http://127.0.0.1:${port}`);
}

// ---------------------------------------------------------------- window / menu
function createWindow() {
  win = new BrowserWindow({ width: 1440, height: 920, minWidth: 1040, minHeight: 640, title: 'TrainerBoi', backgroundColor: '#0b0c10',
    icon: path.join(__dirname, 'assets', 'icon.png'), autoHideMenuBar: true,
    titleBarStyle: 'hidden', titleBarOverlay: { color: '#0b0c10', symbolColor: '#8e93a1', height: 38 },
    webPreferences: { contextIsolation: true, nodeIntegration: false } });
  win.setMenuBarVisibility(false);   // our own title bar replaces it; shortcuts still work (Ctrl+R, F12, Ctrl+/-)
  win.loadURL(`http://127.0.0.1:${port}`);
  win.webContents.setWindowOpenHandler(({ url }) => { if (/^https?:/.test(url) && !url.includes(`127.0.0.1:${port}`)) shell.openExternal(url); return { action: 'deny' }; });
  // Never let the app window navigate away from our own local server; open real websites in the system browser instead.
  win.webContents.on('will-navigate', (e, url) => {
    if (!url.startsWith(`http://127.0.0.1:${port}`)) { e.preventDefault(); if (/^https?:/.test(url)) shell.openExternal(url); }
  });
  win.webContents.on('will-attach-webview', (e) => e.preventDefault());
  win.on('closed', () => { win = null; });
}

function buildMenu() {
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: 'TrainerBoi', submenu: [
      { label: 'Open my files (runs, datasets)', click: () => shell.openPath(appDir()) },
      { label: 'Show log', click: () => shell.showItemInFolder(path.join(DATA, 'studio.log')) },
      { label: 'Restart engine', click: () => restart() },
      { label: 'Repair installation…', click: async () => { try { fs.rmSync(READY, { force: true }); } catch (e) { /* ignore */ } app.relaunch(); app.quit(); } },
      { type: 'separator' }, { role: 'quit' } ] },
    { label: 'View', submenu: [{ role: 'reload' }, { role: 'togglefullscreen' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { role: 'resetZoom' }, { role: 'toggleDevTools' }] },
    { label: 'Help', submenu: [{ label: 'Hugging Face', click: () => shell.openExternal('https://huggingface.co') }] },
  ]));
}

// ---------------------------------------------------------------- auto-update (only when built with TB_UPDATE_URL)
function checkForUpdates() {
  if (IS_DEV || !fs.existsSync(path.join(process.resourcesPath, 'app-update.yml'))) return;
  try {
    const { autoUpdater } = require('electron-updater');
    autoUpdater.autoDownload = true;
    autoUpdater.on('error', (e) => log('Update check failed:', e && e.message));
    autoUpdater.on('update-downloaded', (info) => {
      const r = dialog.showMessageBoxSync(win, { type: 'info', buttons: ['Restart now', 'Later'], defaultId: 0,
        message: `TrainerBoi ${info.version} is ready to install.`, detail: 'Your runs and datasets are kept.' });
      if (r === 0) { quitting = true; stopBackend(); autoUpdater.quitAndInstall(); }
    });
    autoUpdater.checkForUpdates();
  } catch (e) { log('Updater unavailable:', e.message); }
}

// ---------------------------------------------------------------- lifecycle
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) app.quit();
app.on('second-instance', () => { if (win) { if (win.isMinimized()) win.restore(); win.focus(); } });

app.whenReady().then(async () => {
  try {
    // Deny every browser permission (camera, microphone, location, notifications…) - the app doesn't need any.
    session.defaultSession.setPermissionRequestHandler((_wc, _perm, cb) => cb(false));
    session.defaultSession.setPermissionCheckHandler(() => false);
    buildMenu();
    if (!isReady()) await setup();
    else syncApp();
    const splash = new BrowserWindow({ width: 420, height: 260, frame: false, resizable: false, backgroundColor: '#0b0c10', alwaysOnTop: true });
    splash.loadURL('data:text/html,' + encodeURIComponent(
      '<body style="margin:0;background:#0b0c10;color:#e5e7eb;font-family:Segoe UI,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;flex-direction:column">' +
      '<div style="font-size:34px">🚂</div><h2 style="margin:8px 0">TrainerBoi</h2><div style="opacity:.7">Starting the engine…</div></body>'));
    await startBackend();
    splash.close();
    createWindow();
    checkForUpdates();
  } catch (e) {
    log('FATAL', e.stack || e);
    dialog.showErrorBox('TrainerBoi could not start', `${e.message}\n\nLog: ${LOG}`);
    app.quit();
  }
});

app.on('before-quit', () => { quitting = true; stopBackend(); });
app.on('window-all-closed', () => { app.quit(); });
ipcMain.handle('open-log', () => shell.showItemInFolder(LOG));

// Release build: signed (if a certificate is provided) and with an auto-update feed.
//   set TB_UPDATE_URL=https://downloads.example.com/trainerboi     (folder that will host latest.yml + the installer)
//   set CSC_LINK=path\to\certificate.pfx  and  CSC_KEY_PASSWORD=...   (code-signing certificate; optional but recommended)
// Run from a machine where symlinks are allowed (Windows Developer Mode, or CI) so the signing tools can unpack.
const { execSync } = require('child_process');
const url = process.env.TB_UPDATE_URL;
const signing = !!process.env.CSC_LINK;
const args = ['electron-builder', '--win'];
if (url) args.push(`-c.publish.provider=generic`, `-c.publish.url=${url}`);
if (signing) args.push('-c.win.signAndEditExecutable=true');
else console.warn('No CSC_LINK set: building an UNSIGNED installer (SmartScreen will warn users).');
if (!url) console.warn('No TB_UPDATE_URL set: this build will not auto-update.');
execSync('npm --prefix ../ui install && npm --prefix ../ui run build', { stdio: 'inherit' });
execSync('node scripts/stage.js', { stdio: 'inherit' });
execSync('npx ' + args.join(' '), { stdio: 'inherit', env: { ...process.env, CSC_IDENTITY_AUTO_DISCOVERY: signing ? 'true' : 'false' } });

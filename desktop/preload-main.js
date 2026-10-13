// Lets the page tell the shell which theme is showing, so the native window buttons match.
const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('tbShell', { setTheme: (t) => ipcRenderer.send('set-theme', t) });

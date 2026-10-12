const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('tb', {
  on: (channel, cb) => ipcRenderer.on(channel, (_e, payload) => cb(payload)),
  openLog: () => ipcRenderer.invoke('open-log'),
});

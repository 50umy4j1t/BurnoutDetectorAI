const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('stressLensApi', {
  getBootstrap: () => ipcRenderer.invoke('app:getBootstrap'),
  startMonitoring: (modelChoice) => ipcRenderer.invoke('session:startMain', { modelChoice }),
  stopMonitoring: () => ipcRenderer.invoke('session:stopMain'),
  captureHeartRate: () => ipcRenderer.invoke('heart:capture'),
  setTtsEnabled: (enabled) => ipcRenderer.invoke('tts:setEnabled', { enabled }),
  speakText: (text, reason) => ipcRenderer.invoke('tts:speak', { text, reason }),
  sendChat: (message, includeLatestReport, modelChoice) => ipcRenderer.invoke('chat:send', {
    message,
    includeLatestReport,
    modelChoice,
  }),
  listReports: () => ipcRenderer.invoke('reports:list'),
  readReport: (fileName) => ipcRenderer.invoke('reports:read', { fileName }),
  getLogs: () => ipcRenderer.invoke('logs:get'),
  openReportFolder: () => ipcRenderer.invoke('shell:openReportFolder'),
  onBridgeEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('bridge-event', listener);
    return () => ipcRenderer.removeListener('bridge-event', listener);
  },
});

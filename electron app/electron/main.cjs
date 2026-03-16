const { app, BrowserWindow, ipcMain, shell } = require('electron');
const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { spawn } = require('child_process');

const electronAppRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(electronAppRoot, '..');
const bridgeScript = path.join(electronAppRoot, 'python', 'bridge.py');
const pythonFromVenv = path.join(repoRoot, '.venv', 'Scripts', 'python.exe');
const pythonExecutable = fs.existsSync(pythonFromVenv) ? pythonFromVenv : 'python';
const useDevServer = process.argv.includes('--dev');
const rendererEntry = path.join(electronAppRoot, 'dist', 'index.html');

let mainWindow = null;
let bridgeProcess = null;
let bridgeReady = false;
let nextRequestId = 1;
const pendingRequests = new Map();
const bridgeLogs = [];

function buildPythonEnv() {
  return {
    ...process.env,
    PYTHONUTF8: '1',
    PYTHONIOENCODING: 'utf-8',
    PYTHONUNBUFFERED: '1',
  };
}

function mirrorLogToTerminal(source, message) {
  const rendered = `[${source}] ${message}`;
  if (source.includes('stderr') || /failed|error/i.test(message)) {
    console.error(rendered);
    return;
  }
  console.log(rendered);
}

function appendLog(source, message) {
  const entry = {
    source,
    message,
    timestamp: new Date().toISOString(),
  };
  mirrorLogToTerminal(source, message);
  bridgeLogs.push(entry);
  if (bridgeLogs.length > 400) {
    bridgeLogs.shift();
  }
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('bridge-event', {
      event: 'log',
      data: entry,
    });
  }
}

function emitBridgeEvent(event, data) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('bridge-event', { event, data });
  }
}

function attachBridgeReaders(child) {
  const stdoutReader = readline.createInterface({ input: child.stdout });
  stdoutReader.on('line', (line) => {
    if (!line.trim()) {
      return;
    }

    let payload;
    try {
      payload = JSON.parse(line);
    } catch {
      appendLog('bridge:stdout', line);
      return;
    }

    if (payload.type === 'event') {
      if (payload.event === 'log' && payload.data) {
        appendLog(payload.data.source || 'bridge:event', payload.data.message || '');
      }
      emitBridgeEvent(payload.event, payload.data);
      return;
    }

    if (typeof payload.id === 'number' && pendingRequests.has(payload.id)) {
      const request = pendingRequests.get(payload.id);
      pendingRequests.delete(payload.id);
      if (payload.ok) {
        request.resolve(payload.result);
      } else {
        request.reject(new Error(payload.error || 'Bridge request failed'));
      }
    }
  });

  const stderrReader = readline.createInterface({ input: child.stderr });
  stderrReader.on('line', (line) => {
    if (line.trim()) {
      appendLog('bridge:stderr', line);
    }
  });
}

function startBridge() {
  if (bridgeProcess) {
    return;
  }

  bridgeProcess = spawn(pythonExecutable, ['-u', bridgeScript], {
    cwd: repoRoot,
    env: buildPythonEnv(),
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true,
  });
  bridgeReady = true;
  attachBridgeReaders(bridgeProcess);

  bridgeProcess.on('exit', (code) => {
    appendLog('bridge', `Bridge exited with code ${code}`);
    bridgeProcess = null;
    bridgeReady = false;
    for (const request of pendingRequests.values()) {
      request.reject(new Error('Bridge process exited'));
    }
    pendingRequests.clear();
    emitBridgeEvent('bridge-status', { running: false, exitCode: code });
  });

  bridgeProcess.on('error', (error) => {
    appendLog('bridge', `Bridge failed to start: ${error.message}`);
  });

  emitBridgeEvent('bridge-status', { running: true });
}

function callBridge(method, params = {}) {
  startBridge();
  if (!bridgeProcess || !bridgeReady) {
    return Promise.reject(new Error('Bridge is not available'));
  }

  const requestId = nextRequestId++;
  const request = { id: requestId, method, params };

  return new Promise((resolve, reject) => {
    pendingRequests.set(requestId, { resolve, reject });
    bridgeProcess.stdin.write(`${JSON.stringify(request)}\n`);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1520,
    height: 980,
    minWidth: 1280,
    minHeight: 820,
    backgroundColor: '#09111c',
    title: 'StressLens Electron',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.cjs'),
    },
  });

  if (useDevServer) {
    mainWindow.loadURL('http://localhost:5173');
  } else if (fs.existsSync(rendererEntry)) {
    mainWindow.loadFile(rendererEntry);
  } else {
    mainWindow.loadURL(
      `data:text/html,${encodeURIComponent(`
        <!doctype html>
        <html>
          <head>
            <meta charset="utf-8" />
            <title>StressLens Electron</title>
            <style>
              body {
                margin: 0;
                font-family: Segoe UI Variable Display, Bahnschrift, sans-serif;
                background: #09111c;
                color: #f4f7fb;
                display: grid;
                place-items: center;
                min-height: 100vh;
              }
              main {
                width: min(640px, calc(100vw - 48px));
                padding: 28px;
                border-radius: 18px;
                background: rgba(14, 24, 39, 0.92);
                border: 1px solid rgba(129, 164, 197, 0.25);
                box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
              }
              h1 {
                margin: 0 0 12px;
                font-size: 28px;
              }
              p {
                margin: 0 0 12px;
                line-height: 1.6;
                color: #d2deea;
              }
              code {
                display: block;
                margin-top: 12px;
                padding: 12px 14px;
                border-radius: 12px;
                background: #07111b;
                color: #8ed0ff;
                font-family: Cascadia Code, Consolas, monospace;
              }
            </style>
          </head>
          <body>
            <main>
              <h1>Renderer build not found</h1>
              <p>Run one of these commands from the electron app folder before starting Electron.</p>
              <code>npm install\nnpm run build\nnpm start</code>
              <p>For live development, run <strong>npm run dev</strong>.</p>
            </main>
          </body>
        </html>
      `)}`
    );
  }
}

ipcMain.handle('app:getBootstrap', async () => callBridge('bootstrap'));
ipcMain.handle('session:startMain', async (_event, payload) => callBridge('launch_main', payload));
ipcMain.handle('session:stopMain', async () => callBridge('terminate_main'));
ipcMain.handle('heart:capture', async () => callBridge('capture_heart_rate'));
ipcMain.handle('tts:setEnabled', async (_event, payload) => callBridge('set_tts_enabled', payload));
ipcMain.handle('chat:send', async (_event, payload) => callBridge('chat', payload));
ipcMain.handle('reports:list', async () => callBridge('list_reports'));
ipcMain.handle('reports:read', async (_event, payload) => callBridge('read_report', payload));
ipcMain.handle('logs:get', async () => ({ logs: bridgeLogs }));
ipcMain.handle('shell:openReportFolder', async () => shell.openPath(repoRoot));

app.whenReady().then(() => {
  startBridge();
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('before-quit', () => {
  if (bridgeProcess && bridgeReady) {
    try {
      bridgeProcess.stdin.write(`${JSON.stringify({ id: 0, method: 'shutdown', params: {} })}\n`);
    } catch {
      // ignore shutdown write errors during quit
    }
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

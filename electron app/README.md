# StressLens Electron App

This folder contains the isolated Electron sidecar for the existing Python burnout monitor.

What it does:
- Launches the unchanged main.py camera monitor in a separate process
- Lists and reads report_*.txt files from the repository root
- Lets you chat with the selected Ollama model from the desktop UI
- Captures heart-rate and SpO2 readings on demand through the serial sensor

What it does not do:
- It does not edit or replace main.py
- It does not embed the OpenCV camera feed inside Electron
- It does not auto-start any dev server unless you run the dev script yourself

Manual setup:

```powershell
cd "D:\deletelater\posterproject\electron app"
npm install
```

Run in development:

```powershell
npm run dev
```

Run a built desktop bundle locally:

```powershell
npm run build
npm start
```

Monitor controls:
- Use the Electron window to launch the monitor, chat, browse reports, and trigger a heart reading.
- Use the native camera window for the original keyboard controls: Q, R, P, and S.
- Press Q in the camera window when you want the original clean shutdown and final report flow.
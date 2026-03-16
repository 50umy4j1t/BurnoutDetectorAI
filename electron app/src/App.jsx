import { useEffect, useState } from 'react';

const EMOTION_COLORS = {
  happy: '#22c55e',
  sad: '#3b82f6',
  angry: '#ef4444',
  surprise: '#f59e0b',
  fear: '#a855f7',
  disgust: '#06b6d4',
  neutral: '#64748b',
};

const DEFAULT_TTS = {
  available: false,
  enabled: false,
  state: 'unavailable',
  message: 'Python TTS unavailable',
};

function timestampLabel(value) {
  if (!value) {
    return '--';
  }

  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

function extractLineValue(text, label) {
  if (!text) {
    return '--';
  }

  const line = text
    .split(/\r?\n/)
    .find((item) => item.includes(label));

  if (!line) {
    return '--';
  }

  const index = line.indexOf(label);
  return line.slice(index + label.length).trim() || '--';
}

function parseMetricNumber(value) {
  const match = String(value || '').match(/-?\d+(?:\.\d+)?/);
  return match ? Number(match[0]) : null;
}

function parseDurationToClock(value) {
  const match = String(value || '').match(/(\d+)m\s+(\d+)s/i);
  if (!match) {
    return '00:00';
  }

  const minutes = String(Number(match[1])).padStart(2, '0');
  const seconds = String(Number(match[2])).padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function parseEmotionBreakdown(text) {
  if (!text || !text.includes('AVERAGE EMOTION BREAKDOWN:')) {
    return [];
  }

  const sectionMatch = text.match(/AVERAGE EMOTION BREAKDOWN:\s*([\s\S]*?)^-{5,}/m);
  const section = sectionMatch ? sectionMatch[1] : '';

  return section
    .split(/\r?\n/)
    .map((line) => {
      const match = line.match(/^\s*([A-Za-z_]+)\s+(\d+(?:\.\d+)?)%/);
      if (!match) {
        return null;
      }

      const emotion = match[1].toLowerCase();
      const percentage = Number(match[2]);
      return {
        emotion,
        percentage,
        value: percentage / 100,
      };
    })
    .filter(Boolean);
}

function parseReport(text) {
  if (!text) {
    return {
      duration: '--',
      durationClock: '00:00',
      samples: '--',
      avgBurnout: '--',
      avgBurnoutValue: 0,
      peakBurnout: '--',
      peakBurnoutValue: 0,
      status: 'No report selected',
      heartRate: '--',
      spo2: '--',
      advice: '',
      emotions: [],
    };
  }

  const lines = text.split(/\r?\n/);
  const statusLine = lines.find((item) => item.includes('STATUS:'));
  const adviceMarker = 'AI WELLNESS ADVISOR RESPONSE:';
  const adviceIndex = text.indexOf(adviceMarker);
  const avgBurnout = extractLineValue(text, 'AVG BURNOUT SCORE :');
  const peakBurnout = extractLineValue(text, 'PEAK BURNOUT      :');
  const duration = extractLineValue(text, 'Duration      :');

  return {
    duration,
    durationClock: parseDurationToClock(duration),
    samples: extractLineValue(text, 'Samples       :'),
    avgBurnout,
    avgBurnoutValue: parseMetricNumber(avgBurnout) ?? 0,
    peakBurnout,
    peakBurnoutValue: parseMetricNumber(peakBurnout) ?? 0,
    status: statusLine ? statusLine.split('STATUS:')[1].trim() : 'Report has no risk summary yet',
    heartRate: extractLineValue(text, 'HEART RATE (BPM)  :'),
    spo2: extractLineValue(text, 'BLOOD OXYGEN      :'),
    advice: adviceIndex >= 0 ? text.slice(adviceIndex + adviceMarker.length).trim() : '',
    emotions: parseEmotionBreakdown(text),
  };
}

function statusTone(status) {
  const upper = String(status || '').toUpperCase();
  if (upper.includes('CRITICAL')) {
    return 'critical';
  }
  if (upper.includes('HIGH')) {
    return 'high';
  }
  if (upper.includes('MODERATE')) {
    return 'moderate';
  }
  if (upper.includes('LOW')) {
    return 'low';
  }
  return 'neutral';
}

function toneColor(tone) {
  if (tone === 'critical') {
    return '#ef4444';
  }
  if (tone === 'high' || tone === 'moderate') {
    return '#f59e0b';
  }
  if (tone === 'low') {
    return '#22c55e';
  }
  return '#94a3b8';
}

function createChatEntry(role, content, extras = {}) {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    role,
    content,
    ...extras,
  };
}

function appendHeartReading(current, metrics) {
  if (!metrics || metrics.bpm == null) {
    return current;
  }

  const entry = {
    bpm: Number(metrics.bpm),
    spo2: Number(metrics.spo2),
    capturedAt: metrics.capturedAt || new Date().toISOString(),
  };

  const last = current[current.length - 1];
  if (last && last.capturedAt === entry.capturedAt) {
    return current;
  }

  return [...current.slice(-59), entry];
}

function buildSparkline(values, width, height) {
  if (!values.length) {
    return { linePath: '', areaPath: '' };
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(max - min, 1);
  const step = values.length === 1 ? 0 : width / (values.length - 1);

  const points = values.map((value, index) => {
    const x = index * step;
    const y = height - ((value - min) / range) * (height - 14) - 7;
    return { x, y };
  });

  const linePath = points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(' ');

  const areaPath = `${linePath} L ${points[points.length - 1].x.toFixed(2)} ${height} L 0 ${height} Z`;
  return { linePath, areaPath };
}

function BurnoutGauge({ value, tone, label }) {
  const safeValue = Math.max(0, Math.min(100, Number(value) || 0));
  const style = {
    '--gauge-angle': `${safeValue * 3.6}deg`,
    '--gauge-color': toneColor(tone),
  };

  return (
    <div className="gauge-shell">
      <div className="gauge-ring" style={style}>
        <div className="gauge-inner">
          <div className="gauge-value" style={{ color: toneColor(tone) }}>
            {Math.round(safeValue)}
          </div>
          <div className="gauge-label">Score</div>
        </div>
      </div>
      <div className={`gauge-status tone-${tone}`}>{label}</div>
      <div className="gauge-subtitle">Derived from the selected report</div>
    </div>
  );
}

function HeartChart({ values }) {
  if (!values.length) {
    return <div className="chart-empty">No heart readings captured yet.</div>;
  }

  const width = 620;
  const height = 120;
  const { linePath, areaPath } = buildSparkline(values, width, height);

  return (
    <svg className="hr-chart" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id="heartGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(239,68,68,0.35)" />
          <stop offset="100%" stopColor="rgba(239,68,68,0.02)" />
        </linearGradient>
      </defs>
      <path d={areaPath} fill="url(#heartGradient)" />
      <path d={linePath} fill="none" stroke="#ef4444" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

export default function App() {
  const api = window.stressLensApi;
  const [clock, setClock] = useState(() => new Date().toLocaleTimeString());
  const [loading, setLoading] = useState(true);
  const [models, setModels] = useState([]);
  const [modelChoice, setModelChoice] = useState('1');
  const [running, setRunning] = useState(false);
  const [launchNote, setLaunchNote] = useState('');
  const [reports, setReports] = useState([]);
  const [activeReportName, setActiveReportName] = useState('');
  const [activeReportContent, setActiveReportContent] = useState('');
  const [heartMetrics, setHeartMetrics] = useState(null);
  const [heartHistory, setHeartHistory] = useState([]);
  const [heartSensorInfo, setHeartSensorInfo] = useState(null);
  const [ttsInfo, setTtsInfo] = useState(DEFAULT_TTS);
  const [chatMessages, setChatMessages] = useState([
    createChatEntry(
      'assistant',
      'Hey there! I can chat about the latest report, and Python-side Kokoro TTS will read my replies when it is enabled.'
    ),
  ]);
  const [chatInput, setChatInput] = useState('');
  const [includeLatestReport, setIncludeLatestReport] = useState(true);
  const [chatBusy, setChatBusy] = useState(false);
  const [heartBusy, setHeartBusy] = useState(false);
  const [lastLog, setLastLog] = useState('Waiting for bridge activity.');

  const reportSummary = parseReport(activeReportContent);
  const selectedModel = models.find((item) => item.choice === modelChoice);
  const heartValue = heartMetrics?.bpm ?? parseMetricNumber(reportSummary.heartRate);
  const spo2Value = heartMetrics?.spo2 ?? parseMetricNumber(reportSummary.spo2);
  const chartValues = heartHistory.length ? heartHistory.map((item) => item.bpm) : heartValue ? [heartValue] : [];
  const activeReportMeta = reports.find((item) => item.fileName === activeReportName);
  const tone = statusTone(reportSummary.status);
  const visibleReports = reports.slice(0, 10);

  function appendSystemMessage(message) {
    setChatMessages((current) => [...current.slice(-18), createChatEntry('system', message)]);
  }

  async function openReport(fileName) {
    if (!api || !fileName) {
      return;
    }

    try {
      const payload = await api.readReport(fileName);
      setActiveReportName(payload.fileName);
      setActiveReportContent(payload.content || '');
    } catch (error) {
      setLastLog(`reports: Failed to read ${fileName}: ${error.message}`);
    }
  }

  function applyBootstrap(payload) {
    setModels(payload.models || []);
    setModelChoice(payload.selectedModelChoice || '1');
    setReports(payload.reports || []);
    setRunning(Boolean(payload.mainRunning?.running));
    setLaunchNote(payload.launchNote || 'Use Q, R, P, and S inside the camera window.');
    setHeartSensorInfo(payload.heartSensor || null);
    setTtsInfo(payload.tts || DEFAULT_TTS);
    setLastLog(`bridge: Ready using ${payload.pythonExecutable}`);

    if (payload.latestHeartMetrics) {
      setHeartMetrics(payload.latestHeartMetrics);
      setHeartHistory((current) => appendHeartReading(current, payload.latestHeartMetrics));
    }

    if (payload.latestReport) {
      setActiveReportName(payload.latestReport.fileName);
      setActiveReportContent(payload.latestReport.content || '');
    } else if (payload.reports?.length) {
      openReport(payload.reports[0].fileName);
    }
  }

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setClock(new Date().toLocaleTimeString());
    }, 1000);

    return () => window.clearInterval(intervalId);
  }, []);

  useEffect(() => {
    let disposed = false;

    async function hydrate() {
      if (!api) {
        setLoading(false);
        return;
      }

      setLoading(true);
      try {
        const payload = await api.getBootstrap();
        if (!disposed) {
          applyBootstrap(payload);
        }
      } catch (error) {
        if (!disposed) {
          setLastLog(`bridge: Bootstrap failed: ${error.message}`);
        }
      } finally {
        if (!disposed) {
          setLoading(false);
        }
      }
    }

    hydrate();

    if (!api) {
      return () => {
        disposed = true;
      };
    }

    const dispose = api.onBridgeEvent((payload) => {
      if (disposed) {
        return;
      }

      if (payload.event === 'main-status') {
        setRunning(Boolean(payload.data?.running));
        if (payload.data?.modelChoice) {
          setModelChoice(payload.data.modelChoice);
        }
        if (payload.data?.running) {
          appendSystemMessage(`Camera monitor launched with ${payload.data.modelName}.`);
        } else {
          appendSystemMessage(
            `Camera monitor stopped${payload.data?.exitCode !== undefined ? ` (exit ${payload.data.exitCode})` : ''}.`
          );
        }
      }

      if (payload.event === 'heart-rate') {
        setHeartMetrics(payload.data);
        setHeartHistory((current) => appendHeartReading(current, payload.data));
      }

      if (payload.event === 'report-updated') {
        setReports(payload.data?.reports || []);
        if (payload.data?.fileName && payload.data?.content) {
          setActiveReportName(payload.data.fileName);
          setActiveReportContent(payload.data.content);
          appendSystemMessage(`${payload.data.kind || 'updated'} ${payload.data.fileName}.`);
        }
      }

      if (payload.event === 'tts-status') {
        setTtsInfo(payload.data || DEFAULT_TTS);
      }

      if (payload.event === 'log' && payload.data?.message) {
        setLastLog(`${payload.data.source || 'log'}: ${payload.data.message}`);
      }
    });

    return () => {
      disposed = true;
      if (typeof dispose === 'function') {
        dispose();
      }
    };
  }, []);

  async function handleStart() {
    if (!api) {
      return;
    }

    try {
      const payload = await api.startMonitoring(modelChoice);
      setRunning(Boolean(payload.running));
      appendSystemMessage(
        payload.alreadyRunning
          ? 'main.py was already running.'
          : 'Camera window launched. Use R or S there to generate reports.'
      );
    } catch (error) {
      appendSystemMessage(`Launch failed: ${error.message}`);
    }
  }

  async function handleForceStop() {
    if (!api) {
      return;
    }

    try {
      const payload = await api.stopMonitoring();
      setRunning(false);
      appendSystemMessage(payload.note || 'main.py was stopped.');
    } catch (error) {
      appendSystemMessage(`Stop failed: ${error.message}`);
    }
  }

  async function handleHeartCapture() {
    if (!api) {
      return;
    }

    setHeartBusy(true);
    try {
      const payload = await api.captureHeartRate();
      if (payload.metrics) {
        setHeartMetrics(payload.metrics);
        setHeartHistory((current) => appendHeartReading(current, payload.metrics));
      }
    } catch (error) {
      appendSystemMessage(`Heart capture failed: ${error.message}`);
    } finally {
      setHeartBusy(false);
    }
  }

  async function handleOpenReportFolder() {
    if (!api) {
      return;
    }

    try {
      await api.openReportFolder();
    } catch (error) {
      appendSystemMessage(`Could not open the report folder: ${error.message}`);
    }
  }

  async function handleRefreshReports() {
    if (!api) {
      return;
    }

    try {
      const payload = await api.listReports();
      setReports(payload.reports || []);
      if (!activeReportName && payload.reports?.length) {
        await openReport(payload.reports[0].fileName);
      }
    } catch (error) {
      appendSystemMessage(`Refresh failed: ${error.message}`);
    }
  }

  async function handleToggleTts() {
    if (!api) {
      return;
    }

    try {
      const payload = await api.setTtsEnabled(!ttsInfo.enabled);
      setTtsInfo(payload || DEFAULT_TTS);
    } catch (error) {
      appendSystemMessage(`Could not update Python TTS: ${error.message}`);
    }
  }

  async function handleChatSend() {
    if (!api) {
      return;
    }

    const trimmed = chatInput.trim();
    if (!trimmed) {
      return;
    }

    setChatMessages((current) => [...current.slice(-18), createChatEntry('user', trimmed)]);
    setChatInput('');
    setChatBusy(true);

    try {
      const payload = await api.sendChat(trimmed, includeLatestReport, modelChoice);
      setChatMessages((current) => [
        ...current.slice(-18),
        createChatEntry('assistant', payload.response || 'No reply returned.', {
          thinking: payload.thinking,
          meta: payload.includedReport ? `${payload.model} with ${payload.includedReport}` : payload.model,
        }),
      ]);

      if (payload.tts) {
        setTtsInfo(payload.tts);
      }
    } catch (error) {
      setChatMessages((current) => [
        ...current.slice(-18),
        createChatEntry('system', `Chat failed: ${error.message}`),
      ]);
    } finally {
      setChatBusy(false);
    }
  }

  function handleChatKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleChatSend();
    }
  }

  if (!api) {
    return (
      <div className="missing-shell">
        <div className="missing-card">
          <h1>Open this renderer through Electron</h1>
          <p>The preload bridge is missing in a plain browser tab, so the desktop controls are not available here.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="screen">
      <header className="topbar">
        <div className="topbar-brand">
          <div className="logo">S</div>
          <h1>
            Stress<span>Lens</span> AI
          </h1>
        </div>

        <div className="topbar-controls">
          <select className="topbar-select" value={modelChoice} onChange={(event) => setModelChoice(event.target.value)}>
            {models.map((item) => (
              <option key={item.choice} value={item.choice}>
                {item.label}
              </option>
            ))}
          </select>
          <button className="topbar-btn start" onClick={handleStart} disabled={loading || running}>
            {running ? 'Monitor Running' : 'Launch Monitor'}
          </button>
          <button className="topbar-btn" onClick={handleHeartCapture} disabled={heartBusy}>
            {heartBusy ? 'Capturing...' : 'Measure Heart'}
          </button>
          <button className="topbar-btn stop" onClick={handleForceStop} disabled={!running}>
            Force Stop
          </button>
          <button className="topbar-btn" onClick={handleOpenReportFolder}>
            Open Reports
          </button>
        </div>

        <div className="topbar-status">
          <div className="status-pill">
            <span className={`status-dot ${running ? 'on' : 'off'}`} />
            <span>{running ? 'Monitoring' : 'Idle'}</span>
          </div>
          <span className="clock">{clock}</span>
        </div>
      </header>

      <div className="main">
        <div className="dashboard">
          <section className="card">
            <div className="card-header">
              <span className="card-title">Burnout Risk</span>
              <span className="card-badge">Report-derived</span>
            </div>
            <BurnoutGauge value={reportSummary.avgBurnoutValue} tone={tone} label={tone.toUpperCase()} />
          </section>

          <section className="card">
            <div className="card-header">
              <span className="card-title">Emotion Breakdown</span>
              <span className="card-badge">DeepFace</span>
            </div>
            <div>
              {reportSummary.emotions.length === 0 ? (
                <div className="empty-copy">Use R or S in the camera window to populate the report-driven emotion bars.</div>
              ) : (
                reportSummary.emotions.map((item) => (
                  <div key={item.emotion} className="emotion-row">
                    <span className="emotion-label">{item.emotion}</span>
                    <div className="emotion-bar-bg">
                      <div
                        className="emotion-bar-fill"
                        style={{
                          width: `${item.percentage}%`,
                          background: EMOTION_COLORS[item.emotion] || '#64748b',
                        }}
                      />
                    </div>
                    <span className="emotion-pct">{Math.round(item.percentage)}%</span>
                  </div>
                ))
              )}
            </div>
          </section>

          <div className="metrics-row">
            <div className="metric-card">
              <div className="metric-value" style={{ color: '#ef4444' }}>{heartValue ?? '--'}</div>
              <div className="metric-name">Heart Rate</div>
              <div className="metric-trend">{heartMetrics ? 'Live capture' : heartSensorInfo?.available ? 'Report fallback' : 'No sensor'}</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ color: '#3b82f6' }}>{spo2Value ?? '--'}</div>
              <div className="metric-name">SpO2 %</div>
              <div className="metric-trend">{heartSensorInfo?.available ? `${heartSensorInfo.port} @ ${heartSensorInfo.baud}` : 'No sensor configured'}</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ color: toneColor(tone) }}>{reportSummary.avgBurnout}</div>
              <div className="metric-name">Average Burnout</div>
              <div className="metric-trend">Peak {reportSummary.peakBurnout}</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ color: '#14b8a6' }}>{reportSummary.durationClock}</div>
              <div className="metric-name">Session</div>
              <div className="metric-trend">{reportSummary.samples} samples</div>
            </div>
          </div>

          <section className="card">
            <div className="card-header">
              <span className="card-title">Heart Rate History</span>
              <span className="card-badge badge-red">BPM</span>
            </div>
            <div className="chart-value">
              <span>{heartValue ?? '--'}</span>
              <span className="chart-unit">bpm</span>
            </div>
            <HeartChart values={chartValues} />
          </section>

          <section className="card">
            <div className="card-header">
              <span className="card-title">Session Info</span>
              <span className="card-badge">{selectedModel?.model || 'No model'}</span>
            </div>
            <div className="info-grid">
              <div className="info-row">
                <span className="info-label">Model</span>
                <span className="info-value">{selectedModel?.description || 'No model selected'}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Status</span>
                <span className="info-value">{running ? 'Camera window active' : 'Desktop idle'}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Latest report</span>
                <span className="info-value">{activeReportName || 'No report selected'}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Python TTS</span>
                <span className="info-value">{ttsInfo.message}</span>
              </div>
            </div>
            <p className="info-note">{launchNote}</p>
            <p className="info-note info-log">{lastLog}</p>
          </section>

          <section className="card report-card">
            <div className="card-header">
              <span className="card-title">Latest Report</span>
              <div className="report-toolbar">
                <span className="card-badge">{activeReportMeta ? timestampLabel(activeReportMeta.modifiedAt) : '--'}</span>
                <button className="inline-btn" onClick={handleRefreshReports}>Refresh</button>
              </div>
            </div>

            <div className="report-tabs">
              {visibleReports.length === 0 ? (
                <span className="empty-copy">No report files yet.</span>
              ) : (
                visibleReports.map((item) => (
                  <button
                    key={item.fileName}
                    className={`report-tab ${item.fileName === activeReportName ? 'active' : ''}`}
                    onClick={() => openReport(item.fileName)}
                  >
                    {item.fileName}
                  </button>
                ))
              )}
            </div>

            <pre className="report-content">{activeReportContent || 'Press R in the camera window to generate a report.'}</pre>
          </section>

          <div className="key-legend">
            <span>Camera Window Keys:</span>
            <span className="key-pill"><kbd>Q</kbd> Quit</span>
            <span className="key-pill"><kbd>R</kbd> Report</span>
            <span className="key-pill"><kbd>P</kbd> Privacy</span>
            <span className="key-pill"><kbd>S</kbd> AI Support</span>
            <span className="key-pill"><kbd>H</kbd> Heart Rate</span>
          </div>
        </div>

        <aside className="sidebar">
          <div className="sidebar-header">
            <h2>
              <span className="ai-dot" /> AI Wellness Advisor
            </h2>
            <p>Powered by local Ollama with Python-side Kokoro TTS</p>
          </div>

          <div className="chat-messages">
            {reportSummary.advice ? (
              <article className="chat-msg msg-ai latest-advice">
                <div className="msg-sender">Latest Advice</div>
                {reportSummary.advice}
              </article>
            ) : null}

            {chatMessages.map((message) => (
              <article key={message.id} className={`chat-msg msg-${message.role}`}>
                {message.role === 'assistant' ? <div className="msg-sender">StressLens AI</div> : null}
                {message.meta ? <div className="msg-meta">{message.meta}</div> : null}
                {message.thinking ? <div className="msg-thinking">Thinking: {message.thinking}</div> : null}
                <div>{message.content}</div>
              </article>
            ))}
          </div>

          <div className="tts-bar">
            <span>{ttsInfo.message}</span>
            <button className="mute-toggle" onClick={handleToggleTts} disabled={!ttsInfo.available}>
              {ttsInfo.enabled ? 'Unmuted' : 'Muted'}
            </button>
          </div>

          <div className="chat-input">
            <label className="chat-checkbox">
              <input
                type="checkbox"
                checked={includeLatestReport}
                onChange={(event) => setIncludeLatestReport(event.target.checked)}
              />
              Include latest report in context
            </label>

            <div className="chat-input-row">
              <input
                type="text"
                value={chatInput}
                onChange={(event) => setChatInput(event.target.value)}
                onKeyDown={handleChatKeyDown}
                placeholder="Ask for wellness advice..."
              />
              <button onClick={handleChatSend} disabled={chatBusy}>
                {chatBusy ? 'Sending...' : 'Send'}
              </button>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
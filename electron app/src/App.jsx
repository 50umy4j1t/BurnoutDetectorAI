import { useState, useEffect, useRef } from 'react'
import TopBar from './components/TopBar'
import VideoPanel from './components/VideoPanel'
import BurnoutGauge from './components/BurnoutGauge'
import MetricCards from './components/MetricCards'
import EmotionPanel from './components/EmotionPanel'
import HeartRateChart from './components/HeartRateChart'
import AIChat from './components/AIChat'
import BottomBar from './components/BottomBar'
// TTS is handled by the Python backend (kokoro_onnx + sounddevice)
import './App.css'

/* ── Report parsing utilities ── */

function extractLineValue(text, label) {
  if (!text) return '--'
  const line = text.split(/\r?\n/).find(l => l.includes(label))
  if (!line) return '--'
  return line.slice(line.indexOf(label) + label.length).trim() || '--'
}

function parseMetricNumber(value) {
  const m = String(value || '').match(/-?\d+(?:\.\d+)?/)
  return m ? Number(m[0]) : null
}

function parseEmotionMap(text) {
  if (!text || !text.includes('AVERAGE EMOTION BREAKDOWN:')) {
    return { happy:0.12, sad:0.05, angry:0.03, fear:0.02, surprise:0.04, disgust:0.01, neutral:0.73 }
  }
  const sectionMatch = text.match(/AVERAGE EMOTION BREAKDOWN:\s*([\s\S]*?)^-{5,}/m)
  const section = sectionMatch ? sectionMatch[1] : ''
  const result = {}
  for (const line of section.split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Za-z_]+)\s+(\d+(?:\.\d+)?)%/)
    if (m) result[m[1].toLowerCase()] = Number(m[2]) / 100
  }
  if (Object.keys(result).length === 0)
    return { happy:0.12, sad:0.05, angry:0.03, fear:0.02, surprise:0.04, disgust:0.01, neutral:0.73 }
  return result
}

function parseDurationSeconds(text) {
  const d = extractLineValue(text, 'Duration      :')
  const m = String(d).match(/(\d+)m\s+(\d+)s/i)
  return m ? Number(m[1]) * 60 + Number(m[2]) : 0
}

function parseSamples(text) {
  const v = extractLineValue(text, 'Samples       :')
  return parseMetricNumber(v) ?? 0
}

function parseBias(text) {
  const lines = (text || '').split(/\r?\n/)
  const biasLine = lines.find(l => l.includes('Dominant Emotion :') || l.includes('DOMINANT EMOTION'))
  if (!biasLine) return 'Awaiting session'
  const parts = biasLine.split(':')
  return parts.length > 1 ? parts[1].trim() : 'Awaiting session'
}

function extractAdviceFromReport(text) {
  if (!text) return ''
  const marker = 'AI WELLNESS ADVISOR RESPONSE:'
  const idx = text.indexOf(marker)
  if (idx < 0) return ''
  return text.slice(idx + marker.length).trim()
}

const DEFAULT_EMOTIONS = { happy:0.12, sad:0.05, angry:0.03, fear:0.02, surprise:0.04, disgust:0.01, neutral:0.73 }

const DEFAULT_TTS = { available: true, enabled: true, state: 'ready', message: 'Frontend TTS ready', voice: 'af_heart' }

/* ── Main App ── */

export default function App() {
  const api = window.stressLensApi

  const [loading, setLoading]       = useState(true)
  const [systemOk, setSystemOk]     = useState(false)
  const [systemError, setSystemError] = useState('')
  const [models, setModels]         = useState([])
  const [selectedModel, setSelectedModel] = useState('1')
  const [running, setRunning]       = useState(false)
  const [ttsInfo, setTtsInfo]       = useState(DEFAULT_TTS)
  const [aiOpen, setAiOpen]         = useState(false)
  const [heartBusy, setHeartBusy]   = useState(false)
  const [injectMessages, setInjectMessages] = useState([])

  // Report-derived data
  const [emotions, setEmotions]     = useState(DEFAULT_EMOTIONS)
  const [burnout, setBurnout]       = useState(0)
  const [samples, setSamples]       = useState(0)
  const [elapsed, setElapsed]       = useState(0)
  const [bias, setBias]             = useState('Awaiting session')
  const [heartRate, setHeartRate]   = useState(null)
  const [spo2, setSpo2]             = useState(null)
  const [hrHistory, setHrHistory]   = useState([])

  const elapsedRef = useRef(null)
  const seenAdviceRef = useRef(new Set())

  function applyReport(text) {
    setEmotions(parseEmotionMap(text))
    setBurnout(parseMetricNumber(extractLineValue(text, 'AVG BURNOUT SCORE :')) ?? 0)
    setSamples(parseSamples(text))
    setBias(parseBias(text))
    const elSec = parseDurationSeconds(text)
    if (elSec > 0) setElapsed(elSec)
    const hr = parseMetricNumber(extractLineValue(text, 'HEART RATE (BPM)  :'))
    const sp = parseMetricNumber(extractLineValue(text, 'BLOOD OXYGEN      :'))
    if (hr != null) { setHeartRate(hr); addHr(hr) }
    if (sp != null) setSpo2(sp)
  }

  function addHr(bpm) {
    if (bpm == null) return
    setHrHistory(prev => {
      const entry = { t: prev.length, bpm: Number(bpm) }
      return [...prev.slice(-99), entry]
    })
  }

  function applyBootstrap(payload) {
    setModels(payload.models || [])
    setSelectedModel(payload.selectedModelChoice || '1')
    setRunning(Boolean(payload.mainRunning?.running))
    setTtsInfo(payload.tts || DEFAULT_TTS)
    setSystemOk(true)
    setSystemError('')

    if (payload.latestHeartMetrics) {
      setHeartRate(payload.latestHeartMetrics.bpm)
      setSpo2(payload.latestHeartMetrics.spo2)
      addHr(payload.latestHeartMetrics.bpm)
    }

    if (payload.latestReport?.content) {
      applyReport(payload.latestReport.content)
    }
  }

  // Bootstrap
  useEffect(() => {
    // TTS priming no longer needed — backend handles speech

    if (!api) { setLoading(false); return }
    let disposed = false
    ;(async () => {
      try {
        const payload = await api.getBootstrap()
        if (!disposed) applyBootstrap(payload)
      } catch (e) {
        if (!disposed) { setSystemOk(false); setSystemError(e.message || 'Bridge unreachable') }
      } finally {
        if (!disposed) setLoading(false)
      }
    })()

    const unsub = api.onBridgeEvent(ev => {
      if (disposed) return

      if (ev.event === 'main-status') {
        setRunning(Boolean(ev.data?.running))
        if (ev.data?.modelChoice) setSelectedModel(ev.data.modelChoice)
      }

      if (ev.event === 'heart-rate' && ev.data) {
        setHeartRate(ev.data.bpm)
        setSpo2(ev.data.spo2)
        addHr(ev.data.bpm)
      }

      if (ev.event === 'report-updated' && ev.data?.content) {
        applyReport(ev.data.content)

        const advice = ev.data.aiAdvice || extractAdviceFromReport(ev.data.content)
        if (advice) {
          const key = `${ev.data.fileName || 'unknown'}:${advice}`
          if (!seenAdviceRef.current.has(key)) {
            seenAdviceRef.current.add(key)
            const msg = { role:'ai', text: advice, ts: Date.now(), source:'report' }
            setInjectMessages(p => [...p, msg])
            setAiOpen(true)
          }
        }
      }

      if (ev.event === 'tts-status') {
        setTtsInfo(ev.data || DEFAULT_TTS)
      }

      if (ev.event === 'ai-advice' && ev.data?.advice) {
        const key = `${ev.data.fileName || 'unknown'}:${ev.data.advice}`
        if (!seenAdviceRef.current.has(key)) {
          seenAdviceRef.current.add(key)
          const msg = { role:'ai', text: ev.data.advice, ts: Date.now(), source:'report' }
          setInjectMessages(p => [...p, msg])
          setAiOpen(true)
        }
      }

      if (ev.event === 'log' && ev.data?.message) {
        const lower = (ev.data.message || '').toLowerCase()
        const src = ev.data.source || ''
        if ((src === 'heart' || src.includes('main')) && /place your finger|hold still|listening on/.test(lower)) {
          const msg = { role:'system', text: 'Place your finger on the heart sensor and hold still...', ts: Date.now() }
          setInjectMessages(p => [...p, msg])
          setAiOpen(true)
        }
        if ((src === 'heart' || src.includes('main')) && /heart sensor timed out|timeout/.test(lower)) {
          const msg = { role:'system', text: 'Heart sensor timed out — no reading captured.', ts: Date.now() }
          setInjectMessages(p => [...p, msg])
        }
        if (src.includes('main') && /sending to ai advisor/.test(lower)) {
          const msg = { role:'system', text: 'Report sent to AI advisor — thinking...', ts: Date.now() }
          setInjectMessages(p => [...p, msg])
          setAiOpen(true)
        }
      }
    })

    return () => { disposed = true; if (typeof unsub === 'function') unsub() }
  }, [])

  // Elapsed timer when running
  useEffect(() => {
    if (!running) { if (elapsedRef.current) clearInterval(elapsedRef.current); return }
    elapsedRef.current = setInterval(() => setElapsed(s => s + 1), 1000)
    return () => clearInterval(elapsedRef.current)
  }, [running])

  async function handleStart() {
    if (!api) return
    try {
      const p = await api.startMonitoring(selectedModel)
      setRunning(Boolean(p.running))
    } catch {}
  }

  async function handleStop() {
    if (!api) return
    try { await api.stopMonitoring(); setRunning(false) } catch {}
  }

  async function handleHeartCapture() {
    if (!api) return
    setHeartBusy(true)
    try {
      const p = await api.captureHeartRate()
      if (p.metrics) {
        setHeartRate(p.metrics.bpm)
        setSpo2(p.metrics.spo2)
        addHr(p.metrics.bpm)
      }
    } catch {}
    setHeartBusy(false)
  }

  async function handleTtsToggle() {
    if (!api) return
    try {
      const p = await api.setTtsEnabled(!ttsInfo.enabled)
      setTtsInfo(p || DEFAULT_TTS)
    } catch {}
  }

  async function handleOpenReports() {
    if (!api) return
    try { await api.openReportFolder() } catch {}
  }

  const modelName = models.find(m => m.choice === selectedModel)?.name || 'AI Model'

  const dataForCards = {
    heartRate, spo2, burnout, running, elapsed,
  }

  if (loading) {
    return (
      <div className="app" style={{display:'flex',alignItems:'center',justifyContent:'center'}}>
        <div style={{textAlign:'center',color:'var(--text2)'}}>
          <div style={{fontSize:'24px',marginBottom:'8px'}}>⟳</div>
          Connecting to Python bridge…
        </div>
      </div>
    )
  }

  return (
    <>
    <div className="app">
      <TopBar
        running={running}
        systemOk={systemOk} systemError={systemError}
        models={models} selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        onStart={handleStart} onStop={handleStop}
      />

      <div className="body">
        <div className="content">
          <div className="row-top">
            <VideoPanel running={running} elapsed={elapsed} />
            <BurnoutGauge burnout={burnout} running={running} />
            <EmotionPanel emotions={emotions} samples={samples} bias={bias} />
          </div>
          <div className="row-bottom">
            <MetricCards data={dataForCards} />
            <HeartRateChart history={hrHistory} running={running} />
          </div>
        </div>

        <AIChat
          open={aiOpen} onClose={() => setAiOpen(false)}
          api={api} modelName={modelName}
          ttsInfo={ttsInfo} onTtsToggle={handleTtsToggle}
          onHeartCapture={handleHeartCapture} heartBusy={heartBusy}
          injectMessages={injectMessages}
        />
      </div>

      <BottomBar
        running={running} samples={samples} elapsed={elapsed}
        aiOpen={aiOpen} onToggleAI={() => setAiOpen(o=>!o)}
        onStart={handleStart} onStop={handleStop}
        onHeartCapture={handleHeartCapture} heartBusy={heartBusy}
        onOpenReports={handleOpenReports}
      />
    </div>

    {burnout >= 85 && (
      <div style={{
        position:'fixed', inset:0, zIndex:999, pointerEvents:'none',
        background:'rgba(200,138,122,0.05)',
        boxShadow:'inset 0 0 80px rgba(200,138,122,0.15)',
        animation:'burnout-pulse 3s ease-in-out infinite',
      }}/>
    )}
    {burnout >= 85 && (
      <div style={{
        position:'fixed', top:'50%', left:'50%', transform:'translate(-50%,-50%)',
        zIndex:1000, pointerEvents:'none',
        background:'rgba(26,25,21,0.95)',
        border:'0.5px solid rgba(200,138,122,0.4)',
        borderRadius:'16px', padding:'32px 44px', textAlign:'center',
        boxShadow:'0 8px 40px rgba(0,0,0,0.4)',
      }}>
        <div style={{fontFamily:'var(--font-heading)', fontSize:'20px', fontWeight:700, color:'var(--stress-high)', marginBottom:'10px'}}>
          Needs attention
        </div>
        <div style={{fontFamily:'var(--font-body)', fontSize:'14px', color:'var(--text-secondary)', lineHeight:1.6}}>
          Your burnout score is <strong style={{color:'var(--stress-high)'}}>{Math.round(burnout)}</strong>.<br/>
          You might want to take a break.<br/>
          <em style={{color:'var(--text-tertiary)'}}>Step away from the screen and rest.</em>
        </div>
      </div>
    )}
    </>
  )
}

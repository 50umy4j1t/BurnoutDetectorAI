import { useState, useEffect, useRef } from 'react'
import TopBar from './components/TopBar'
import VideoPanel from './components/VideoPanel'
import BurnoutGauge from './components/BurnoutGauge'
import MetricCards from './components/MetricCards'
import EmotionPanel from './components/EmotionPanel'
import HeartRateChart from './components/HeartRateChart'
import AIChat from './components/AIChat'
import BottomBar from './components/BottomBar'
import './App.css'

const API = 'http://localhost:8000'

const DEFAULT = {
  emotions:{ happy:0.12, sad:0.05, angry:0.03, fear:0.02, surprise:0.04, disgust:0.01, neutral:0.73 },
  burnout:0, bias:'Awaiting session', samples:0, elapsed:0,
  privacy:false, running:false, model:'qwen3:4b',
  heartRate:null, spo2:null, stressLevel:0
}

export default function App() {
  const [data, setData] = useState(DEFAULT)
  const [hrHistory, setHrHistory] = useState(
    Array.from({length:50},(_,i)=>({ t:i, bpm: Math.round(70+Math.sin(i*0.5)*7+Math.cos(i*0.3)*4) }))
  )
  const [aiOpen, setAiOpen]         = useState(false)
  const [models, setModels]         = useState([])
  const [selectedModel, setSelectedModel] = useState('qwen3:4b')
  const [systemOk, setSystemOk]     = useState(true)
  const [systemError, setSystemError] = useState('')
  const wsRef = useRef(null)
  const hrCounter = useRef(0)

  useEffect(() => {
    fetch(`${API}/api/models`)
      .then(r => r.json())
      .then(d => { setModels(d.models||[]); setSelectedModel(d.current||'qwen3:4b'); setSystemOk(true); setSystemError('') })
      .catch(e => { setSystemOk(false); setSystemError(e.message||'Cannot reach backend') })
  }, [])

  useEffect(() => {
    let reconnect
    const connect = () => {
      try {
        const ws = new WebSocket('ws://localhost:8000/ws/emotions')
        wsRef.current = ws
        ws.onopen  = () => { setSystemOk(true); setSystemError('') }
        ws.onmessage = e => {
          try {
            const d = JSON.parse(e.data)
            setData(prev => ({
              ...d,
              emotions: (d.emotions && Object.keys(d.emotions).length > 0) ? d.emotions : prev.emotions
            }))
            if (d.running) {
              hrCounter.current++
              const bpm = d.heartRate ?? Math.round(68+(d.burnout/100)*22+Math.sin(hrCounter.current*0.4)*4)
              setHrHistory(p => [...p.slice(-99), { t: d.elapsed, bpm }])
            }
          } catch(_){}
        }
        ws.onerror = () => { setSystemOk(false); setSystemError('WebSocket failed') }
        ws.onclose = () => { reconnect = setTimeout(connect, 3000) }
      } catch(_) { reconnect = setTimeout(connect, 3000) }
    }
    connect()
    return () => { clearTimeout(reconnect); wsRef.current?.close() }
  }, [])

  const startSession = () =>
    fetch(`${API}/api/session/start`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:selectedModel})}).catch(()=>{})
  const stopSession = () =>
    fetch(`${API}/api/session/stop`,{method:'POST'}).catch(()=>{})
  const togglePrivacy = () =>
    fetch(`${API}/api/settings`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({privacy_mode:!data.privacy})}).catch(()=>{})

  return (
    <>
    <div className="app">
      <TopBar
        running={data.running} privacy={data.privacy}
        systemOk={systemOk} systemError={systemError}
        models={models} selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        onPrivacyToggle={togglePrivacy}
        onStart={startSession} onStop={stopSession}
      />

      <div className="body">
        <div className="content">
          <div className="row-top">
            <VideoPanel privacy={data.privacy} running={data.running} elapsed={data.elapsed} />
            <BurnoutGauge burnout={data.burnout} running={data.running} emotions={data.emotions} />
            <EmotionPanel emotions={data.emotions} samples={data.samples} bias={data.bias} />
          </div>
          <div className="row-bottom">
            <MetricCards data={data} />
            <HeartRateChart history={hrHistory} running={data.running} />
          </div>
        </div>
        <AIChat open={aiOpen} onClose={() => setAiOpen(false)} sessionData={data} />
      </div>

      <BottomBar
        running={data.running} samples={data.samples} elapsed={data.elapsed}
        aiOpen={aiOpen} onToggleAI={() => setAiOpen(o=>!o)}
        onStart={startSession} onStop={stopSession}
      />
    </div>

    {/* ── BURNOUT CRITICAL ALERT OVERLAY ── */}
    {data.burnout >= 85 && (
      <div style={{
        position:'fixed', inset:0, zIndex:999, pointerEvents:'none',
        background:'rgba(255,30,60,0.08)',
        boxShadow:'inset 0 0 120px rgba(255,30,60,0.35)',
        animation:'burnout-pulse 2s ease-in-out infinite',
      }}/>
    )}
    {data.burnout >= 85 && (
      <div style={{
        position:'fixed', top:'50%', left:'50%', transform:'translate(-50%,-50%)',
        zIndex:1000, pointerEvents:'none',
        background:'rgba(10,5,10,0.92)',
        border:'2px solid rgba(255,60,80,0.8)',
        borderRadius:'16px', padding:'28px 40px', textAlign:'center',
        boxShadow:'0 0 60px rgba(255,30,60,0.5)',
        animation:'burnout-pulse 2s ease-in-out infinite',
      }}>
        <div style={{fontSize:'36px', marginBottom:'8px'}}>⚠️</div>
        <div style={{fontFamily:'var(--display)', fontSize:'22px', fontWeight:800, color:'#ff3d5a', letterSpacing:'0.05em', marginBottom:'6px'}}>
          BURNOUT ALERT
        </div>
        <div style={{fontFamily:'var(--mono)', fontSize:'12px', color:'rgba(255,150,160,0.85)', lineHeight:1.7}}>
          Your burnout score is <strong style={{color:'#ff3d5a'}}>{Math.round(data.burnout)}</strong> — critically high.<br/>
          Please stop and take a break immediately.<br/>
          Step away from the screen and rest.
        </div>
      </div>
    )}
    </>
  )
}

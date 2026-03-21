import { Heart, Droplets, Zap, Timer } from 'lucide-react'
import { useState, useEffect } from 'react'

function useTimer(running, elapsed) {
  const [secs, setSecs] = useState(elapsed)
  useEffect(() => { setSecs(elapsed) }, [elapsed])
  useEffect(() => {
    if (!running) return
    const id = setInterval(() => setSecs(s => s + 1), 1000)
    return () => clearInterval(id)
  }, [running])
  const m = Math.floor(secs / 60), s = secs % 60
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`
}

function Card({ children, style }) {
  return <div className="card metric-card" style={style}>{children}</div>
}

function getStressStatus(v) {
  if (v < 25) return { t:'Looking good', c:'var(--stress-low)', bg:'var(--stress-low-bg)' }
  if (v < 50) return { t:'Moderate', c:'var(--stress-moderate)', bg:'var(--stress-moderate-bg)' }
  if (v < 70) return { t:'Elevated', c:'var(--stress-elevated)', bg:'var(--stress-elevated-bg)' }
  return { t:'Needs attention', c:'var(--stress-high)', bg:'var(--stress-high-bg)' }
}

export default function MetricCards({ data }) {
  const hr    = data.heartRate ?? '--'
  const spo2  = data.spo2 ?? '--'
  const timer = useTimer(data.running, data.elapsed)
  const stress = getStressStatus(data.burnout)

  return (
    <div style={{width:'33%', flexShrink:0, display:'grid', gridTemplateColumns:'1fr 1fr', gridTemplateRows:'1fr 1fr', gap:'var(--card-gap)'}}>

      <Card>
        <div style={{display:'flex', justifyContent:'space-between', alignItems:'flex-start'}}>
          <span className="card-hd">Heart rate</span>
          <span className="heart-pulse"><Heart size={15} style={{color:'var(--stress-high)', opacity:0.7}}/></span>
        </div>
        <div style={{marginTop:'8px'}}>
          <span style={{fontFamily:'var(--font-mono)',fontSize:'24px',fontWeight:300,color:'var(--text-primary)',lineHeight:1}}>{hr}</span>
          <span style={{fontFamily:'var(--font-body)',fontSize:'12px',color:'var(--text-tertiary)',marginLeft:'4px'}}>bpm</span>
        </div>
        <div className="metric-lbl">Heart rate</div>
        <div style={{marginTop:'8px',fontFamily:'var(--font-body)',fontSize:'11px',fontWeight:500,
          padding:'3px 10px',borderRadius:'99px',display:'inline-block',
          color:'var(--text-secondary)',border:'0.5px solid var(--border-default)',background:'transparent'}}>
          {hr==='--' ? <em>Connect sensor to begin</em> : 'Normal'}
        </div>
      </Card>

      <Card>
        <div style={{display:'flex', justifyContent:'space-between', alignItems:'flex-start'}}>
          <span className="card-hd">SpO2</span>
          <Droplets size={15} style={{color:'var(--emotion-neutral)', opacity:0.7}}/>
        </div>
        <div style={{marginTop:'8px'}}>
          <span style={{fontFamily:'var(--font-mono)',fontSize:'24px',fontWeight:300,color:'var(--text-primary)',lineHeight:1}}>{spo2}</span>
          <span style={{fontFamily:'var(--font-body)',fontSize:'12px',color:'var(--text-tertiary)',marginLeft:'4px'}}>%</span>
        </div>
        <div className="metric-lbl">Blood oxygen</div>
        <div style={{marginTop:'8px',fontFamily:'var(--font-body)',fontSize:'11px',fontWeight:500,
          padding:'3px 10px',borderRadius:'99px',display:'inline-block',
          color:'var(--text-secondary)',border:'0.5px solid var(--border-default)',background:'transparent'}}>
          {spo2==='--' ? <em>Connect sensor to begin</em> : 'Normal'}
        </div>
      </Card>

      <Card>
        <div style={{display:'flex', justifyContent:'space-between', alignItems:'flex-start'}}>
          <span className="card-hd">Stress index</span>
          <Zap size={14} style={{color:stress.c, opacity:0.7, transition:'color 0.5s'}}/>
        </div>
        <div style={{marginTop:'8px'}}>
          <span style={{fontFamily:'var(--font-mono)',fontSize:'24px',fontWeight:300,color:'var(--text-primary)',lineHeight:1,transition:'color 0.5s'}}>
            {data.burnout>0?Math.round(data.burnout):'--'}
          </span>
        </div>
        <div className="metric-lbl">Stress index</div>
        <div style={{marginTop:'8px',fontFamily:'var(--font-body)',fontSize:'11px',fontWeight:500,
          padding:'3px 10px',borderRadius:'99px',display:'inline-block',
          color:stress.c,border:`0.5px solid ${stress.c}33`,background:stress.bg,transition:'all 0.5s'}}>
          {stress.t}
        </div>
      </Card>

      <Card>
        <div style={{display:'flex', justifyContent:'space-between', alignItems:'flex-start'}}>
          <span className="card-hd">Session</span>
          <Timer size={14} style={{color:'var(--accent-primary)',opacity:data.running?0.7:0.3}}/>
        </div>
        <div style={{marginTop:'8px'}}>
          <span style={{fontFamily:'var(--font-mono)',fontSize:'24px',fontWeight:300,color:'var(--text-primary)',lineHeight:1}}>
            {timer}
          </span>
        </div>
        <div className="metric-lbl">Active time</div>
        <div style={{marginTop:'8px',fontFamily:'var(--font-body)',fontSize:'11px',fontWeight:500,
          padding:'3px 10px',borderRadius:'99px',display:'inline-block',
          color: data.running ? 'var(--session-active)' : 'var(--text-tertiary)',
          border: data.running ? '0.5px solid rgba(122,171,138,0.3)' : '0.5px solid var(--border-default)',
          background: data.running ? 'var(--stress-low-bg)' : 'transparent'}}>
          {data.running ? 'Recording' : 'Resting'}
        </div>
      </Card>

    </div>
  )
}

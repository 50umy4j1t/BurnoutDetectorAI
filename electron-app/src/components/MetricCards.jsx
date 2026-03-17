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

export default function MetricCards({ data }) {
  const hr    = data.heartRate ?? '--'
  const spo2  = data.spo2 ?? '--'
  const timer = useTimer(data.running, data.elapsed)
  const burnoutStatus = data.burnout < 25 ? {t:'↔ Stable', c:'var(--accent)'}
    : data.burnout < 50 ? {t:'↑ Low', c:'#7ae76e'}
    : data.burnout < 70 ? {t:'↑ Moderate', c:'var(--yellow)'}
    : data.burnout < 85 ? {t:'↑ High', c:'#ff8c42'}
    : {t:'↑ Critical', c:'var(--red)'}

  return (
    <div style={{width:'33%', flexShrink:0, display:'grid', gridTemplateColumns:'1fr 1fr', gridTemplateRows:'1fr 1fr', gap:'10px'}}>

      {/* Heart Rate */}
      <Card>
        <div style={{display:'flex', justifyContent:'space-between', alignItems:'flex-start'}}>
          <span className="card-hd">Heart Rate</span>
          <span className="heart-pulse"><Heart size={16} fill="var(--red)" stroke="none"/></span>
        </div>
        <div style={{marginTop:'6px'}}>
          <span style={{fontFamily:'var(--display)',fontSize:'36px',fontWeight:800,color:'var(--red)',lineHeight:1}}>{hr}</span>
          <span style={{fontSize:'12px',color:'var(--red)',opacity:.6,marginLeft:'3px'}}>bpm</span>
        </div>
        <div style={{fontSize:'9px',color:'var(--text2)',marginTop:'3px',letterSpacing:'0.1em',textTransform:'uppercase'}}>Heart Rate</div>
        <div style={{marginTop:'5px',fontSize:'8.5px',padding:'2px 7px',borderRadius:'9px',display:'inline-block',
          color:'var(--text2)',border:'1px solid var(--border)',background:'transparent'}}>
          ↔ {hr==='--'?'No Sensor':'Normal'}
        </div>
      </Card>

      {/* SPO2 */}
      <Card>
        <div style={{display:'flex', justifyContent:'space-between', alignItems:'flex-start'}}>
          <span className="card-hd">SPO2 %</span>
          <Droplets size={15} style={{color:'var(--blue)'}}/>
        </div>
        <div style={{marginTop:'6px'}}>
          <span style={{fontFamily:'var(--display)',fontSize:'36px',fontWeight:800,color:'var(--blue)',lineHeight:1}}>{spo2}</span>
          <span style={{fontSize:'12px',color:'var(--blue)',opacity:.6,marginLeft:'3px'}}>%</span>
        </div>
        <div style={{fontSize:'9px',color:'var(--text2)',marginTop:'3px',letterSpacing:'0.1em',textTransform:'uppercase'}}>Blood Oxygen</div>
        <div style={{marginTop:'5px',fontSize:'8.5px',padding:'2px 7px',borderRadius:'9px',display:'inline-block',
          color:'var(--blue)',border:'1px solid rgba(56,178,240,0.3)',background:'rgba(56,178,240,0.08)'}}>
          ↓ {spo2==='--'?'No Sensor':'Normal'}
        </div>
      </Card>

      {/* Stress Index */}
      <Card>
        <div style={{display:'flex', justifyContent:'space-between', alignItems:'flex-start'}}>
          <span className="card-hd">Stress Index</span>
          <Zap size={14} style={{color:burnoutStatus.c,transition:'color 0.5s'}}/>
        </div>
        <div style={{marginTop:'6px'}}>
          <span style={{fontFamily:'var(--display)',fontSize:'36px',fontWeight:800,color:burnoutStatus.c,lineHeight:1,transition:'color 0.5s'}}>
            {data.burnout>0?Math.round(data.burnout):'--'}
          </span>
        </div>
        <div style={{fontSize:'9px',color:'var(--text2)',marginTop:'3px',letterSpacing:'0.1em',textTransform:'uppercase'}}>Stress Index</div>
        <div style={{marginTop:'5px',fontSize:'8.5px',padding:'2px 7px',borderRadius:'9px',display:'inline-block',
          color:burnoutStatus.c,border:`1px solid ${burnoutStatus.c}44`,background:`${burnoutStatus.c}11`,transition:'all 0.5s'}}>
          {burnoutStatus.t}
        </div>
      </Card>

      {/* Session Timer */}
      <Card style={{background:'var(--bg-card2)'}}>
        <div style={{display:'flex', justifyContent:'space-between', alignItems:'flex-start'}}>
          <span className="card-hd">Session</span>
          <Timer size={14} style={{color:'var(--accent)',opacity:data.running?1:0.4}}/>
        </div>
        <div style={{marginTop:'6px'}}>
          <span style={{fontFamily:'var(--display)',fontSize:'28px',fontWeight:800,color:'var(--accent)',lineHeight:1,letterSpacing:'0.05em'}}>
            {timer}
          </span>
        </div>
        <div style={{fontSize:'9px',color:'var(--text2)',marginTop:'3px',letterSpacing:'0.1em',textTransform:'uppercase'}}>Active Time</div>
        <div style={{marginTop:'5px',fontSize:'8.5px',padding:'2px 7px',borderRadius:'9px',display:'inline-block',
          color:'var(--accent)',border:'1px solid var(--border-hi)',background:'var(--accent-d)'}}>
          {data.running ? '● Recording' : '○ Idle'}
        </div>
      </Card>

    </div>
  )
}

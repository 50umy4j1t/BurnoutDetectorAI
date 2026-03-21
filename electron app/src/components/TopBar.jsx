import { useState, useEffect } from 'react'
import { ChevronDown } from 'lucide-react'

function Clock() {
  const [t, setT] = useState(new Date())
  useEffect(()=>{ const id=setInterval(()=>setT(new Date()),1000); return ()=>clearInterval(id) },[])
  return <span className="clock">{t.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false})}</span>
}

export default function TopBar({ running, systemOk, systemError, models, selectedModel, onModelChange, onStart, onStop }) {
  return (
    <div className="topbar">
      <div className="logo">
        <div className="logo-sq">S</div>
        <span className="logo-name">
          <span style={{color:'var(--text-primary)'}}>Stress</span>
          <span style={{color:'var(--accent-primary)'}}>Lens</span>
          <span style={{fontFamily:'var(--font-body)',color:'var(--text-tertiary)',fontSize:'12px',marginLeft:'6px',fontWeight:400}}>AI</span>
        </span>
      </div>

      <div className="topbar-mid">
        <div className="sel-wrap">
          <select value={selectedModel} onChange={e=>onModelChange(e.target.value)} className="model-sel">
            {models.length>0
              ? models.map(m=><option key={m.choice} value={m.choice}>{m.name}</option>)
              : <option value={selectedModel}>Loading...</option>}
          </select>
          <ChevronDown size={11} className="sel-arrow" />
        </div>

        {!running
          ? <button onClick={onStart} className="pill pill-green"><span className="dot dot-g" />Start session</button>
          : <button onClick={onStop}  className="pill pill-red"><span className="dot dot-r dot-blink" />Stop session</button>}
      </div>

      <div className="topbar-right">
        <div
          className={systemOk ? 'sys-ok' : 'sys-err'}
          title={systemError||''}
          style={{cursor:systemError?'help':'default', maxWidth:220, overflow:'hidden', whiteSpace:'nowrap', textOverflow:'ellipsis'}}
        >
          <span className="dot" style={{background: systemOk?'var(--session-active)':'var(--stress-high)'}} />
          {systemOk ? 'Bridge connected' : `${systemError||'Bridge offline'}`}
        </div>
        <Clock />
      </div>
    </div>
  )
}

import { useState, useEffect } from 'react'
import { Shield, ShieldOff, Activity, AlertCircle, ChevronDown } from 'lucide-react'

function Clock() {
  const [t, setT] = useState(new Date())
  useEffect(()=>{ const id=setInterval(()=>setT(new Date()),1000); return ()=>clearInterval(id) },[])
  return <span className="clock">{t.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false})}</span>
}

export default function TopBar({ running, privacy, systemOk, systemError, models, selectedModel, onModelChange, onPrivacyToggle, onStart, onStop }) {
  return (
    <div className="topbar">
      <div className="logo">
        <div className="logo-sq">S</div>
        <span className="logo-name">
          <span style={{color:'var(--text)'}}>Stress</span>
          <span style={{color:'var(--accent)'}}>Lens</span>
          <span style={{color:'var(--text2)',fontSize:'11px',marginLeft:'5px',fontWeight:400}}> AI</span>
        </span>
      </div>

      <div className="topbar-mid">
        <div className="sel-wrap">
          <select value={selectedModel} onChange={e=>onModelChange(e.target.value)} className="model-sel">
            {models.length>0
              ? models.map(m=><option key={m.id} value={m.id}>{m.name}</option>)
              : <option value={selectedModel}>{selectedModel}</option>}
          </select>
          <ChevronDown size={11} className="sel-arrow" />
        </div>

        {!running
          ? <button onClick={onStart} className="pill pill-green"><span className="dot dot-g" />Launch Monitor</button>
          : <button onClick={onStop}  className="pill pill-red"><span className="dot dot-r dot-blink" />Stop Session</button>}

        <button onClick={onPrivacyToggle} className="pill pill-dim">
          {privacy ? <ShieldOff size={12}/> : <Shield size={12}/>}
          Privacy Mode {privacy?'ON':'OFF'}
        </button>
      </div>

      <div className="topbar-right">
        <div
          className={systemOk ? 'sys-ok' : 'sys-err'}
          title={systemError||''}
          style={{cursor:systemError?'help':'default', maxWidth:220, overflow:'hidden', whiteSpace:'nowrap', textOverflow:'ellipsis'}}
        >
          <span className="dot" style={{background: systemOk?'var(--accent)':'var(--red)', boxShadow:`0 0 7px ${systemOk?'var(--accent)':'var(--red)'}`}} />
          {systemOk ? 'All Systems Active' : `${systemError||'Backend Offline'}`}
        </div>
        <Clock />
      </div>
    </div>
  )
}

import { FileText, MessageSquare, Square, Heart } from 'lucide-react'

function fmt(s){ const m=Math.floor(s/60),sec=s%60; return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}` }

export default function BottomBar({ running, samples, elapsed, aiOpen, onToggleAI, onStart, onStop, onHeartCapture, heartBusy, onOpenReports }) {
  return (
    <div className="bottombar">
      <div className="bb-info">
        <div className="bb-name">
          StressLens AI
          <span style={{fontFamily:'var(--font-body)',color:'var(--text-tertiary)',fontSize:'11px',marginLeft:'8px',fontWeight:400,fontStyle:'italic'}}>
            v1.0.0
          </span>
        </div>
        <div className="bb-sub">Local processing &middot; No data leaves your machine</div>
      </div>

      <div className="bb-actions">
        <button className="bb-btn" onClick={onOpenReports}><FileText size={12}/>Open reports</button>
        <button className="bb-btn" onClick={onHeartCapture} disabled={heartBusy}>
          <Heart size={12}/>{heartBusy ? 'Capturing...' : 'Capture heart rate'}
        </button>
        <button className={`bb-btn ${aiOpen?'active':''}`} onClick={onToggleAI}>
          <MessageSquare size={12}/>
          {aiOpen ? 'Close advisor' : 'Ask advisor'}
        </button>
        {running
          ? <button className="bb-btn end" onClick={onStop}><Square size={11} fill="currentColor"/>End session</button>
          : <button className="bb-btn active" onClick={onStart}>Start session</button>}
      </div>

      <div className="bb-stats">
        <span><strong>{samples}</strong> <em>readings collected</em></span>
        <span style={{opacity:.3}}>|</span>
        <span><strong>{fmt(elapsed)}</strong> elapsed</span>
        <span style={{opacity:.3}}>|</span>
        <span style={{color: running?'var(--session-active)':'var(--text-tertiary)'}}>
          {running ? 'Recording' : 'Resting'}
        </span>
      </div>
    </div>
  )
}

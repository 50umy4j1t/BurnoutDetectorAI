import { FileText, MessageSquare, Square } from 'lucide-react'

function fmt(s){ const m=Math.floor(s/60),sec=s%60; return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}` }

export default function BottomBar({ running, samples, elapsed, aiOpen, onToggleAI, onStart, onStop }) {
  return (
    <div className="bottombar">
      <div className="bb-info">
        <div className="bb-name">
          <span style={{color:'var(--accent)'}}>Stress</span>Lens AI
          <span style={{color:'var(--text3)',fontSize:'10px',marginLeft:'8px',fontWeight:400}}> v1.0.0 — Ethical AI Surveillance System</span>
        </div>
        <div className="bb-sub">Local processing · No data leaves your machine</div>
      </div>

      <div className="bb-actions">
        <button className="bb-btn"><FileText size={12}/>Generate Report</button>
        <button className={`bb-btn ${aiOpen?'active':''}`} onClick={onToggleAI}>
          <MessageSquare size={12}/>
          {aiOpen ? 'Close AI Advisor' : 'Ask AI Advisor'}
        </button>
        {running
          ? <button className="bb-btn end" onClick={onStop}><Square size={11} fill="currentColor"/>End Session</button>
          : <button className="bb-btn active" onClick={onStart}>● Launch Session</button>}
      </div>

      <div className="bb-stats">
        <span><strong>{samples}</strong> samples</span>
        <span style={{opacity:.3}}>|</span>
        <span><strong>{fmt(elapsed)}</strong> elapsed</span>
        <span style={{opacity:.3}}>|</span>
        <span style={{color: running?'var(--accent)':'var(--text3)'}}>
          {running ? '● Recording' : '○ Idle'}
        </span>
      </div>
    </div>
  )
}

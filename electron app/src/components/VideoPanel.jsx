import { Video, MonitorPlay } from 'lucide-react'

function fmt(s){ const m=Math.floor(s/60),sec=s%60; return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}` }

export default function VideoPanel({ running, elapsed }) {
  return (
    <div className="card video-panel panel-hoverable">
      <div className="video-frame">
        {running ? (
          <>
            <div className="v-badge v-live">Recording</div>
            <div className="v-placeholder">
              <MonitorPlay size={28} style={{opacity:.4, color:'var(--accent-primary)'}} />
              <span style={{fontFamily:'var(--font-body)', fontSize:'12px', color:'var(--text-secondary)'}}>
                Camera running in separate window
              </span>
              <span style={{fontFamily:'var(--font-body)', fontSize:'11px', fontStyle:'italic', color:'var(--text-tertiary)'}}>
                Use R, S, P, Q in the camera window
              </span>
            </div>
          </>
        ) : (
          <div className="v-placeholder">
            <Video size={24} style={{opacity:.3, color:'var(--text-tertiary)'}} />
            <span style={{fontFamily:'var(--font-body)', fontSize:'13px', fontStyle:'italic', color:'var(--text-tertiary)'}}>
              Ready when you are
            </span>
          </div>
        )}
      </div>
      <div className="v-footer">
        <span>Session</span>
        <span className="v-timer">{fmt(elapsed)}</span>
      </div>
    </div>
  )
}

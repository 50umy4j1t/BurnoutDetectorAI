import { Video } from 'lucide-react'

function fmt(s){ const m=Math.floor(s/60),sec=s%60; return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}` }

export default function VideoPanel({ privacy, running, elapsed }) {
  return (
    <div className="card video-panel panel-hoverable">
      <div className="video-frame">
        {running ? (
          <>
            <img src="http://localhost:8000/api/video_feed" alt="Live" />
            <div className="scanline" />
            <div className="v-badge v-live">● LIVE</div>
            {privacy && <div className="v-badge v-blur">⬡ BLURRED</div>}
          </>
        ) : (
          <div className="v-placeholder">
            <Video size={26} style={{opacity:.2}} />
            <span style={{fontSize:'9.5px',letterSpacing:'0.14em',textTransform:'uppercase'}}>Session Inactive</span>
            <span style={{fontSize:'9px',opacity:.45}}>Launch Monitor to begin</span>
          </div>
        )}
      </div>
      <div className="v-footer">
        <span>Session Timer</span>
        <span className="v-timer">{fmt(elapsed)}</span>
      </div>
    </div>
  )
}

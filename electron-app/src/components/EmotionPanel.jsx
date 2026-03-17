import { useState } from 'react'

const EMOTION_COLORS = {
  neutral:  '#64748b',
  happy:    '#00d296',
  surprise: '#38b2f0',
  sad:      '#818cf8',
  angry:    '#ff3d5a',
  fear:     '#f472b6',
  disgust:  '#fb923c',
}

function getHeightPct(hoveredIdx, idx) {
  if (hoveredIdx === null) return 80
  const dist = Math.abs(hoveredIdx - idx)
  if (dist === 0) return 100
  if (dist === 1) return 88
  if (dist === 2) return 72
  if (dist === 3) return 60
  return 52
}

export default function EmotionPanel({ emotions, samples, bias }) {
  const [hovered, setHovered] = useState(null)
  const entries = Object.entries(emotions).sort(([,a],[,b]) => b - a)

  return (
    <div className="card emotion-col panel-hoverable"
      style={{display:'flex', flexDirection:'column', padding:'12px 14px', gap:'8px'}}>

      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
        <span className="card-hd">Emotion Breakdown</span>
        <span style={{fontSize:'9px',letterSpacing:'0.1em',textTransform:'uppercase',
          padding:'2px 8px',borderRadius:'20px',
          background:'var(--accent-d)',border:'1px solid var(--border-hi)',color:'var(--accent)'}}>
          DeepFace
        </span>
      </div>

      {/* Chart area */}
      <div style={{flex:1, display:'flex', gap:'5px', minHeight:0, alignItems:'stretch', overflow:'hidden'}}>
        {entries.map(([name, value], idx) => {
          const pct       = Math.round(value * 100)
          const color     = EMOTION_COLORS[name] ?? '#64748b'
          const isHovered = hovered === idx
          const heightPct = getHeightPct(hovered, idx)

          return (
            <div key={name}
              onMouseEnter={() => setHovered(idx)}
              onMouseLeave={() => setHovered(null)}
              style={{
                flex:1, display:'flex', flexDirection:'column',
                alignItems:'center', gap:'4px',
                height:`${heightPct}%`,
                alignSelf:'flex-end',
                transition:'height 0.3s cubic-bezier(0.4,0,0.2,1)',
                cursor:'default',
              }}>

              {/* % label */}
              <span style={{
                fontSize:'11px', fontFamily:'var(--display)', fontWeight:700,
                color: isHovered ? color : 'var(--text2)',
                lineHeight:1, flexShrink:0,
                transition:'color 0.2s',
              }}>{pct}%</span>

              {/* Track with fill from bottom */}
              <div style={{
                flex:1, width:'100%',
                background:'rgba(255,255,255,0.05)',
                borderRadius:'5px',
                border:`1px solid ${isHovered ? color+'55' : color+'18'}`,
                overflow:'hidden',
                display:'flex', flexDirection:'column', justifyContent:'flex-end',
                transition:'border-color 0.25s',
              }}>
                <div style={{
                  width:'100%',
                  height:`${Math.max(pct, 1)}%`,
                  background: isHovered
                    ? `linear-gradient(to top, ${color}, ${color}99)`
                    : `linear-gradient(to top, ${color}88, ${color}44)`,
                  borderRadius:'4px',
                  boxShadow: isHovered ? `0 0 14px ${color}88` : 'none',
                  transition:'height 0.7s cubic-bezier(0.4,0,0.2,1), background 0.25s, box-shadow 0.25s',
                }}/>
              </div>

              {/* Name */}
              <span style={{
                fontSize:'8px',
                color: isHovered ? color : 'var(--text2)',
                textTransform:'capitalize', textAlign:'center',
                lineHeight:1.2, whiteSpace:'nowrap', flexShrink:0,
                transition:'color 0.2s',
              }}>{name}</span>
            </div>
          )
        })}
      </div>

      <div style={{display:'flex',justifyContent:'space-between',
        fontSize:'9px',color:'var(--text3)',paddingTop:'4px',borderTop:'1px solid var(--border)'}}>
        <span>Bias: <strong style={{color:'var(--text2)'}}>{bias}</strong></span>
        <span>{samples} samples</span>
      </div>
    </div>
  )
}

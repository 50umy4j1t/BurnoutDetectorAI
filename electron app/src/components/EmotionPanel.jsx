import { useState } from 'react'

const EMOTION_COLORS = {
  neutral:  'var(--emotion-neutral)',
  happy:    'var(--emotion-happy)',
  surprise: 'var(--emotion-surprise)',
  sad:      'var(--emotion-sad)',
  angry:    'var(--emotion-angry)',
  fear:     'var(--emotion-fear)',
  disgust:  'var(--emotion-disgust)',
}

const RAW_COLORS = {
  neutral:  '#8BB0CC',
  happy:    '#A8C4A0',
  surprise: '#A0C4B8',
  sad:      '#C4B5D4',
  angry:    '#D4A48A',
  fear:     '#B0A8C4',
  disgust:  '#C4B0A0',
}

export default function EmotionPanel({ emotions, samples, bias }) {
  const [hovered, setHovered] = useState(null)
  const entries = Object.entries(emotions).sort(([,a],[,b]) => b - a)
  const maxVal = Math.max(...entries.map(([,v]) => v), 0.01)

  return (
    <div className="card emotion-col panel-hoverable"
      style={{display:'flex', flexDirection:'column', padding:'var(--card-padding)', gap:'12px'}}>

      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
        <span className="card-hd">Emotion breakdown</span>
        <span style={{fontFamily:'var(--font-body)', fontSize:'11px', fontStyle:'italic', color:'var(--text-tertiary)'}}>
          via DeepFace
        </span>
      </div>

      {/* Horizontal bar chart */}
      <div style={{flex:1, display:'flex', flexDirection:'column', gap:'6px', justifyContent:'center', minHeight:0, overflow:'auto'}}>
        {entries.map(([name, value], idx) => {
          const pct = Math.round(value * 100)
          const rawColor = RAW_COLORS[name] ?? '#8BB0CC'
          const isHovered = hovered === idx
          const barWidth = (value / maxVal) * 100

          return (
            <div key={name}
              onMouseEnter={() => setHovered(idx)}
              onMouseLeave={() => setHovered(null)}
              style={{
                display:'flex', alignItems:'center', gap:'10px',
                cursor:'default', padding:'3px 0',
              }}>

              <span style={{
                fontFamily:'var(--font-body)', fontSize:'12px',
                color: isHovered ? rawColor : 'var(--text-secondary)',
                width:'70px', flexShrink:0, textTransform:'capitalize',
                transition:'color 150ms ease',
              }}>{name}</span>

              <div style={{
                flex:1, height:'10px', borderRadius:'4px',
                background:'var(--bg-raised)', overflow:'hidden',
              }}>
                <div style={{
                  width:`${Math.max(barWidth, 2)}%`, height:'100%', borderRadius:'4px',
                  background: rawColor,
                  opacity: isHovered ? 0.8 : 0.45,
                  transition:'width 0.7s cubic-bezier(0.4,0,0.2,1), opacity 150ms ease',
                }}/>
              </div>

              <span style={{
                fontFamily:'var(--font-body)', fontSize:'12px', fontWeight:500,
                color: isHovered ? rawColor : 'var(--text-secondary)',
                width:'36px', textAlign:'right', flexShrink:0,
                transition:'color 150ms ease',
              }}>{pct}%</span>
            </div>
          )
        })}
      </div>

      <div style={{display:'flex', justifyContent:'space-between',
        fontFamily:'var(--font-body)', fontSize:'11px', color:'var(--text-tertiary)',
        paddingTop:'8px', borderTop:'0.5px solid var(--border-subtle)'}}>
        <span>Bias: <strong style={{color:'var(--text-secondary)', fontWeight:500}}>{bias}</strong></span>
        <span><em>{samples} readings collected</em></span>
      </div>
    </div>
  )
}

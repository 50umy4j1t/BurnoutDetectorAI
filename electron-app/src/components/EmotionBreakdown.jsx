const EMOTION_COLORS = {
  neutral:  '#64748b',
  happy:    '#00d296',
  surprise: '#38b2f0',
  sad:      '#818cf8',
  angry:    '#ff3d5a',
  fear:     '#f472b6',
  disgust:  '#fb923c',
}

export default function EmotionBreakdown({ emotions, samples, bias, elapsed }) {
  const sorted = Object.entries(emotions).sort(([,a],[,b]) => b - a)

  return (
    <div className="card" style={{flex:1, display:'flex', flexDirection:'column', padding:'12px 14px', gap:'6px', minWidth:0}}>
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'2px'}}>
        <span className="card-hd">Emotion Breakdown</span>
        <span style={{
          fontSize:'9px', letterSpacing:'0.1em', textTransform:'uppercase',
          padding:'2px 8px', borderRadius:'20px',
          background:'var(--accent-d)', border:'1px solid var(--border-hi)', color:'var(--accent)'
        }}>DeepFace</span>
      </div>

      <div style={{display:'flex', flexDirection:'column', gap:'6px', flex:1, justifyContent:'center'}}>
        {sorted.map(([name, value]) => {
          const pct = Math.round(value * 100)
          const color = EMOTION_COLORS[name] ?? '#64748b'
          return (
            <div key={name} style={{display:'flex', alignItems:'center', gap:'8px'}}>
              {/* Emotion name */}
              <span style={{
                fontSize:'9.5px', textTransform:'capitalize', letterSpacing:'0.04em',
                color:'var(--text2)', width:'52px', textAlign:'right', flexShrink:0
              }}>{name}</span>

              {/* Bar track */}
              <div style={{
                flex:1, height:'5px', background:'rgba(255,255,255,0.05)',
                borderRadius:'3px', overflow:'hidden'
              }}>
                <div style={{
                  height:'100%', borderRadius:'3px',
                  width:`${pct}%`,
                  background:`linear-gradient(90deg, ${color}88, ${color})`,
                  boxShadow: pct > 35 ? `0 0 8px ${color}55` : 'none',
                  transition:'width 0.7s cubic-bezier(0.4,0,0.2,1)',
                }}/>
              </div>

              {/* Percentage */}
              <span style={{fontSize:'9px', color:'var(--text3)', width:'28px', textAlign:'right', flexShrink:0, fontFamily:'var(--mono)'}}>
                {pct}%
              </span>
            </div>
          )
        })}
      </div>

      <div style={{
        display:'flex', justifyContent:'space-between',
        fontSize:'9px', color:'var(--text3)',
        paddingTop:'5px', borderTop:'1px solid var(--border)'
      }}>
        <span>Bias: <strong style={{color:'var(--text2)'}}>{bias}</strong></span>
        <span>{samples} samples</span>
      </div>
    </div>
  )
}

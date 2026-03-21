function getColor(v) {
  if (v < 25) return 'var(--stress-low)'
  if (v < 50) return 'var(--stress-moderate)'
  if (v < 70) return 'var(--stress-elevated)'
  return 'var(--stress-high)'
}
function getBg(v) {
  if (v < 25) return 'var(--stress-low-bg)'
  if (v < 50) return 'var(--stress-moderate-bg)'
  if (v < 70) return 'var(--stress-elevated-bg)'
  return 'var(--stress-high-bg)'
}
function getLabel(v) {
  if (v < 25) return 'Looking good'
  if (v < 50) return 'Moderate'
  if (v < 70) return 'Elevated'
  return 'Needs attention'
}

export default function BurnoutGauge({ burnout, running }) {
  const val   = Math.min(100, Math.max(0, burnout))
  const color = getColor(val)
  const bg    = getBg(val)
  const label = getLabel(val)

  return (
    <div className="card burnout-panel panel-hoverable">
      <div className="burnout-hd">
        <span className="card-hd">Burnout score</span>
        <span className="rt-badge">
          {running ? <em>Real-time</em> : <em>Based on your reports</em>}
        </span>
      </div>

      <div className="gauge-area" style={{flexDirection:'column', gap:'16px'}}>
        {/* Large score number */}
        <div style={{textAlign:'center'}}>
          <div style={{
            fontFamily:'var(--font-heading)', fontSize:'48px', fontWeight:700,
            color:'var(--text-primary)', lineHeight:1,
            transition:'color 0.5s',
          }}>
            {Math.round(val)}
          </div>
        </div>

        {/* Progress bar */}
        <div style={{width:'80%', maxWidth:'220px'}}>
          <div style={{
            width:'100%', height:'6px', borderRadius:'99px',
            background:'var(--bg-raised)', overflow:'hidden',
          }}>
            <div style={{
              width:`${val}%`, height:'100%', borderRadius:'99px',
              background:'linear-gradient(90deg, #7AAB8A, #C4B06A, #D4A06A, #C88A7A)',
              transition:'width 0.9s cubic-bezier(0.4,0,0.2,1)',
            }}/>
          </div>
        </div>

        {/* Status pill */}
        <div style={{
          fontFamily:'var(--font-body)', fontSize:'11px', fontWeight:500,
          padding:'4px 14px', borderRadius:'99px',
          color, background:bg,
          transition:'all 0.5s',
        }}>
          {label}
        </div>
      </div>
    </div>
  )
}

function polar(cx, cy, r, deg) {
  const rad = (deg * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

// Arc goes from 225° to 315° (270° sweep, gap at bottom)
const START = 225
const SWEEP = 270
const CX = 160, CY = 145
const R_ARC  = 122
const R_TICK = 102

const SEGMENTS = [
  { from:0,  to:25,  color:'#00d296' },
  { from:25, to:50,  color:'#a8e063' },
  { from:50, to:70,  color:'#f5c842' },
  { from:70, to:85,  color:'#ff8c42' },
  { from:85, to:100, color:'#ff3d5a' },
]

function getColor(v) {
  if (v < 25) return '#00d296'
  if (v < 50) return '#a8e063'
  if (v < 70) return '#f5c842'
  if (v < 85) return '#ff8c42'
  return '#ff3d5a'
}
function getLabel(v) {
  if (v < 25) return 'NEUTRAL'
  if (v < 50) return 'LOW RISK'
  if (v < 70) return 'MODERATE'
  if (v < 85) return 'HIGH RISK'
  return 'CRITICAL'
}

function arcD(r, fromVal, toVal) {
  const a1 = START + (fromVal / 100) * SWEEP
  const a2 = START + (toVal   / 100) * SWEEP
  const p1 = polar(CX, CY, r, a1)
  const p2 = polar(CX, CY, r, a2)
  const lg = (a2 - a1) > 180 ? 1 : 0
  return `M ${p1.x} ${p1.y} A ${r} ${r} 0 ${lg} 1 ${p2.x} ${p2.y}`
}

export default function BurnoutGauge({ burnout, running }) {
  const val   = Math.min(100, Math.max(0, burnout))
  const color = getColor(val)
  const label = getLabel(val)
  const needleAngle = START + (val / 100) * SWEEP

  // Major ticks every 20, minor every 10
  const ticks = []
  for (let v = 0; v <= 100; v += 10) {
    const major = v % 20 === 0
    const ang   = START + (v / 100) * SWEEP
    const rOuter = R_ARC - 2
    const rInner = R_ARC - (major ? 12 : 8)
    const rLabel = R_ARC - 22
    ticks.push({ v, ang, major, rOuter, rInner, rLabel })
  }

  return (
    <div className="card burnout-panel panel-hoverable" style={{overflow:'visible'}}>
      <div className="burnout-hd">
        <span className="card-hd">Burnout Risk</span>
        <span className="rt-badge">{running ? 'Real-time' : 'Report-derived'}</span>
      </div>

      <div className="gauge-area" style={{flexDirection:'column', flex:1, width:'100%'}}>
        <svg viewBox="0 0 320 310"
          style={{width:'100%', height:'100%', overflow:'visible', flexShrink:0}}>
          <defs>
            <filter id="gneedle">
              <feGaussianBlur stdDeviation="3" result="b"/>
              <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <radialGradient id="bggrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%"  stopColor="#1a2d45"/>
              <stop offset="100%" stopColor="#07111e"/>
            </radialGradient>
          </defs>

          {/* Background circle */}
          <circle cx={CX} cy={CY} r={R_ARC + 16}
            fill="url(#bggrad)" stroke="rgba(255,255,255,0.06)" strokeWidth={1}/>

          {/* Smooth spectrum arc — 100 tiny slices for continuous gradient */}
          {Array.from({length: 100}, (_, i) => {
            const a1  = START + (i / 100) * SWEEP
            const a2  = START + ((i+1) / 100) * SWEEP
            const p1  = polar(CX, CY, R_ARC, a1)
            const p2  = polar(CX, CY, R_ARC, a2)
            // green(0) → lime(25) → yellow(50) → orange(75) → red(100)
            const t = i / 100
            let r, g, b
            if (t < 0.25) {
              const s = t / 0.25
              r = Math.round(0   + s * 168); g = Math.round(210 + s * 14);  b = Math.round(150 - s * 150)
            } else if (t < 0.5) {
              const s = (t - 0.25) / 0.25
              r = Math.round(168 + s * 77);  g = Math.round(224 - s * 24);  b = 0
            } else if (t < 0.75) {
              const s = (t - 0.5) / 0.25
              r = Math.round(245 + s * 10);  g = Math.round(200 - s * 60);  b = 0
            } else {
              const s = (t - 0.75) / 0.25
              r = 255; g = Math.round(140 - s * 140); b = Math.round(s * 20)
            }
            const isActive = i < val
            return (
              <path key={i}
                d={`M ${p1.x} ${p1.y} A ${R_ARC} ${R_ARC} 0 0 1 ${p2.x} ${p2.y}`}
                fill="none"
                stroke={`rgb(${r},${g},${b})`}
                strokeWidth={14}
                strokeLinecap="butt"
                opacity={isActive ? 0.9 : 0.15}
              />
            )
          })}

          {/* Tick marks */}
          {ticks.map(t => {
            const p1 = polar(CX, CY, t.rOuter, t.ang)
            const p2 = polar(CX, CY, t.rInner, t.ang)
            const lp = polar(CX, CY, t.rLabel, t.ang)
            return (
              <g key={t.v}>
                <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
                  stroke={t.major ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.2)'}
                  strokeWidth={t.major ? 2 : 1}/>
                {t.major && (
                  <text x={lp.x} y={lp.y + 4} textAnchor="middle"
                    style={{fontFamily:'var(--mono)', fontSize:'9px', fill:'rgba(204,232,220,0.5)'}}>
                    {t.v}
                  </text>
                )}
              </g>
            )
          })}

          {/* Inner glow ring */}
          <circle cx={CX} cy={CY} r={72}
            fill="none" stroke={color} strokeWidth={1} opacity={0.15}
            style={{transition:'stroke 0.5s'}}/>

          {/* Arrow — curved base on inner circle, tip points to scale */}
          <g transform={`rotate(${needleAngle - 270} ${CX} ${CY})`}
            style={{transition:'transform 0.9s cubic-bezier(0.4,0,0.2,1)'}}>
            <path
              d={`M ${CX-22},${CY-70} A 72,72 0 0,1 ${CX+22},${CY-70} L ${CX},${CY-84} Z`}
              fill={color}
              filter="url(#gneedle)"
              style={{transition:'fill 0.5s'}}
            />
          </g>

          {/* Score — dead center inside the circle */}
          <text x={CX} y={CY + 8} textAnchor="middle"
            style={{
              fontFamily:'var(--display)', fontSize:'64px', fontWeight:800,
              fill:color, transition:'fill 0.5s',
              filter:`drop-shadow(0 0 22px ${color}cc)`
            }}>
            {Math.round(val)}
          </text>
          <text x={CX} y={CY + 28} textAnchor="middle"
            style={{fontFamily:'var(--mono)', fontSize:'9px', fill:'rgba(204,232,220,0.4)', letterSpacing:'0.14em'}}>
            BURNOUT SCORE
          </text>
        </svg>

        <div style={{textAlign:'center', marginTop:'-18px'}}>
          <span style={{fontFamily:'var(--display)', fontSize:'14px', fontWeight:700,
            color, letterSpacing:'0.1em', transition:'color 0.5s'}}>
            {label}
          </span>
        </div>
      </div>
    </div>
  )
}

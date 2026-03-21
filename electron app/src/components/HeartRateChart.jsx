import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{background:'var(--bg-raised)',border:'0.5px solid var(--border-default)',borderRadius:'8px',padding:'6px 12px',fontFamily:'var(--font-mono)',fontSize:'12px',fontWeight:300}}>
      <span style={{color:'var(--stress-high)'}}>{payload[0]?.value} bpm</span>
    </div>
  )
}

export default function HeartRateChart({ history, running }) {
  const avg = history.length ? Math.round(history.reduce((s,p)=>s+p.bpm,0)/history.length) : 0

  return (
    <div className="card chart-panel panel-hoverable">
      <div className="chart-hd">
        <span className="card-hd">Heart rate history</span>
        <div style={{display:'flex',alignItems:'center',gap:'12px'}}>
          {avg > 0 && <span style={{fontFamily:'var(--font-body)',fontSize:'11px',color:'var(--text-secondary)'}}>Avg: <strong style={{color:'var(--stress-high)',fontWeight:500}}>{avg} bpm</strong></span>}
          <div style={{display:'flex',alignItems:'center',gap:'5px',fontFamily:'var(--font-body)',fontSize:'11px',color:'var(--text-secondary)'}}>
            <div style={{width:'16px',height:'2px',borderRadius:'1px',background:'var(--stress-high)'}}/>bpm
          </div>
        </div>
      </div>

      <div className="chart-body">
        {history.length === 0 ? (
          <div style={{display:'flex',alignItems:'center',justifyContent:'center',height:'100%',
            fontFamily:'var(--font-body)',fontStyle:'italic',color:'var(--text-tertiary)',fontSize:'13px'}}>
            No heart readings yet. Use the capture button to begin.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={history} margin={{top:8,right:8,left:-22,bottom:0}}>
              <defs>
                <linearGradient id="bpmG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#C88A7A" stopOpacity={0.2}/>
                  <stop offset="95%" stopColor="#C88A7A" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(232,228,220,0.04)" strokeDasharray="3 3" vertical={false}/>
              <XAxis dataKey="t" tick={{fill:'var(--text-tertiary)',fontSize:9,fontFamily:'var(--font-mono)'}}
                tickLine={false} axisLine={false} interval="preserveStartEnd"/>
              <YAxis tick={{fill:'var(--text-tertiary)',fontSize:9,fontFamily:'var(--font-mono)'}}
                tickLine={false} axisLine={false}/>
              <Tooltip content={<CustomTooltip/>} cursor={{stroke:'rgba(200,138,122,0.2)',strokeWidth:1}}/>
              {avg>0 && <ReferenceLine y={avg} stroke="rgba(200,138,122,0.2)" strokeDasharray="4 4"/>}
              <Area type="monotoneX" dataKey="bpm" stroke="#C88A7A" strokeWidth={1.5} fill="url(#bpmG)" dot={false} activeDot={{r:3,fill:'#C88A7A'}} isAnimationActive={true} animationDuration={300}/>
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}

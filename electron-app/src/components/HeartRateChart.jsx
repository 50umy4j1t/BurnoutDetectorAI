import { AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{background:'var(--bg-card2)',border:'1px solid var(--border-hi)',borderRadius:'6px',padding:'6px 10px',fontFamily:'var(--mono)',fontSize:'10px'}}>
      <span style={{color:'var(--red)'}}>{payload[0]?.value} BPM</span>
    </div>
  )
}

export default function HeartRateChart({ history, running }) {
  const avg = history.length ? Math.round(history.reduce((s,p)=>s+p.bpm,0)/history.length) : 0

  return (
    <div className="card chart-panel panel-hoverable">
      <div className="chart-hd">
        <span className="card-hd">Heart Rate History</span>
        <div style={{display:'flex',alignItems:'center',gap:'12px'}}>
          {running && <span style={{fontSize:'9.5px',color:'var(--text2)'}}>Avg: <strong style={{color:'var(--red)'}}>{avg} BPM</strong></span>}
          <div style={{display:'flex',alignItems:'center',gap:'5px',fontSize:'9.5px',color:'var(--text2)'}}>
            <div style={{width:'16px',height:'2px',borderRadius:'1px',background:'var(--red)'}}/>BPM
          </div>
        </div>
      </div>

      <div className="chart-body">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={history} margin={{top:8,right:8,left:-22,bottom:0}}>
            <defs>
              <linearGradient id="bpmG" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#ff3d5a" stopOpacity={0.28}/>
                <stop offset="95%" stopColor="#ff3d5a" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" vertical={false}/>
            <XAxis dataKey="t" tick={{fill:'rgba(204,232,220,0.22)',fontSize:8,fontFamily:'var(--mono)'}}
              tickLine={false} axisLine={false} tickFormatter={v=>`${v}s`} interval="preserveStartEnd"/>
            <YAxis tick={{fill:'rgba(204,232,220,0.22)',fontSize:8,fontFamily:'var(--mono)'}}
              tickLine={false} axisLine={false}/>
            <Tooltip content={<CustomTooltip/>} cursor={{stroke:'rgba(255,61,90,0.25)',strokeWidth:1}}/>
            {avg>0 && <ReferenceLine y={avg} stroke="rgba(255,61,90,0.25)" strokeDasharray="4 4"/>}
            <Area type="monotoneX" dataKey="bpm" stroke="#ff3d5a" strokeWidth={2} fill="url(#bpmG)" dot={false} activeDot={{r:3,fill:'#ff3d5a'}} isAnimationActive={true} animationDuration={300}/>
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

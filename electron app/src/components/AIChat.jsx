import { useState, useRef, useEffect } from 'react'
import { Heart, Volume2, VolumeX, Loader } from 'lucide-react'

const WELCOME = {
  role:'ai',
  text:"Hey there! I'm your wellness advisor, running fully offline via Ollama. I'll keep an eye on your stress levels and check in when things get intense. Just focus on your work \u2014 I've got your back.",
  ts: Date.now(),
}

export default function AIChat({ open, onClose, api, modelName, ttsInfo, onTtsToggle, onHeartCapture, heartBusy, injectMessages }) {
  const [messages, setMessages] = useState([WELCOME])
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [includeReport, setIncludeReport] = useState(true)
  const bottomRef = useRef(null)
  const inputRef  = useRef(null)

  // TTS state comes from bridge tts-status events via ttsInfo
  const ttsState = ttsInfo?.state || 'ready'
  const ttsMessage = ttsInfo?.message || ''
  const isTtsBusy = ['loading', 'generating', 'speaking'].includes(ttsState)

  useEffect(()=>{ bottomRef.current?.scrollIntoView({behavior:'smooth'}) },[messages, loading])
  useEffect(()=>{ if(open) setTimeout(()=>inputRef.current?.focus(), 450) },[open])

  const consumedRef = useRef(0)
  useEffect(()=>{
    if (!injectMessages || injectMessages.length <= consumedRef.current) return
    const fresh = injectMessages.slice(consumedRef.current)
    consumedRef.current = injectMessages.length
    setMessages(p => [...p, ...fresh])
    // TTS is handled by the Python backend — no frontend speak calls needed
  },[injectMessages])

  const addMsg = (role, text) => setMessages(p=>[...p,{role,text,ts:Date.now()}])

  const send = async () => {
    const txt = input.trim()
    if (!txt || loading || !api) return
    setInput(''); addMsg('user', txt); setLoading(true)
    try {
      const payload = await api.sendChat(txt, includeReport, undefined)
      const reply = payload.response || 'Sorry, no response.'
      addMsg('ai', reply)
      // Backend handles TTS for chat replies too (ttsQueued in response)
    } catch(e) { addMsg('ai', `Could not reach the bridge: ${e.message}`) }
    setLoading(false)
  }

  return (
    <div className={`ai-panel ${open ? 'open' : ''}`}>
      <div className="ai-hd">
        <div className="ai-hd-row">
          <div className="ai-dot" />
          <span className="ai-title">Wellness advisor</span>
          <button className="ai-collapse" onClick={onClose}>Close</button>
        </div>
        <div className="ai-sub">Powered by Ollama ({modelName}) &mdash; running locally</div>
      </div>

      <div className="ai-msgs">
        {messages.map((m,i)=>(
          <div key={i} className={m.role==='system' ? 'msg-sys' : m.role==='ai' ? 'msg-ai' : 'msg-you'}>
            <div className="msg-lbl">{m.role==='ai' ? 'Advisor' : m.role==='system' ? 'System' : 'You'}</div>
            {m.text}
          </div>
        ))}
        {loading && (
          <div className="msg-ai">
            <div className="msg-lbl">Advisor</div>
            <div className="typing"><span/><span/><span/></div>
          </div>
        )}
        <div ref={bottomRef}/>
      </div>

      <div className="ai-foot">
        {/* TTS status indicator — driven by backend tts-status events */}
        {ttsInfo?.enabled && isTtsBusy && (
          <div style={{
            display:'flex', alignItems:'center', gap:'8px',
            padding:'6px 12px', borderRadius:'var(--radius-pill)',
            background:'rgba(122,170,181,0.08)',
            border:'0.5px solid rgba(122,170,181,0.2)',
            fontFamily:'var(--font-body)', fontSize:'11px', fontStyle:'italic',
            color:'var(--accent-primary)',
          }}>
            <Loader size={12} style={{animation:'spin 1s linear infinite'}}/>
            <span>{ttsMessage || ttsState}</span>
            <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
          </div>
        )}
        {ttsInfo?.enabled && ttsState === 'error' && (
          <div style={{
            display:'flex', alignItems:'center', gap:'8px',
            padding:'6px 12px', borderRadius:'var(--radius-pill)',
            background:'var(--stress-high-bg)',
            border:'0.5px solid rgba(200,138,122,0.3)',
            fontFamily:'var(--font-body)', fontSize:'11px', fontStyle:'italic',
            color:'var(--stress-high)',
          }}>
            <span>{ttsMessage || 'TTS error'}</span>
          </div>
        )}

        <div style={{display:'flex',gap:'6px',flexWrap:'wrap'}}>
          <button className="bb-btn" style={{flex:1,justifyContent:'center'}}
            onClick={onHeartCapture} disabled={heartBusy}>
            <Heart size={11}/>{heartBusy ? 'Capturing...' : 'Capture heart rate'}
          </button>
          <button className={`bb-btn ${ttsInfo?.enabled?'active':''}`} style={{flex:1,justifyContent:'center'}}
            onClick={onTtsToggle}>
            {ttsInfo?.enabled ? <Volume2 size={11}/> : <VolumeX size={11}/>}
            TTS {ttsInfo?.enabled ? 'on' : 'off'}
            {ttsInfo?.enabled && isTtsBusy && (
              <Loader size={10} style={{animation:'spin 1s linear infinite', marginLeft:'2px'}}/>
            )}
          </button>
          <label style={{display:'flex',alignItems:'center',gap:'5px',fontFamily:'var(--font-body)',fontSize:'11px',color:'var(--text-secondary)',cursor:'pointer',padding:'0 4px'}}>
            <input type="checkbox" checked={includeReport} onChange={e=>setIncludeReport(e.target.checked)}
              style={{accentColor:'var(--accent-primary)',width:'12px',height:'12px'}}/>
            Include report
          </label>
        </div>
        <div className="ai-input-row">
          <input
            ref={inputRef}
            className="ai-input"
            placeholder="Ask your wellness advisor..."
            value={input}
            onChange={e=>setInput(e.target.value)}
            onKeyDown={e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()} }}
            disabled={loading}
          />
          <button className="ai-send" onClick={send} disabled={loading||!input.trim()}>Send</button>
        </div>
      </div>
    </div>
  )
}

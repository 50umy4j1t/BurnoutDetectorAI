import { useState, useRef, useEffect } from 'react'
import { FileText, MessageSquare } from 'lucide-react'

const API = 'http://localhost:8000'

const WELCOME = {
  role:'ai',
  text:"Hey there! I'm your AI Wellness Advisor, powered by Ollama — running fully offline. I'll monitor your stress levels and check in when things get intense. Just focus on your work — I've got your back. 💚"
}

export default function AIChat({ open, onClose, sessionData }) {
  const [messages, setMessages] = useState([WELCOME])
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const bottomRef = useRef(null)
  const inputRef  = useRef(null)

  useEffect(()=>{ bottomRef.current?.scrollIntoView({behavior:'smooth'}) },[messages, loading])
  useEffect(()=>{ if(open) setTimeout(()=>inputRef.current?.focus(), 450) },[open])

  const addMsg = (role, text) => setMessages(p=>[...p,{role,text,ts:Date.now()}])

  const send = async () => {
    const txt = input.trim()
    if (!txt || loading) return
    setInput(''); addMsg('user', txt); setLoading(true)
    try {
      const r = await fetch(`${API}/api/chat`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:txt})})
      const d = await r.json()
      addMsg('ai', d.response||'Sorry, no response.')
    } catch { addMsg('ai',"Couldn't reach backend right now.") }
    setLoading(false)
  }

  const getSupport = async () => {
    setLoading(true)
    try {
      const r = await fetch(`${API}/api/support`,{method:'POST'})
      const d = await r.json()
      addMsg('ai', d.response||'Report generated.')
    } catch { addMsg('ai',"Couldn't generate support report.") }
    setLoading(false)
  }

  const genReport = async () => {
    setLoading(true)
    try {
      await fetch(`${API}/api/report`,{method:'POST'})
      addMsg('ai','Session report generated ✓')
    } catch { addMsg('ai',"Couldn't generate report.") }
    setLoading(false)
  }

  return (
    <div className={`ai-panel ${open ? 'open' : ''}`}>
      {/* Header */}
      <div className="ai-hd">
        <div className="ai-hd-row">
          <div className="ai-dot" />
          <span className="ai-title">AI Wellness Advisor</span>
          <button className="ai-collapse" onClick={onClose}>✕ Close</button>
        </div>
        <div className="ai-sub">Powered by Ollama ({sessionData.model}) — running locally</div>
      </div>

      {/* Messages */}
      <div className="ai-msgs">
        {messages.map((m,i)=>(
          <div key={i} className={m.role==='ai' ? 'msg-ai' : 'msg-you'}>
            <div className="msg-lbl">{m.role==='ai' ? 'StressLens AI' : 'You'}</div>
            {m.text}
          </div>
        ))}
        {loading && (
          <div className="msg-ai">
            <div className="msg-lbl">StressLens AI</div>
            <div className="typing"><span/><span/><span/></div>
          </div>
        )}
        <div ref={bottomRef}/>
      </div>

      {/* Footer */}
      <div className="ai-foot">
        <div style={{display:'flex',gap:'6px'}}>
          <button className="bb-btn" style={{flex:1,justifyContent:'center'}} onClick={genReport} disabled={loading}>
            <FileText size={11}/>Report
          </button>
          <button className="bb-btn" style={{flex:1,justifyContent:'center'}} onClick={getSupport} disabled={loading}>
            <MessageSquare size={11}/>AI Support
          </button>
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

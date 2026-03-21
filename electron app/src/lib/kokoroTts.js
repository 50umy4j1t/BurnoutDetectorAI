import { KokoroTTS } from 'kokoro-js'

const MODEL_ID = 'onnx-community/Kokoro-82M-v1.0-ONNX'
const DEFAULT_VOICE = 'af_heart'

let enginePromise = null
let speechQueue = Promise.resolve(false)
let audioContext = null
let activeSource = null
let primed = false

// Observable TTS state for UI feedback
let _stateCallback = null
let _currentState = 'idle' // idle | loading-model | generating | playing | error

function setState(state, detail = '') {
  _currentState = state
  if (_stateCallback) _stateCallback(state, detail)
}

export function onTtsStateChange(cb) {
  _stateCallback = cb
  return () => { if (_stateCallback === cb) _stateCallback = null }
}

export function getTtsState() {
  return _currentState
}

function normalizeText(text) {
  const cleaned = String(text || '').replace(/\s+/g, ' ').trim()
  if (!cleaned) return ''
  return cleaned.slice(0, 900)
}

async function ensureEngine(progress_callback) {
  if (!enginePromise) {
    setState('loading-model', 'Downloading TTS model (first time only)...')
    console.log('[tts] Loading Kokoro model:', MODEL_ID)
    enginePromise = KokoroTTS.from_pretrained(MODEL_ID, {
      dtype: 'q8',
      device: 'wasm',
      progress_callback: (p) => {
        if (p?.status === 'progress' && p.total) {
          const pct = Math.round((p.loaded / p.total) * 100)
          setState('loading-model', `Downloading model: ${pct}%`)
        }
        if (progress_callback) progress_callback(p)
      },
    })
    enginePromise.then(() => {
      console.log('[tts] Model loaded successfully')
      setState('idle')
    }).catch((err) => {
      console.error('[tts] Model load failed:', err)
      enginePromise = null
      setState('error', `Model load failed: ${err.message}`)
    })
  }
  return enginePromise
}

function getAudioContext() {
  if (audioContext && audioContext.state !== 'closed') return audioContext
  const Ctx = window.AudioContext || window.webkitAudioContext
  if (!Ctx) {
    throw new Error('WebAudio is not available in this renderer')
  }
  audioContext = new Ctx()
  console.log('[tts] AudioContext created, state:', audioContext.state)
  return audioContext
}

async function ensureAudioContextUnlocked() {
  const ctx = getAudioContext()
  if (ctx.state === 'suspended') {
    console.log('[tts] Resuming suspended AudioContext...')
    await ctx.resume()
    console.log('[tts] AudioContext resumed, state:', ctx.state)
  }
  return ctx
}

export function primeSpeechPlayback() {
  if (primed || typeof window === 'undefined') return
  primed = true

  const unlock = async () => {
    try {
      const ctx = await ensureAudioContextUnlocked()
      console.log('[tts] AudioContext unlocked via user gesture, state:', ctx.state)
    } catch (err) {
      console.warn('[tts] AudioContext unlock failed:', err)
    } finally {
      window.removeEventListener('pointerdown', unlock, true)
      window.removeEventListener('keydown', unlock, true)
      window.removeEventListener('touchstart', unlock, true)
    }
  }

  window.addEventListener('pointerdown', unlock, true)
  window.addEventListener('keydown', unlock, true)
  window.addEventListener('touchstart', unlock, true)
}

async function playRawAudio(rawAudio) {
  const ctx = await ensureAudioContextUnlocked()
  const samples = rawAudio?.audio
  const sampleRate = Number(rawAudio?.sampling_rate || 24000)

  if (!samples || !samples.length) {
    throw new Error('Generated audio buffer is empty')
  }

  console.log('[tts] Playing audio: samples=%d, sampleRate=%d, duration=%.1fs',
    samples.length, sampleRate, samples.length / sampleRate)

  // Handle Float32Array or regular array
  let float32Samples
  if (samples instanceof Float32Array) {
    float32Samples = samples
  } else if (samples.data && samples.data instanceof Float32Array) {
    float32Samples = samples.data
  } else {
    float32Samples = new Float32Array(samples)
  }

  const buffer = ctx.createBuffer(1, float32Samples.length, sampleRate)
  buffer.copyToChannel(float32Samples, 0)

  const source = ctx.createBufferSource()
  source.buffer = buffer
  source.connect(ctx.destination)
  activeSource = source

  setState('playing', `Playing (${Math.round(float32Samples.length / sampleRate)}s)`)

  await new Promise((resolve) => {
    source.onended = () => resolve()
    source.start(0)
  })

  if (activeSource === source) {
    activeSource = null
  }
}

export function stopSpeech() {
  if (activeSource) {
    try {
      activeSource.stop(0)
      activeSource.disconnect()
    } catch {
      // no-op
    }
    activeSource = null
  }
  setState('idle')
}

export function enqueueSpeak(text, options = {}) {
  const normalized = normalizeText(text)
  if (!normalized) return Promise.resolve(false)

  const voice = options.voice || DEFAULT_VOICE
  const speed = Number(options.speed || 1.05)
  const progress_callback = options.progress_callback

  speechQueue = speechQueue
    .catch(() => false)
    .then(async () => {
      setState('generating', 'Generating speech...')
      console.log('[tts] Generating speech for: "%s..." (voice=%s, speed=%s)',
        normalized.slice(0, 60), voice, speed)

      const engine = await ensureEngine(progress_callback)
      const rawAudio = await engine.generate(normalized, { voice, speed })

      console.log('[tts] Generation complete, raw audio:', {
        hasAudio: !!rawAudio?.audio,
        audioLength: rawAudio?.audio?.length,
        sampleRate: rawAudio?.sampling_rate,
      })

      await playRawAudio(rawAudio)
      setState('idle')
      return true
    })
    .catch((err) => {
      console.error('[tts] kokoro-js failed:', err)
      setState('error', err.message || 'TTS failed')
      // Reset to idle after showing error briefly
      setTimeout(() => { if (_currentState === 'error') setState('idle') }, 4000)
      return false
    })

  return speechQueue
}

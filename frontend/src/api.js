// Thin client for the FastAPI backend. Same-origin by default; set VITE_API_URL to point elsewhere.
const BASE = import.meta.env.VITE_API_URL ?? ''

async function failure(res) {
  try {
    const body = await res.json()
    return new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${res.status})`)
  } catch {
    return new Error(`Request failed (${res.status})`)
  }
}

export async function fetchEffects() {
  const res = await fetch(`${BASE}/api/effects`)
  if (!res.ok) throw await failure(res)
  return res.json()
}

export async function fetchSample(speaker) {
  const res = await fetch(`${BASE}/api/sample?speaker=${speaker}`)
  if (!res.ok) throw await failure(res)
  return res.blob()
}

export async function analyze(wav) {
  const form = new FormData()
  form.append('file', wav, 'audio.wav')
  const res = await fetch(`${BASE}/api/analyze`, { method: 'POST', body: form })
  if (!res.ok) throw await failure(res)
  return res.json()
}

export async function processAudio(wav, effect, params) {
  const form = new FormData()
  form.append('file', wav, 'audio.wav')
  form.append('effect', effect)
  form.append('params', JSON.stringify(params))
  const res = await fetch(`${BASE}/api/process`, { method: 'POST', body: form })
  if (!res.ok) throw await failure(res)
  return {
    blob: await res.blob(),
    seconds: Number(res.headers.get('X-Processing-Time') ?? 0),
  }
}

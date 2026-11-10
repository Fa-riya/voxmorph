// Browser-side audio handling: every input (any format the browser can decode,
// or a microphone recording) is converted to 22.05 kHz mono 16-bit WAV before
// upload, so the backend always receives the same, small, lossless format.

export const TARGET_RATE = 22050
export const MAX_SECONDS = 60

export async function toWav(blob) {
  const ctx = new AudioContext()
  let decoded
  try {
    decoded = await ctx.decodeAudioData(await blob.arrayBuffer())
  } catch {
    throw new Error('This file could not be decoded. Try a WAV, MP3, OGG or M4A recording.')
  } finally {
    ctx.close()
  }
  if (decoded.duration > MAX_SECONDS) {
    throw new Error(`Recordings can be up to ${MAX_SECONDS} seconds long. This one is ${Math.round(decoded.duration)} seconds.`)
  }

  const length = Math.ceil(decoded.duration * TARGET_RATE)
  const offline = new OfflineAudioContext(1, length, TARGET_RATE) // mono mixdown + resampling
  const src = offline.createBufferSource()
  src.buffer = decoded
  src.connect(offline.destination)
  src.start()
  const rendered = await offline.startRendering()
  return { wav: encodeWav(rendered.getChannelData(0), TARGET_RATE), duration: rendered.duration }
}

function encodeWav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const write = (offset, text) => [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)))
  write(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  write(8, 'WAVE')
  write(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, rate, true)
  view.setUint32(28, rate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  write(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

export async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Recording needs a browser with microphone access over HTTPS or localhost.')
  }
  let stream
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  } catch {
    throw new Error('Microphone access was blocked. Allow it in the browser’s site settings and try again.')
  }
  const recorder = new MediaRecorder(stream)
  const chunks = []
  recorder.ondataavailable = (e) => chunks.push(e.data)
  recorder.start()
  return () =>
    new Promise((resolve) => {
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        resolve(new Blob(chunks, { type: recorder.mimeType }))
      }
      recorder.stop()
    })
}

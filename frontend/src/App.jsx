import { useEffect, useRef, useState } from 'react'
import { analyze, fetchEffects, fetchSample, processAudio } from './api.js'
import { MAX_SECONDS, startRecording, toWav } from './audio.js'
import Spectrogram from './Spectrogram.jsx'

const REPO_URL = 'https://github.com/Fa-riya/voxmorph'

function defaults(effect) {
  return Object.fromEntries((effect?.params ?? []).map((p) => [p.name, p.default]))
}

function formatHz(value) {
  return value == null ? 'Unvoiced' : `${Math.round(value)} Hz`
}

function useObjectUrl(blob) {
  const [url, setUrl] = useState(null)
  useEffect(() => {
    if (!blob) return setUrl(null)
    const u = URL.createObjectURL(blob)
    setUrl(u)
    return () => URL.revokeObjectURL(u)
  }, [blob])
  return url
}

export default function App() {
  const [effects, setEffects] = useState([])
  const [effectId, setEffectId] = useState('female')
  const [params, setParams] = useState({})
  const [source, setSource] = useState(null) // { name, wav, analysis }
  const [result, setResult] = useState(null) // { wav, analysis, seconds, effectId, params }
  const [busy, setBusy] = useState(null) // 'source' | 'process' | null
  const [error, setError] = useState(null)
  const [recording, setRecording] = useState(null) // { stop, started }
  const [elapsed, setElapsed] = useState(0)
  const fileInput = useRef(null)

  const effect = effects.find((e) => e.id === effectId)
  const sourceUrl = useObjectUrl(source?.wav)
  const resultUrl = useObjectUrl(result?.wav)
  const stale = result && (result.effectId !== effectId || JSON.stringify(result.params) !== JSON.stringify(params))

  useEffect(() => {
    fetchEffects()
      .then((list) => {
        setEffects(list)
        setParams(defaults(list.find((e) => e.id === 'female')))
      })
      .catch(() => setError('The VoxMorph server is not reachable. Start the backend on port 8000 and reload.'))
    loadSource(() => fetchSample('male').then((wav) => ({ wav, name: 'Synthetic male voice' })))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!recording) return
    const id = setInterval(() => {
      const s = (Date.now() - recording.started) / 1000
      setElapsed(s)
      if (s >= MAX_SECONDS) stopRecording()
    }, 200)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recording])

  async function loadSource(getAudio) {
    setBusy('source')
    setError(null)
    try {
      const { wav, name } = await getAudio()
      const analysis = await analyze(wav)
      setSource({ wav, name, analysis })
      setResult(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(null)
    }
  }

  function onFile(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (file) loadSource(async () => ({ ...(await toWav(file)), name: file.name }))
  }

  async function beginRecording() {
    setError(null)
    try {
      setRecording({ stop: await startRecording(), started: Date.now() })
      setElapsed(0)
    } catch (e) {
      setError(e.message)
    }
  }

  async function stopRecording() {
    if (!recording) return
    const blob = await recording.stop()
    setRecording(null)
    loadSource(async () => ({ ...(await toWav(blob)), name: 'Microphone recording' }))
  }

  function chooseEffect(e) {
    setEffectId(e.id)
    setParams(defaults(e))
  }

  async function transform() {
    if (!source || !effect) return
    setBusy('process')
    setError(null)
    try {
      const { blob, seconds } = await processAudio(source.wav, effectId, params)
      const analysis = await analyze(blob)
      setResult({ wav: blob, analysis, seconds, effectId, params })
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(null)
    }
  }

  const downloadName = `${(source?.name ?? 'voice').replace(/\.[^.]+$/, '').replace(/\s+/g, '-').toLowerCase()}-${result?.effectId}.wav`

  return (
    <div className="page">
      <header className="masthead">
        <div className="brand">
          <svg className="mark" viewBox="0 0 32 32" aria-hidden="true">
            <g fill="none" strokeLinecap="round" strokeWidth="3">
              <path d="M7 19v-6M12 23V9M17 20v-8" stroke="#B02E78" />
              <path d="M22 22V10M27 18v-4" stroke="#E98A3F" />
            </g>
          </svg>
          <span className="wordmark">VoxMorph</span>
        </div>
        <p className="tagline">Transform a voice with signal processing, then see exactly what changed.</p>
        <nav className="links">
          <a href="/docs">API reference</a>
          <a href={REPO_URL}>Source code</a>
        </nav>
      </header>

      <main className="bench">
        <aside className="controls" aria-label="Controls">
          <section className="step">
            <h2>
              <span className="step-no">1</span> Voice
            </h2>
            <div className="button-row">
              <button className="btn" onClick={() => fileInput.current.click()} disabled={busy || recording}>
                Upload audio
              </button>
              {recording ? (
                <button className="btn btn-recording" onClick={stopRecording}>
                  Stop recording <span className="tabular">{elapsed.toFixed(1)}s</span>
                </button>
              ) : (
                <button className="btn" onClick={beginRecording} disabled={busy}>
                  Record
                </button>
              )}
              <input ref={fileInput} type="file" accept="audio/*" hidden onChange={onFile} />
            </div>
            <p className="hint">
              Or use a synthetic voice:{' '}
              {['male', 'female'].map((s) => (
                <button
                  key={s}
                  className="link-button"
                  disabled={busy || recording}
                  onClick={() => loadSource(() => fetchSample(s).then((wav) => ({ wav, name: `Synthetic ${s} voice` })))}
                >
                  {s}
                </button>
              ))}
            </p>
            <div className="source" aria-live="polite">
              {busy === 'source' ? (
                <span className="muted">Analysing audio…</span>
              ) : source ? (
                <>
                  <strong>{source.name}</strong>
                  <span className="muted tabular">{source.analysis.duration.toFixed(2)} s</span>
                </>
              ) : (
                <span className="muted">No voice loaded yet.</span>
              )}
            </div>
          </section>

          <section className="step">
            <h2>
              <span className="step-no">2</span> Effect
            </h2>
            <div className="effects" role="radiogroup" aria-label="Effect">
              {effects.map((e) => (
                <button
                  key={e.id}
                  role="radio"
                  aria-checked={e.id === effectId}
                  className={`effect ${e.id === effectId ? 'is-selected' : ''}`}
                  onClick={() => chooseEffect(e)}
                >
                  {e.name}
                </button>
              ))}
            </div>
            {effect && (
              <div className="effect-detail">
                <p>{effect.description}</p>
                <ul className="techniques" aria-label="Signal processing used">
                  {effect.techniques.map((t) => (
                    <li key={t}>{t}</li>
                  ))}
                </ul>
              </div>
            )}
          </section>

          <section className="step">
            <h2>
              <span className="step-no">3</span> Settings
            </h2>
            {effect?.params.length ? (
              <>
                {effect.params.map((p) => (
                  <label key={p.name} className="slider">
                    <span className="slider-head">
                      <span>{p.label}</span>
                      <span className="tabular">
                        {params[p.name]}
                        {p.unit && ` ${p.unit}`}
                      </span>
                    </span>
                    <input
                      type="range"
                      min={p.min}
                      max={p.max}
                      step={p.step}
                      value={params[p.name] ?? p.default}
                      onChange={(ev) => setParams({ ...params, [p.name]: Number(ev.target.value) })}
                    />
                  </label>
                ))}
                <button className="link-button" onClick={() => setParams(defaults(effect))}>
                  Reset to defaults
                </button>
              </>
            ) : (
              <p className="muted">This effect has no settings.</p>
            )}
          </section>

          <button className="btn btn-primary" onClick={transform} disabled={!source || !effect || busy || recording}>
            {busy === 'process' ? 'Transforming…' : 'Transform voice'}
          </button>
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </aside>

        <section className="readout" aria-label="Results">
          <dl className="stats">
            <div>
              <dt>Original median pitch</dt>
              <dd className="tabular">{formatHz(source?.analysis.summary.median_f0)}</dd>
            </div>
            <div>
              <dt>Transformed median pitch</dt>
              <dd className="tabular">{result ? formatHz(result.analysis.summary.median_f0) : 'Not yet'}</dd>
            </div>
            <div>
              <dt>Processing time</dt>
              <dd className="tabular">{result ? `${Math.round(result.seconds * 1000)} ms` : 'Not yet'}</dd>
            </div>
          </dl>

          <article className="track">
            <header className="track-head">
              <h3>Original</h3>
              {sourceUrl && <audio controls src={sourceUrl} />}
            </header>
            <Spectrogram analysis={source?.analysis} label="Original" />
          </article>

          <article className={`track ${stale ? 'is-stale' : ''}`}>
            <header className="track-head">
              <h3>
                Transformed
                {result && <span className="muted"> with {effects.find((e) => e.id === result.effectId)?.name}</span>}
              </h3>
              {resultUrl && (
                <div className="track-actions">
                  <audio controls src={resultUrl} />
                  <a className="btn" href={resultUrl} download={downloadName}>
                    Download WAV
                  </a>
                </div>
              )}
            </header>
            <Spectrogram analysis={result?.analysis} label="Transformed" />
            {stale && <p className="hint">Settings changed since this result. Transform again to update it.</p>}
          </article>

          <p className="legend">
            <span className="swatch-line" aria-hidden="true" /> Pitch contour from the YIN estimator.
            <span className="swatch-ramp" aria-hidden="true" /> Energy over an 80 dB range, 60 Hz to 8 kHz on a log scale.
          </p>
        </section>
      </main>

      <footer className="footer">
        Audio is processed at 22.05 kHz and returned straight away. Nothing is stored on the server.
      </footer>
    </div>
  )
}

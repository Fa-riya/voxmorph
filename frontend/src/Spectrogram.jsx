import { useEffect, useRef } from 'react'

// Indigo -> magenta -> amber -> pale yellow. Perceptually ordered, readable on a light page.
const STOPS = [
  [0.0, [20, 19, 52]],
  [0.3, [58, 31, 120]],
  [0.55, [176, 46, 120]],
  [0.78, [240, 133, 74]],
  [1.0, [252, 236, 170]],
]

const LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    const k = STOPS.findIndex(([p]) => p >= t)
    const [p1, c1] = STOPS[Math.max(k, 1)]
    const [p0, c0] = STOPS[Math.max(k, 1) - 1]
    const f = (t - p0) / (p1 - p0)
    for (let j = 0; j < 3; j++) lut[i * 3 + j] = c0[j] + f * (c1[j] - c0[j])
  }
  return lut
})()

const FREQ_TICKS = [100, 250, 500, 1000, 2000, 4000, 8000]
const AXIS = { left: 44, bottom: 22, top: 6, right: 8 }

function decode(b64) {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

function timeStep(duration) {
  return [0.25, 0.5, 1, 2, 5, 10, 15].find((s) => duration / s <= 8) ?? 20
}

export default function Spectrogram({ analysis, label, height = 230 }) {
  const wrap = useRef(null)
  const canvas = useRef(null)

  useEffect(() => {
    if (!analysis) return
    const { rows, cols, data, min_freq: minFreq, max_freq: maxFreq } = analysis.spectrogram
    const pixels = decode(data)

    // Render the spectrogram once at native resolution, then scale it onto the visible canvas.
    const img = new ImageData(cols, rows)
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = pixels[r * cols + c]
        const o = ((rows - 1 - r) * cols + c) * 4 // row 0 is the lowest frequency: draw it at the bottom
        img.data[o] = LUT[v * 3]
        img.data[o + 1] = LUT[v * 3 + 1]
        img.data[o + 2] = LUT[v * 3 + 2]
        img.data[o + 3] = 255
      }
    }
    const source = document.createElement('canvas')
    source.width = cols
    source.height = rows
    source.getContext('2d').putImageData(img, 0, 0)

    const draw = () => {
      const width = wrap.current.clientWidth
      const dpr = window.devicePixelRatio || 1
      const el = canvas.current
      el.width = width * dpr
      el.height = height * dpr
      const ctx = el.getContext('2d')
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, width, height)

      const plotW = width - AXIS.left - AXIS.right
      const plotH = height - AXIS.top - AXIS.bottom
      const x0 = AXIS.left
      const y0 = AXIS.top
      ctx.imageSmoothingEnabled = true
      ctx.drawImage(source, x0, y0, plotW, plotH)

      const yOf = (hz) => y0 + plotH * (1 - Math.log(hz / minFreq) / Math.log(maxFreq / minFreq))
      const xOf = (t) => x0 + plotW * (t / analysis.duration)

      // Pitch contour
      ctx.strokeStyle = '#9EF0FF'
      ctx.lineWidth = 2
      ctx.lineJoin = 'round'
      ctx.beginPath()
      let pen = false
      analysis.pitch.f0.forEach((f, i) => {
        if (f == null) {
          pen = false
          return
        }
        const x = xOf(analysis.pitch.times[i])
        const y = yOf(f)
        pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y)
        pen = true
      })
      ctx.stroke()

      // Axes
      ctx.fillStyle = '#5E6383'
      ctx.font = '500 11px "Schibsted Grotesk", system-ui, sans-serif'
      ctx.textAlign = 'right'
      ctx.textBaseline = 'middle'
      FREQ_TICKS.filter((f) => f >= minFreq && f <= maxFreq).forEach((f) => {
        ctx.fillText(f < 1000 ? `${f}` : `${f / 1000}k`, x0 - 8, Math.min(Math.max(yOf(f), y0 + 6), y0 + plotH - 4))
      })
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      const step = timeStep(analysis.duration)
      for (let t = 0; t <= analysis.duration + 1e-6; t += step) {
        ctx.fillText(`${+t.toFixed(2)}s`, Math.min(Math.max(xOf(t), x0 + 10), x0 + plotW - 12), y0 + plotH + 6)
      }
    }

    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(wrap.current)
    return () => observer.disconnect()
  }, [analysis, height])

  return (
    <div className="spectrogram" ref={wrap} style={{ height }}>
      {analysis ? (
        <canvas ref={canvas} style={{ width: '100%', height }} role="img" aria-label={`${label} spectrogram with pitch contour`} />
      ) : (
        <div className="spectrogram-empty" style={{ height }}>
          {label === 'Transformed' ? 'Choose an effect and transform the voice to compare.' : 'Add a voice to see its spectrogram.'}
        </div>
      )}
    </div>
  )
}

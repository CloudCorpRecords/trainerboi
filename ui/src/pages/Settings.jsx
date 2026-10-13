import { useEffect, useState } from 'react'
import { Activity, FolderOpen, ShieldCheck, Thermometer } from 'lucide-react'
import { post, usePoll, useJob, useHardware, toast, fmt } from '../api'
import { Button, Card, Chip, Field, PageHeader, Progress, Stat } from '../ui'
import { Sparkline } from '../charts'
import { Mascot, tempMood } from '../mascot'
import { openAll, setOpenAll } from '../rewards'

export default function Settings() {
  const { hw, history } = useHardware()
  const { data: s, reload } = usePoll('/api/settings', 600000)
  const { data: health, reload: reloadHealth } = usePoll('/api/gpu/health', 600000)
  const [cap, setCap] = useState(82)
  const [job, follow] = useJob()
  const [open, setOpen] = useState(openAll())
  useEffect(() => { if (s) setCap(s.gpu_max_temp) }, [s?.gpu_max_temp])
  useEffect(() => { if (job?.state === 'done') reloadHealth(); if (job?.state === 'error') toast(job.error, 'error') }, [job?.state])
  const g = hw?.gpu
  const running = hw?.jobs?.length > 0

  const check = async () => { try { const r = await post('/api/gpu/check'); follow(r.job) } catch (e) { toast(e.message, 'error') } }
  const saveCap = async () => { await post('/api/settings', { gpu_max_temp: cap }); reload(); toast('Saved. Applies to the next training run.', 'success') }
  const p = job?.progress

  return (
    <div className="page" style={{ maxWidth: 900 }}>
      <PageHeader title="Care & comfort 💜" sub="Keeping your graphics card happy and your results trustworthy. Everything here is live, straight from your computer." />

      <div className="grid g4" style={{ marginBottom: 16 }}>
        <Stat label="GPU" value={g?.name || '–'}><div className="small faint">{g?.vram_total_gb ? `${Math.round(g.vram_total_gb)} GB memory` : ''}</div></Stat>
        <Stat label="Activity" value={g?.util != null ? Math.round(g.util) : '–'} unit="%"><Sparkline data={history.util} min={0} max={100} /></Stat>
        <Stat label="Temperature" value={g?.temp != null ? Math.round(g.temp) : '–'} unit="°C" tone={g?.temp >= 85 ? 'bad' : g?.temp >= 78 ? 'warn' : 'ok'}><Sparkline data={history.temp} color="var(--warn)" /></Stat>
        <Stat label="Power" value={g?.power != null ? Math.round(g.power) : '–'} unit={g?.power_limit ? `/ ${Math.round(g.power_limit)} W` : 'W'}><Sparkline data={history.power} min={0} max={g?.power_limit || 150} color="var(--ok)" /></Stat>
      </div>

      <div className="stack">
        <Card title="Computer">
          <div className="grid g2 small">
            <div><span className="faint">CPU</span><div>{hw?.cpu?.name} · {hw?.cpu?.cores} cores / {hw?.cpu?.threads} threads</div></div>
            <div><span className="faint">Memory</span><div>{hw ? `${hw.ram.used_gb.toFixed(1)} / ${hw.ram.total_gb.toFixed(0)} GB used` : '–'}</div></div>
            <div><span className="faint">Disk free</span><div>{hw ? `${hw.disk_free_gb.toFixed(0)} GB` : '–'}</div></div>
            <div><span className="faint">Data folder</span><div className="row" style={{ gap: 6 }}><code style={{ wordBreak: 'break-all' }}>{s?.root}</code><Button size="sm" variant="ghost" icon={FolderOpen} onClick={() => post('/api/reveal', { path: s?.root })} /></div></div>
          </div>
        </Card>

        <Card title="Keeping it cozy 🧊">
          <p className="muted" style={{ marginBottom: 14 }}>Laptop GPUs can silently compute wrong answers when they get very hot (we measured this at about 86°C on this machine). TrainerBoi pauses training when the GPU reaches the limit below and continues once it has cooled down.</p>
          <Field label={`Pause training at ${cap}°C${cap === 0 ? ' (off)' : ''}`} hint="82°C is a good default. Lower is safer but slower. 0 turns the safety pause off.">
            <input type="range" min="0" max="92" value={cap} onChange={(e) => setCap(+e.target.value)} />
          </Field>
          <div style={{ marginTop: 10 }}><Button icon={Thermometer} onClick={saveCap} disabled={s && cap === s.gpu_max_temp}>Save</Button></div>
        </Card>

        <Card title="Open everything 🗝️">
          <p className="muted" style={{ marginBottom: 12 }}>Boi's tricks (teach-back, birth certificates, the advanced Workshop methods) open up gently as you explore. If you already know your way around, switch this on and everything is yours from the start.</p>
          <label className="row"><input type="checkbox" checked={open} onChange={(e) => { setOpen(e.target.checked); setOpenAll(e.target.checked); toast(e.target.checked ? 'Everything is open. 🗝️' : 'Back to unlocking as you go. 🌱', 'success') }} /> Open everything from the start</label>
        </Card>

        <Card title="Wellness check 🩺" action={health && <Chip tone={health.stable ? 'ok' : 'bad'}>{health.stable ? 'Feeling great' : 'Needs a little care'}</Chip>}>
          <p className="muted" style={{ marginBottom: 12 }}>Runs the same calculations thousands of times under full load for about 3 minutes and checks the answers never change. Do this once — it tells you whether your GPU can be trusted for long runs.</p>
          {health && (
            <div className={`callout ${health.stable ? 'ok' : 'bad'}`} style={{ marginBottom: 12 }}>
              <ShieldCheck size={16} style={{ flexShrink: 0 }} />
              <div>{health.stable ? 'No calculation errors in the last check.' : `${health.errors ?? 'Some'} calculation error(s) found${health.max_temp ? ` (GPU reached ${Math.round(health.max_temp)}°C)` : ''}. Use a cooling pad, set Windows power mode to Balanced, clean the fans, and keep the safety pause on.`}</div>
            </div>
          )}
          <div className="row wrap">
            <Button icon={Activity} variant="primary" onClick={check} disabled={job?.state === 'running' || running}>{job?.state === 'running' ? 'Checking in…' : 'Run the wellness check'}</Button>
            {running && <span className="small muted">Available when no training is running.</span>}
          </div>
          {job?.state === 'running' && (
            <div style={{ marginTop: 14 }}><Progress value={p ? p.done / p.total : null} />
              <div className="small muted" style={{ marginTop: 6 }}>{p ? `${p.done}/${p.total}s · GPU ${p.temp}°C · errors so far: ${p.errors}` : 'Starting…'}</div></div>
          )}
        </Card>
      </div>
    </div>
  )
}

import { usePoll } from '../api'
import { Progress } from '../ui'
import { Wrench, AlertTriangle } from 'lucide-react'

export default function Advanced() {
  const { data } = usePoll('/api/classic/status', 2000)
  if (data?.state === 'ready') return <iframe className="frame" src="/classic/" title="Advanced tools" />
  return (
    <div className="page" style={{ maxWidth: 560 }}>
      <div className="empty" style={{ paddingTop: 90 }}>
        {data?.state === 'error' ? <AlertTriangle size={30} /> : <Wrench size={30} />}
        <h3>{data?.state === 'error' ? 'Advanced tools could not start' : data?.state === 'off' ? 'Advanced tools are turned off' : 'Loading advanced tools…'}</h3>
        <p>{data?.state === 'error' ? data.error : 'The dataset builder, AutoTrain, RAG, audio, comparison and tuning tools load in the background the first time (about a minute).'}</p>
        {data?.state === 'loading' && <div style={{ width: 240, marginTop: 8 }}><Progress indeterminate /></div>}
      </div>
    </div>
  )
}

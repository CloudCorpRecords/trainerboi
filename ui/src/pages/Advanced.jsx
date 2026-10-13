import { usePoll } from '../api'
import { Progress } from '../ui'
import { Mascot } from '../mascot'

export default function Advanced() {
  const { data } = usePoll('/api/classic/status', 2000)
  if (data?.state === 'ready') return <iframe className="frame" src="/classic/" title="Advanced tools" />
  return (
    <div className="page" style={{ maxWidth: 560 }}>
      <div className="empty" style={{ paddingTop: 90 }}>
        <Mascot mood={data?.state === 'error' ? 'oops' : 'working'} size={110} />
        <h3>{data?.state === 'error' ? 'The toolbox needs a moment' : data?.state === 'off' ? 'The toolbox is switched off' : 'Unpacking the toolbox…'}</h3>
        <p>{data?.state === 'error' ? data.error : 'The dataset builder, AutoTrain, RAG, audio, comparison and tuning tools are being unpacked in the background — about a minute the first time.'}</p>
        {data?.state === 'loading' && <div style={{ width: 240, marginTop: 8 }}><Progress indeterminate /></div>}
      </div>
    </div>
  )
}

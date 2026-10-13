// Boi's friends: Professor Hoot the owl (maths) and Biscuit the bunny (data). Same style: pure SVG, themed, blinking.

export function Owl({ mood = 'happy', size = 96 }) {
  const think = mood === 'think', cheer = mood === 'cheer'
  return (
    <svg className={`friend owl mood-${mood}`} width={size} height={size} viewBox="0 0 120 120" role="img" aria-label="Professor Hoot the owl">
      <g className="f-bob">
        <ellipse cx="60" cy="68" rx="36" ry="38" fill="var(--f-owl)" />
        <path d="M34 36 L28 20 L46 30Z M86 36 L92 20 L74 30Z" fill="var(--f-owl)" />
        <ellipse cx="60" cy="80" rx="22" ry="24" fill="var(--f-belly)" />
        {[0, 1, 2].map((i) => <path key={i} d={`M${50 + i * 10} 76 q3 5 6 0`} stroke="var(--f-owl)" strokeWidth="2" fill="none" strokeLinecap="round" opacity=".6" />)}
        <g className={`f-wing l ${cheer ? 'up' : ''}`}><ellipse cx="27" cy="72" rx="9" ry="20" fill="var(--f-owl2)" /></g>
        <g className={`f-wing r ${cheer ? 'up' : ''}`}><ellipse cx="93" cy="72" rx="9" ry="20" fill="var(--f-owl2)" /></g>
        <g className="f-eyes">
          <circle cx="45" cy="54" r="13" fill="#fff" /><circle cx="75" cy="54" r="13" fill="#fff" />
          <g className="f-pupils" style={{ transform: think ? 'translate(3px,-4px)' : undefined }}>
            <circle cx="46" cy="55" r="6" fill="var(--m-ink)" /><circle cx="74" cy="55" r="6" fill="var(--m-ink)" />
            <circle cx="48" cy="52.5" r="2" fill="#fff" /><circle cx="76" cy="52.5" r="2" fill="#fff" />
          </g>
        </g>
        <g fill="none" stroke="var(--m-ink)" strokeWidth="2.4"><circle cx="45" cy="54" r="15" /><circle cx="75" cy="54" r="15" /><path d="M60 54 L60 54 M58 54 Q60 52 62 54" /></g>
        <path d="M55 66 L65 66 L60 75Z" fill="#ffd36b" stroke="#ffab7d" strokeWidth="1.5" strokeLinejoin="round" />
        {cheer && <g className="f-sparks"><path d="M14 30 l2 5 5 2 -5 2 -2 5 -2 -5 -5 -2 5 -2z" /><path d="M104 22 l1.5 4 4 1.5 -4 1.5 -1.5 4 -1.5 -4 -4 -1.5 4 -1.5z" /></g>}
        {think && <g className="f-think"><circle cx="96" cy="28" r="2.4" /><circle cx="103" cy="19" r="3.4" /><text x="104" y="14" fontSize="14" textAnchor="middle" fill="var(--accent)" fontWeight="800">?</text></g>}
        <path d="M48 104 l-4 8 M56 104 l0 8 M64 104 l0 8 M72 104 l4 8" stroke="#ffab7d" strokeWidth="3" strokeLinecap="round" />
      </g>
    </svg>
  )
}

export function Bunny({ mood = 'happy', size = 96 }) {
  const worried = mood === 'oops', cheer = mood === 'cheer'
  return (
    <svg className={`friend bunny mood-${mood}`} width={size} height={size} viewBox="0 0 120 120" role="img" aria-label="Biscuit the bunny">
      <g className="f-bob">
        <g className={`f-ear l ${worried ? 'down' : ''}`}><ellipse cx="44" cy="26" rx="9" ry="24" fill="var(--f-bun)" /><ellipse cx="44" cy="28" rx="4.5" ry="16" fill="#ffc0dd" /></g>
        <g className={`f-ear r ${worried ? 'down' : ''}`}><ellipse cx="76" cy="26" rx="9" ry="24" fill="var(--f-bun)" /><ellipse cx="76" cy="28" rx="4.5" ry="16" fill="#ffc0dd" /></g>
        <ellipse cx="60" cy="86" rx="30" ry="26" fill="var(--f-bun)" />
        <circle cx="60" cy="62" r="29" fill="var(--f-bun)" />
        <ellipse cx="41" cy="70" rx="6.5" ry="4" fill="#ff9ecb" opacity=".6" /><ellipse cx="79" cy="70" rx="6.5" ry="4" fill="#ff9ecb" opacity=".6" />
        <g className="f-eyes">
          {cheer ? <><path d="M42 60 Q48 52 54 60" className="m-line" /><path d="M66 60 Q72 52 78 60" className="m-line" /></>
            : <><ellipse cx="48" cy="60" rx="5" ry={worried ? 7 : 6.2} fill="var(--m-ink)" /><ellipse cx="72" cy="60" rx="5" ry={worried ? 7 : 6.2} fill="var(--m-ink)" /><circle cx="49.6" cy="57.6" r="1.8" fill="#fff" /><circle cx="73.6" cy="57.6" r="1.8" fill="#fff" /></>}
        </g>
        {worried && <><path d="M41 51 L55 55" className="m-line thin" /><path d="M79 51 L65 55" className="m-line thin" /></>}
        <path d="M56 68 Q60 72 64 68 Q60 66 56 68Z" fill="#ff7fa8" />
        <path d={worried ? 'M54 77 Q60 72 66 77' : cheer ? 'M52 72 Q60 84 68 72Z' : 'M54 73 Q60 79 66 73'} className="m-line" fill={cheer ? '#ff7fa8' : 'none'} />
        <ellipse cx="42" cy="108" rx="11" ry="6" fill="var(--f-bun)" /><ellipse cx="78" cy="108" rx="11" ry="6" fill="var(--f-bun)" />
        <circle cx="92" cy="96" r="8" fill="#fff" opacity=".9" />
        {cheer && <g className="f-sparks"><path d="M14 40 l2 5 5 2 -5 2 -2 5 -2 -5 -5 -2 5 -2z" /><path d="M104 34 l1.5 4 4 1.5 -4 1.5 -1.5 4 -1.5 -4 -4 -1.5 4 -1.5z" /></g>}
      </g>
    </svg>
  )
}

/** A friend with a little speech bubble. */
export function Friend({ who, mood, quip, size = 88 }) {
  const C = who === 'owl' ? Owl : Bunny
  return (
    <div className="friend-wrap">
      {quip && <div className="friend-quip" key={quip}>{quip}</div>}
      <C mood={mood} size={size} />
    </div>
  )
}

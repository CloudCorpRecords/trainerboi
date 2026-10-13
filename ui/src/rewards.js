// Stickers, ranks and "tricks": little rewards for real things you do. Nothing here costs money or can be bought.
// Everything is stored only in this browser (localStorage). Stickers celebrate; tricks (PERKS) are real abilities
// that open up as you go - and "Open everything" in Care & comfort turns them all on if you'd rather skip ahead.
export const STICKERS = [
  { id: 'hello', emoji: '👋', title: 'Hello, world', how: 'Open TrainerBoi for the first time' },
  { id: 'first_model', emoji: '🌱', title: 'First sprout', how: 'Finish teaching your first model', hat: 'party' },
  { id: 'three_models', emoji: '🌷', title: 'Little garden', how: 'Have three models in your garden', hat: 'flowers' },
  { id: 'five_models', emoji: '🌳', title: 'Little forest', how: 'Have five models in your garden' },
  { id: 'math', emoji: '🧮', title: 'Number friend', how: 'Finish a Math playground session', hat: 'glasses' },
  { id: 'chatted', emoji: '💬', title: 'Hello, little model', how: 'Chat with a model you made', hat: 'headphones' },
  { id: 'lesson', emoji: '🎓', title: 'Good teacher', how: 'Save a lesson from one of your chats' },
  { id: 'shared', emoji: '🎁', title: 'Gift giver', how: 'Wrap up, publish or share a model', hat: 'crown' },
  { id: 'travel', emoji: '🧳', title: 'Packed for travel', how: 'Make an Ollama / LM Studio pack' },
  { id: 'postcard', emoji: '📜', title: 'Proud parent', how: 'Download a birth certificate for a model' },
  { id: 'cleaned', emoji: '🧹', title: 'Tidy helper', how: 'Make a tidied-up copy of a dataset', hat: 'sparkle' },
  { id: 'gathered', emoji: '🥣', title: 'Little chef', how: 'Gather a batch of data in the data kitchen', hat: 'chef' },
  { id: 'workshop', emoji: '🔧', title: 'Tinkerer', how: 'Start a run from the Workshop', hat: 'hardhat' },
  { id: 'taste', emoji: '🍰', title: 'Fine taste', how: 'Vote in a taste test', hat: 'tophat' },
  { id: 'poked', emoji: '💜', title: 'Best friends', how: 'Give Boi ten happy clicks', hat: 'bow' },
  { id: 'night_owl', emoji: '🌙', title: 'Night owl', how: 'Visit after midnight', hat: 'nightcap' },
  { id: 'cozy3', emoji: '🛋️', title: 'Cozy regular', how: 'Visit on three different days' },
  { id: 'cozy7', emoji: '🕯️', title: 'Old friend', how: 'Visit on seven different days' },
]

// Real abilities that open up. `needs` is a sticker id. Add one here + check perkOn('id') where it lives.
export const PERKS = [
  { id: 'teachback', emoji: '🎓', title: 'Teach-back', needs: 'chatted', where: 'In a model\'s chat',
    what: 'Rate a reply 👍, or fix it with your own better answer. Boi keeps your lessons as a dataset you can train on — so your model learns from your corrections.' },
  { id: 'certificate', emoji: '📜', title: 'Birth certificates', needs: 'first_model', where: 'A model\'s Share tab',
    what: 'Download a pretty one-page certificate for any model: its name, what it was taught from, and when it was born. Lovely to send along with a share.' },
  { id: 'advanced', emoji: '🔧', title: 'Advanced Workshop', needs: 'first_model', where: 'Workshop',
    what: 'Opens the preference, reinforcement-learning and from-scratch methods, so you meet the simple ones first and the clever ones when you\'re ready.' },
]

export const RANKS = [
  { at: 0, emoji: '🌰', name: 'Seedling' }, { at: 3, emoji: '🌱', name: 'Sprout' }, { at: 6, emoji: '🌿', name: 'Sapling' },
  { at: 10, emoji: '🌸', name: 'Bloom' }, { at: 14, emoji: '🌳', name: 'Grove-keeper' }, { at: 18, emoji: '🌟', name: 'Starlight' },
]

export const HATS = [
  { id: 'none', label: 'No hat', emoji: '🚂' }, { id: 'party', label: 'Party hat', emoji: '🎉' }, { id: 'flowers', label: 'Flower crown', emoji: '🌼' },
  { id: 'glasses', label: 'Smart glasses', emoji: '🤓' }, { id: 'headphones', label: 'Headphones', emoji: '🎧' }, { id: 'crown', label: 'Little crown', emoji: '👑' },
  { id: 'sparkle', label: 'Sparkle tiara', emoji: '✨' }, { id: 'bow', label: 'Big bow', emoji: '🎀' }, { id: 'nightcap', label: 'Night cap', emoji: '🌛' },
  { id: 'chef', label: 'Chef hat', emoji: '👨‍🍳' }, { id: 'hardhat', label: 'Hard hat', emoji: '⛑️' }, { id: 'tophat', label: 'Top hat', emoji: '🎩' },
]

const KEY = 'tb-stickers'
const read = () => { try { return JSON.parse(localStorage.getItem(KEY) || '{}') } catch { return {} } }
export const earned = () => read()
export const hasHat = (hat) => hat === 'none' || STICKERS.some((s) => s.hat === hat && read()[s.id])

export function earn(id) {
  const all = read()
  if (all[id]) return false
  all[id] = Date.now()
  try { localStorage.setItem(KEY, JSON.stringify(all)) } catch { /* ignore */ }
  window.dispatchEvent(new CustomEvent('tb-earned', { detail: STICKERS.find((s) => s.id === id) }))
  return true
}

// "Open everything": skip the gentle unlocking (kept in this browser)
export const openAll = () => { try { return localStorage.getItem('tb-openall') === '1' } catch { return false } }
export function setOpenAll(v) {
  try { localStorage.setItem('tb-openall', v ? '1' : '0') } catch { /* ignore */ }
  window.dispatchEvent(new CustomEvent('tb-earned'))        // lets screens re-read what's unlocked
}
export const perkOn = (id) => {
  const p = PERKS.find((x) => x.id === id)
  return !p || openAll() || !!read()[p.needs]
}

export function rank(count = Object.keys(read()).filter((k) => STICKERS.some((s) => s.id === k)).length) {
  let i = 0
  RANKS.forEach((r, n) => { if (count >= r.at) i = n })
  const next = RANKS[i + 1] || null
  return { ...RANKS[i], index: i, count, next, toNext: next ? next.at - count : 0 }
}

export const getHat = () => { try { return localStorage.getItem('tb-hat') || 'none' } catch { return 'none' } }
export function setHat(h) {
  try { localStorage.setItem('tb-hat', h) } catch { /* ignore */ }
  window.dispatchEvent(new CustomEvent('tb-hat', { detail: h }))
}

// click counter for "Best friends"
export function pokeCount() {
  try { const n = (+localStorage.getItem('tb-pokes') || 0) + 1; localStorage.setItem('tb-pokes', String(n)); return n } catch { return 0 }
}

// Cozy days: how many different days you've dropped by. It only ever goes up - no streaks to lose, no guilt.
export function visitToday() {
  try {
    const days = JSON.parse(localStorage.getItem('tb-days') || '[]')
    const today = new Date().toISOString().slice(0, 10)
    if (!days.includes(today)) { days.push(today); localStorage.setItem('tb-days', JSON.stringify(days.slice(-400))) }
    if (days.length >= 3) earn('cozy3')
    if (days.length >= 7) earn('cozy7')
    return days.length
  } catch { return 1 }
}
export const cozyDays = () => { try { return JSON.parse(localStorage.getItem('tb-days') || '[]').length } catch { return 0 } }

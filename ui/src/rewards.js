// Stickers and hats: little rewards for real things you do. Stored only in this browser (localStorage).
export const STICKERS = [
  { id: 'hello', emoji: '👋', title: 'Hello, world', how: 'Open TrainerBoi for the first time' },
  { id: 'first_model', emoji: '🌱', title: 'First sprout', how: 'Finish teaching your first model', hat: 'party' },
  { id: 'three_models', emoji: '🌷', title: 'Little garden', how: 'Have three models in your garden', hat: 'flowers' },
  { id: 'math', emoji: '🧮', title: 'Number friend', how: 'Finish a Math playground session', hat: 'glasses' },
  { id: 'chatted', emoji: '💬', title: 'Hello, little model', how: 'Chat with a model you made', hat: 'headphones' },
  { id: 'shared', emoji: '🎁', title: 'Gift giver', how: 'Wrap up, publish or share a model', hat: 'crown' },
  { id: 'cleaned', emoji: '🧹', title: 'Tidy helper', how: 'Make a tidied-up copy of a dataset', hat: 'sparkle' },
  { id: 'poked', emoji: '💜', title: 'Best friends', how: 'Give Boi ten happy clicks', hat: 'bow' },
  { id: 'night_owl', emoji: '🌙', title: 'Night owl', how: 'Visit after midnight', hat: 'nightcap' },
  { id: 'gathered', emoji: '🥣', title: 'Little chef', how: 'Gather a batch of data in the data kitchen', hat: 'chef' },
  { id: 'workshop', emoji: '🔧', title: 'Tinkerer', how: 'Start a run from the Workshop', hat: 'hardhat' },
  { id: 'taste', emoji: '🍰', title: 'Fine taste', how: 'Vote in a taste test', hat: 'tophat' },
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

export const getHat = () => { try { return localStorage.getItem('tb-hat') || 'none' } catch { return 'none' } }
export function setHat(h) {
  try { localStorage.setItem('tb-hat', h) } catch { /* ignore */ }
  window.dispatchEvent(new CustomEvent('tb-hat', { detail: h }))
}

// click counter for "Best friends"
export function pokeCount() {
  try { const n = (+localStorage.getItem('tb-pokes') || 0) + 1; localStorage.setItem('tb-pokes', String(n)); return n } catch { return 0 }
}

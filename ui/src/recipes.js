// Ready-made "recipes": one click fills in the goal, personality and time for you.
export const RECIPES = [
  { id: 'pirate', emoji: '🏴‍☠️', title: 'Pirate buddy', blurb: 'A chatty sea-dog who answers everything in pirate speak.', goal: 'persona', effort: 'quick',
    persona: 'You are Captain Biscuit, a cheerful pirate. You answer helpfully but always in pirate speak, with a pinch of humour. Keep answers short and friendly.' },
  { id: 'storyteller', emoji: '🧸', title: 'Bedtime storyteller', blurb: 'Teach a tiny brand-new model to write cozy little stories.', goal: 'scratch', effort: 'quick' },
  { id: 'helper', emoji: '🛎️', title: 'Friendly helper', blurb: 'A warm, patient assistant who explains things simply.', goal: 'persona', effort: 'balanced',
    persona: 'You are Sunny, a warm and patient helper. You explain things simply, check that the person understands, and always suggest a friendly next step.' },
  { id: 'study', emoji: '📚', title: 'Study buddy for my notes', blurb: 'Drop in your notes or PDFs, then ask questions about them.', goal: 'docs', effort: 'balanced', needs: 'Add your notes or PDFs below 👇' },
  { id: 'coder', emoji: '💻', title: 'Code explainer', blurb: 'Point it at a GitHub repo and it learns to explain the code.', goal: 'code', effort: 'balanced', needs: 'Paste a GitHub repo link below 👇' },
  { id: 'trivia', emoji: '🌍', title: 'Trivia whiz', blurb: 'Pick a topic you love — it reads Wikipedia to become an expert.', goal: 'docs', effort: 'balanced', needs: 'Type a topic you love below 👇 (like “volcanoes” or “jazz”)' },
  { id: 'polite', emoji: '🎩', title: 'Polite assistant', blurb: 'Nudge a model toward kinder, clearer answers.', goal: 'prefer', effort: 'balanced' },
  { id: 'math', emoji: '🧮', title: 'Math whiz', blurb: 'Practise with problems that check themselves.', link: '/math' },
]

const ADJ = ['fluffy', 'sparkly', 'sleepy', 'bouncy', 'cozy', 'giggly', 'sunny', 'tiny', 'dreamy', 'jolly', 'cuddly', 'mellow', 'twinkly', 'snuggly', 'breezy', 'peachy']
const ANI = ['otter', 'panda', 'fox', 'bunny', 'koala', 'hedgehog', 'penguin', 'duckling', 'kitten', 'puppy', 'owl', 'turtle', 'llama', 'hamster', 'dolphin', 'chick']
const pick = (a) => a[Math.floor(Math.random() * a.length)]
/** A silly, friendly name for a run, like "fluffy-otter-42". */
export const cuteName = () => `${pick(ADJ)}-${pick(ANI)}-${Math.floor(10 + Math.random() * 89)}`

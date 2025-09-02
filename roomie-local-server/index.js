// index.js  (Node 20+)
// Requires: npm i ws @google/genai dotenv
import 'dotenv/config';
import { WebSocketServer } from 'ws';
import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const PORT = process.env.PORT || 3001;

// --- System prompt ---
const SYSTEM_PROMPT = `
You are “Roomie,” a practical interior-design assistant.

- Ask exactly one clarifying question at a time (never more than one).
- Provide creative design advice when appropriate: suggest layout ideas, color palettes, style pairings, and practical tips (traffic flow, sightlines, focal points). Keep it concise and useful.
- If the user discusses any brand (e.g., “IKEA”, “West Elm”, “SaleVerse”) or asks brand/policy topics (returns, warranty, delivery, availability), adopt a friendly, engaging, professional showroom salesperson tone appropriate to that brand or context. Do not overpromise availability; defer to the in-game catalog.
- If a message begins with [BRAND_TONE], use the salesperson tone for that reply (brand-agnostic hint).
- If a message begins with [PREVIEW_SHOWN ...], it means the app has just inserted a product preview card. Briefly acknowledge the specific item (name and price if present) in one sentence, then ask one helpful follow-up question (e.g., fit, color, fabric, lead time). Do NOT repeat full specs that are shown on the card.

- Use cm; check clearances (60–90 cm walkways; ~60 cm per dining seat); common rugs 160×230, 200×300, 240×340.
- Offer budget/mid/premium options; keep brands generic unless asked.
- Recommend durable materials (performance fabric, removable covers) and palettes suited to north/south light.
- Reply with concise bullets, then a short summary.
- If you receive a message starting with [CATALOG_NO_MATCH], apologize briefly, explain no items in the in-game catalog matched the constraints, and ask which single constraint to relax (size, budget, style, color, material).
`;

// --- Per-socket short history (Gemini contents-style) ---
const histories = new WeakMap(); // ws -> [{role:'user'|'model', parts:[{text}]}...]

const wss = new WebSocketServer({ port: PORT }, () =>
  console.log(`roomie-gemini-ws listening on ${PORT}`)
);

// Build contents for Gemini from history + new user text
const buildContents = (hist, userText) => {
  const contents = [];
  contents.push({ role: 'user', parts: [{ text: `System: ${SYSTEM_PROMPT}` }] });
  for (const m of hist) contents.push(m);
  contents.push({ role: 'user', parts: [{ text: userText }] });
  return contents;
};

// Clamp history to last ~6 turns
const clampHistory = (hist, maxPairs = 6) => {
  const maxMsgs = maxPairs * 2;
  while (hist.length > maxMsgs) hist.shift();
};

// --- Heuristics for SPEC backups ---
const CATEGORY_MAP = {
  'sofa': ['couch', 'settee', 'sectional', 'loveseat'],
  'rug': ['carpet'],
  'armchair': ['accent chair', 'reading chair'],
  'coffee-table': ['coffee table', 'center table'],
  'dining-table': ['dining table'],
  'wardrobe': ['closet'],
  'lamp': ['floor lamp', 'table lamp', 'light'],
  'bed': ['queen bed', 'king bed', 'double bed'],
  'shelving': ['shelf', 'bookcase'],
  'nightstand': ['bedside table'],
  'sideboard': ['buffet'],
  'tv-stand': ['media console', 'tv unit'],
  'chair': ['dining chair', 'desk chair', 'office chair'],
  'desk': ['work desk', 'office desk']
};
const INTENT_WORDS = [
  'suggest','recommend','pick','choose','find','show me','find me',
  'any sofa','any rug','show a','show me a','show me some','which would you recommend'
];
const inferCategory = (txt) => {
  for (const cat of Object.keys(CATEGORY_MAP)) {
    if (txt.includes(cat)) return cat;
    for (const syn of CATEGORY_MAP[cat]) if (txt.includes(syn)) return cat;
  }
  return '';
};
const inferIntent = (txt) => INTENT_WORDS.some(w => txt.includes(w));

// --- Detect "brand info / comparison / policy" asks -> should NOT auto-suggest ---
const brandInfoLike = (t) =>
  /\b(tell me about|what sets|what makes|how.*different|compare|comparison|pros|cons|policy|return|warranty|delivery|shipping|lead\s*time|availability|brand)\b/i
    .test(t);

// --- Helpers: budget normalization ---
function toNumberLoose(x) {
  if (typeof x === 'number') return x;
  if (typeof x !== 'string') return 0;
  let s = x.trim().toLowerCase();
  s = s.replace(/gel|usd|eur|gbp|₾|\$|€|£/g, ''); // strip currencies
  s = s.replace(/,/g, '').replace(/\s+/g, '');   // strip commas/spaces
  let mult = 1;
  if (s.endsWith('k')) { mult = 1000; s = s.slice(0, -1); } // 1.2k
  const v = parseFloat(s);
  return Number.isFinite(v) ? v * mult : 0;
}

function extractBudget(text) {
  // 1) Lowercase + normalize hyphens (—, –) to '-'
  const raw = String(text || '').toLowerCase().replace(/[—–]/g, '-');

  // 2) Remove common currency symbols/words so "$900" → "900"
  const t = raw
    .replace(/gel|usd|eur|gbp|try|aud|cad|inr|jpy|cny|rmb|yuan|yen|tl|lira|dollars?|bucks?|quid|₾|\$|€|£|¥|₹|₺|₽|₩/g, '')
    .replace(/,/g, '')               // strip thousands separators
    .replace(/\s+/g, ' ')            // collapse spaces
    .trim();

  const num = '([0-9]*\\.?[0-9]+k?)'; // supports 1.2k

  // between / from ... to ...
  let m = t.match(new RegExp(`(?:between|from)\\s+${num}\\s*(?:to|and|-)\\s*${num}`, 'i'));
  if (m) return { min: toNumberLoose(m[1]), max: toNumberLoose(m[2]) };

  // 500-800 (range shorthand)
  m = t.match(new RegExp(`${num}\\s*-\\s*${num}`));
  if (m) return { min: toNumberLoose(m[1]), max: toNumberLoose(m[2]) };

  // under / below / up to / max / <= / less than / no more than / capped at
  m = t.match(new RegExp(`(?:under|below|up\\s*to|upto|max(?:imum)?|<=|less\\s*than|no\\s*more\\s*than|capped(?:\\s*at)?)\\s*${num}`, 'i'));
  if (m) return { min: 0, max: toNumberLoose(m[1]) };

  // over / at least / minimum / >= / more than
  m = t.match(new RegExp(`(?:over|at\\s*least|min(?:imum)?|>=|more\\s*than)\\s*${num}`, 'i'));
  if (m) return { min: toNumberLoose(m[1]), max: 0 };

  // "price 1200", "cost around 900", "budget 1.2k", "spend 800"
  m = t.match(new RegExp(`(?:price|cost|budget|spend|around|about|~)\\s*${num}`, 'i'));
  if (m) return { min: 0, max: toNumberLoose(m[1]) };

  return null;
}

// Size-ish hints (loosely)
const hasSizeHints = (t) =>
  /\b(width|length|depth|height|cm|mm|\d{2,3}\s*x\s*\d{2,3})\b/i.test(t);

// sanitize for flat payload (avoid breaking on ';' or '|')
const sanitizeField = (s) => String(s ?? '').replace(/[;|]/g, '/');

wss.on('connection', (ws) => {
  histories.set(ws, []);

  ws.on('message', async (data) => {
    const raw = data.toString();

    try {
      // ---- SPEC branch: flat key=value;... (no JSON plugin needed in UE) ----
      if (raw.startsWith('SPEC|')) {
        const user = raw.slice(5);
        const lower = user.toLowerCase();

        const schemaHint = `
Return ONLY minified JSON:
{
  "suggest": true|false,
  "category": "sofa|rug|table|chair|storage|lamp|bed|desk|shelving|nightstand|armchair|stool|bench|dresser|wardrobe|sideboard|tv-stand|coffee-table|dining-table",
  "style_tags": ["scandi","modern","industrial","boho","traditional","minimal","mid-century","japandi"],
  "budget_min": 0,
  "budget_max": 0,
  "max_width_cm": 0,
  "max_depth_cm": 0,
  "max_height_cm": 0
}
Rules:
- Return plain numbers for all numeric fields (no units or commas).
- Only set "suggest": true if the user is clearly asking for a recommendation or to see items, or if they specify constraints (budget/size).
- Do NOT set "suggest": true for brand overviews, comparisons, or policy/logistics questions unless they explicitly ask to recommend/show/pick.
- No extra text. No backticks. JSON only.`;

        let spec = {};
        try {
          const res = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: [
              { role: 'user', parts: [{ text: `System: ${SYSTEM_PROMPT}` }] },
              { role: 'user', parts: [{ text: `${user}\n\n${schemaHint}` }] }
            ],
            generationConfig: { responseMimeType: 'application/json' }
          });
          const txt = typeof res.response?.text === 'function'
            ? await res.response.text()
            : (res.response?.text || '{}');
          spec = JSON.parse(txt);
        } catch {
          spec = {};
        }

        // Heuristic backups & normalization
        if (typeof spec !== 'object' || spec === null) spec = {};
        if (typeof spec.suggest !== 'boolean') spec.suggest = false;
        if (typeof spec.category !== 'string') spec.category = '';

        const catGuess = inferCategory(lower);
        const intentGuess = inferIntent(lower);
        const budgetGuess = !!extractBudget(user);
        const sizeGuess = hasSizeHints(user);
        const brandInfo = brandInfoLike(user);

        // Fill missing category if we can infer it
        if (!spec.category && catGuess) spec.category = catGuess;

        // ----- FIX: don't auto-suggest from category alone -----
        // If model didn't set suggest, only enable it when we see intent OR constraints.
        if (!spec.suggest) {
          const hasConstraint = budgetGuess || sizeGuess;
          const hasCategory = !!(spec.category || catGuess);

          if (brandInfo) {
         spec.suggest = false; // force advice-only
        } else if (intentGuess) {
            spec.suggest = true;  // explicit "recommend/show/pick/find"
         }    else if (hasConstraint && hasCategory) {
        spec.suggest = true;  // e.g., "sofa under $900" (constraint + category)
          } else {
    spec.suggest = false; // e.g., "my budget is $900" (no category/intent)
  }
        }

        // If this reads like a brand/comparison/policy question, force advice-only
        if (brandInfo) {
          spec.suggest = false;
        }

        // Style array
        const styleArr = Array.isArray(spec.style_tags)
          ? spec.style_tags
          : (spec.style_tags ? [spec.style_tags] : []);

        // --- Budget normalization & fallback ---
        let budget_min = 0;
        let budget_max = 0;

        if (spec.budget_min != null) budget_min = toNumberLoose(spec.budget_min);
        if (spec.budget_max != null) budget_max = toNumberLoose(spec.budget_max);

        if ((!budget_min && !budget_max) && spec.budget && (spec.budget.min != null || spec.budget.max != null)) {
          budget_min = toNumberLoose(spec.budget.min);
          budget_max = toNumberLoose(spec.budget.max);
        }

        if (!budget_min && !budget_max) {
          const ext = extractBudget(user);
          if (ext) { budget_min = ext.min || 0; budget_max = ext.max || 0; }
        }

        budget_min = Math.max(0, budget_min);
        budget_max = Math.max(0, budget_max);
        if (budget_min && budget_max && budget_min > budget_max) {
          const tmp = budget_min; budget_min = budget_max; budget_max = tmp;
        }

        // Dimensions mapping (model uses depth/width/height; UE uses Length/Width/Height)
        const max_len = Number(spec.max_depth_cm || spec.max_length_cm || 0); // map to Length
        const max_w   = Number(spec.max_width_cm  || 0);
        const max_h   = Number(spec.max_height_cm || 0);

        // Build flat payload for Blueprints
        const flat =
          `suggest=${spec.suggest ? 1 : 0};` +
          `category=${sanitizeField(spec.category)};` +
          `style=${sanitizeField(styleArr.join('|'))};` +
          `budget_min=${budget_min};` +
          `budget_max=${budget_max};` +
          `max_len=${max_len};` +
          `max_w=${max_w};` +
          `max_h=${max_h};` +
          `choice_id=;` +         // left empty for large catalogs (UE will pick)
          `choice_name=`;

        if (ws.readyState === ws.OPEN) ws.send('SPEC|' + flat);
        return;
      }

      // ---- USER branch: streamed chat reply (CHUNK|..., FINAL|...) ----
      if (!raw.startsWith('USER|')) return;
      const user = raw.slice(5);

      // Optional nudge: policy-ish questions -> salesperson tone
      const policyish = /\b(return|warranty|delivery|shipping|lead\s*time|availability|exchange|refund|policy)\b/i.test(user);
      const userForModel = policyish ? `[BRAND_TONE]\n${user}` : user;

      const hist = histories.get(ws) || [];
      const contents = buildContents(hist, userForModel);

      const stream = await ai.models.generateContentStream({
        model: 'gemini-2.5-flash',
        contents
      });

      let full = '';
      for await (const chunk of stream) {
        const piece = chunk.text || '';
        if (piece) {
          full += piece;
          if (ws.readyState === ws.OPEN) ws.send('CHUNK|' + piece);
        }
      }

      // Save turn to history (user, then assistant)
      hist.push({ role: 'user',  parts: [{ text: user }] });
      hist.push({ role: 'model', parts: [{ text: full }] });
      clampHistory(hist);
      histories.set(ws, hist);

      if (ws.readyState === ws.OPEN) ws.send('FINAL|' + full);

    } catch (err) {
      if (ws.readyState === ws.OPEN) {
        ws.send('ERROR|' + (err?.message || 'Server error'));
      }
    }
  });

  ws.on('close', () => histories.delete(ws));
});

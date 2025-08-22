// index.js  (Node 20+)
// Requires: npm i ws @google/genai dotenv
import 'dotenv/config';
import { WebSocketServer } from 'ws';
import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const PORT = process.env.PORT || 3001;

// --- System prompt (with CATALOG_NO_MATCH behavior) ---
const SYSTEM_PROMPT = `
You are “Roomie,” a practical interior-design assistant.
- Ask 1–3 clarifying questions if needed (room size in m² or WxL in cm, light direction, budget, style, pets/kids).
- Use cm; check clearances (60–90 cm walkways; ~60 cm per dining seat); common rugs 160×230, 200×300, 240×340.
- Offer budget/mid/premium options; keep brands generic unless asked.
- Recommend durable materials (performance fabric, removable covers) and palettes suited to north/south light.
- Reply with concise bullets, then a short summary.
- If you receive a message starting with [CATALOG_NO_MATCH], apologize briefly, explain no items in the in-game catalog matched the constraints, and ask the user which constraints to relax (size, budget, style, color, material).
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
  'suggest','recommend','pick','choose','find','show me','need a','looking for','buy','get a','find me','any sofa','any rug'
];
const inferCategory = (txt) => {
  for (const cat of Object.keys(CATEGORY_MAP)) {
    if (txt.includes(cat)) return cat;
    for (const syn of CATEGORY_MAP[cat]) if (txt.includes(syn)) return cat;
  }
  return '';
};
const inferIntent = (txt) => INTENT_WORDS.some(w => txt.includes(w));

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
  const t = String(text || '').toLowerCase();
  const clean = (s) => s.replace(/gel|usd|eur|gbp|₾|\$|€|£/g, '').replace(/,/g, '').trim();

  // between / from ... to ...
  let m = t.match(/(?:between|from)\s+([\d\.\,k]+)\s*(?:to|and|-)\s*([\d\.\,k]+)/i);
  if (m) return { min: toNumberLoose(clean(m[1])), max: toNumberLoose(clean(m[2])) };

  // 500-800
  m = t.match(/([\d\.\,k]+)\s*-\s*([\d\.\,k]+)/);
  if (m) return { min: toNumberLoose(clean(m[1])), max: toNumberLoose(clean(m[2])) };

  // under / up to / max / <= / less than
  m = t.match(/(?:under|up\s*to|upto|max(?:imum)?|<=|less\s+than)\s+([\d\.\,k]+)/);
  if (m) return { min: 0, max: toNumberLoose(clean(m[1])) };

  // over / at least / minimum / >= / more than
  m = t.match(/(?:over|at\s*least|minimum|>=|more\s+than|min)\s+([\d\.\,k]+)/);
  if (m) return { min: toNumberLoose(clean(m[1])), max: 0 };

  // "budget 1200", "around 900" → treat as max cap
  m = t.match(/(?:budget|around|about|~)\s+([\d\.\,k]+)/);
  if (m) return { min: 0, max: toNumberLoose(clean(m[1])) };

  return null;
}

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
- If the user is NOT asking for a product recommendation, set "suggest": false and leave other fields default.
- If the user IS asking for a product, set "suggest": true and fill what you can (category must be a single canonical value above).
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
        if (!spec.suggest && (intentGuess || catGuess)) spec.suggest = true;
        if (!spec.category && catGuess) spec.category = catGuess;

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
        // console.log('SPEC flat ->', flat);
        return;
      }

      // ---- USER branch: streamed chat reply (CHUNK|..., FINAL|...) ----
      if (!raw.startsWith('USER|')) return;
      const user = raw.slice(5);

      const hist = histories.get(ws) || [];
      const contents = buildContents(hist, user);

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

// index.js  (Node 20+)
// Requires: npm i ws @google/genai dotenv
import "dotenv/config";
import { WebSocketServer } from "ws";
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const PORT = process.env.PORT || 3001;

// --- System prompt (unified salesperson tone + one question) ---
const SYSTEM_PROMPT = `
You are “Roomie,” a friendly, engaging, professional interior-design assistant and showroom salesperson for home interiors.
Your goal is to help the user make confident purchase decisions while giving creative, practical design advice.

Behavior
- Always keep a supportive salesperson tone (never pushy).
- Ask exactly one clarifying question per message. End each reply with exactly one question (max one “?” in the whole message).
- Be concise: use short bullets, then a one-sentence summary. keep texts medium-sized and engaging, don't risk boring a customer with long text.
- Use centimeters; mind clearances (60–90 cm walkways; ~60 cm per dining seat) and common rug sizes (160×230, 200×300, 240×340).
- Do not overpromise availability; defer to the in-game catalog and previews shown in the app.
- Offer budget/mid/premium options when relevant; keep brands generic unless asked.

UI coordination
- If a message begins with [PREVIEW_SHOWN ...], only briefly acknowledge the item (name/price).
- If a message begins with [CART_UPDATED ...], confirm the change and ask one helpful follow-up.
- If a message begins with [CHECKOUT_STARTED], guide the user through confirming their cart in 1–2 brief turns (still one question per reply).
- If a message begins with [ORDER_CREATED ...], congratulate and offer next steps (receipt, tracking, continue browsing).
- If a message begins with [CATALOG_NO_MATCH], apologize briefly and ask which single constraint to relax (size, budget, style, color, material).
- if a message begins with [ITEM_FOCUS ...]:
      1. Treat this as the item the user wants to modify. Acknowledge it briefly by it's name and price.
      2. Ask exactly ONE question: “Would you like to configure the model?” 
      3. Keep replies short and focused on that path.
`;

// --- Per-socket short history (Gemini contents-style) ---
const histories = new WeakMap(); // ws -> [{role:'user'|'model', parts:[{text}]}...]

const wss = new WebSocketServer({ port: PORT }, () =>
  console.log(`roomie-gemini-ws listening on ${PORT}`)
);

// Build contents for Gemini from history + new user text
const buildContents = (hist, userText) => {
  const contents = [];
  contents.push({
    role: "user",
    parts: [{ text: `System: ${SYSTEM_PROMPT}` }],
  });
  for (const m of hist) contents.push(m);
  contents.push({ role: "user", parts: [{ text: userText }] });
  return contents;
};

// Clamp history to last ~6 turns
const clampHistory = (hist, maxPairs = 6) => {
  const maxMsgs = maxPairs * 2;
  while (hist.length > maxMsgs) hist.shift();
};

// --- Heuristics for SPEC backups (manual, predictable) ---
const CATEGORY_MAP = {
  sofa: ["couch", "settee", "sectional", "loveseat"],
  rug: ["carpet"],
  armchair: ["accent chair", "reading chair"],
  "coffee-table": ["coffee table", "center table"],
  "dining-table": ["dining table"],
  wardrobe: ["closet"],
  lamp: ["floor lamp", "table lamp", "light"],
  bed: ["queen bed", "king bed", "double bed"],
  shelving: ["shelf", "bookcase"],
  nightstand: ["bedside table"],
  sideboard: ["buffet"],
  "tv-stand": ["media console", "tv unit"],
  chair: ["dining chair", "desk chair", "office chair"],
  desk: ["work desk", "office desk"],
};
const INTENT_WORDS = [
  "suggest",
  "recommend",
  "pick",
  "choose",
  "find",
  "show me",
  "find me",
  "replace",
  "any sofa",
  "any rug",
  "show a",
  "show me a",
  "show me some",
  "which would you recommend",
  "i need a",
  "i need an",
];
const inferCategory = (txt) => {
  for (const cat of Object.keys(CATEGORY_MAP)) {
    if (txt.includes(cat)) return cat;
    for (const syn of CATEGORY_MAP[cat]) if (txt.includes(syn)) return cat;
  }
  return "";
};
const inferIntent = (txt) => INTENT_WORDS.some((w) => txt.includes(w));

// Treat these as info/comparison/logistics → do NOT auto-suggest
const brandInfoLike = (t) =>
  /\b(tell me about|what sets|what makes|how.*different|compare|comparison|pros|cons|policy|return|warranty|delivery|shipping|lead\s*time|availability)\b/i.test(
    t
  );

// Size-ish hints (loosely)
const hasSizeHints = (t) =>
  /\b(width|length|depth|height|cm|mm|\d{2,3}\s*x\s*\d{2,3})\b/i.test(t);

// --- Helpers: numeric normalization ---
function toNumberLoose(x) {
  if (typeof x === "number") return x;
  if (typeof x !== "string") return 0;
  let s = x.trim().toLowerCase();
  s = s.replace(
    /gel|usd|eur|gbp|try|aud|cad|inr|jpy|cny|rmb|yuan|yen|tl|lira|dollars?|bucks?|quid|₾|\$|€|£|¥|₹|₺|₽|₩/g,
    ""
  );
  s = s.replace(/,/g, "").replace(/\s+/g, "");
  if (s.endsWith("k")) s = String(parseFloat(s) * 1000);
  const v = parseFloat(s);
  return Number.isFinite(v) ? v : 0;
}

function extractBudget(text) {
  const raw = String(text || "")
    .toLowerCase()
    .replace(/[—–]/g, "-");

  // Strip currency words/symbols and commas; normalize spaces
  let t = raw
    .replace(
      /gel|usd|eur|gbp|try|aud|cad|inr|jpy|cny|rmb|yuan|yen|tl|lira|dollars?|bucks?|quid|₾|\$|€|£|¥|₹|₺|₽|₩/g,
      ""
    )
    .replace(/,/g, "")
    .replace(/\s+/g, " ")
    .trim();

  // Collapse digit-grouping spaces (e.g., "1 000" -> "1000", "1 000 000" -> "1000000")
  t = t.replace(/(\d)\s+(?=\d)/g, "$1");

  const num = "([0-9]*\\.?[0-9]+k?)"; // supports 1.2k
  const money = "(?:budget|price|cost|amount|spend|limit|cap)";
  const optIs = "(?:is|=|:)\\s*"; // NEW: allow "is", "=", ":" between cue and number

  const toN = (s) => {
    let v = s;
    if (v.endsWith("k")) v = String(parseFloat(v) * 1000);
    const f = parseFloat(v);
    return Number.isFinite(f) ? f : 0;
  };

  let m;

  // ---- RANGES ----
  m = t.match(
    new RegExp(`(?:between|from)\\s+${num}\\s*(?:to|and|-)\\s*${num}`, "i")
  );
  if (m) return { min: toN(m[1]), max: toN(m[2]) };
  m = t.match(new RegExp(`${num}\\s*[-–]\\s*${num}`));
  if (m) return { min: toN(m[1]), max: toN(m[2]) };

  // ---- MIN before number ----
  // e.g., "minimum budget is 1200", "at least 1200", ">= 1200", "no less than 1200"
  m = t.match(
    new RegExp(
      `(?:min(?:imum)?|at\\s*least|>=|more\\s*than|no\\s*less\\s*than)\\s*(?:${money}\\s*)?(?:${optIs})?(?:of\\s*)?${num}`,
      "i"
    )
  );
  if (m) return { min: toN(m[1]), max: 0 };

  // ---- MAX before number ----
  // e.g., "maximum budget is 1000", "under 1000", "up to 1000", "<= 1000", "at most 1000"
  m = t.match(
    new RegExp(
      `(?:max(?:imum)?|under|below|up\\s*to|upto|<=|less\\s*than|no\\s*more\\s*than|at\\s*most|cap(?:ped)?(?:\\s*at)?)\\s*(?:${money}\\s*)?(?:${optIs})?(?:of\\s*)?${num}`,
      "i"
    )
  );
  if (m) return { min: 0, max: toN(m[1]) };

  // ---- Reversed order (number THEN cue) ----
  // "1200 minimum", "1200 or more", "900 max", "900 or less", "900 up to"
  m = t.match(
    new RegExp(
      `${num}\\s*(?:min(?:imum)?|at\\s*least|or\\s*more|\\+|and\\s*up)`,
      "i"
    )
  );
  if (m) return { min: toN(m[1]), max: 0 };
  m = t.match(
    new RegExp(
      `${num}\\s*(?:max(?:imum)?|or\\s*less|up\\s*to|at\\s*most|cap(?:ped)?(?:\\s*at)?)`,
      "i"
    )
  );
  if (m) return { min: 0, max: toN(m[1]) };

  // ---- Single-number hints → default to MAX cap ----
  // "budget is 1000", "price: 1500", "around 1200"
  m = t.match(
    new RegExp(`(?:${money})\\s*(?:${optIs})?(?:of\\s*)?${num}`, "i")
  );
  if (m) return { min: 0, max: toN(m[1]) };
  m = t.match(new RegExp(`(?:around|about|~)\\s*${num}`, "i"));
  if (m) return { min: 0, max: toN(m[1]) };

  return null;
}

// sanitize for flat payload (avoid breaking on ';' or '|')
const sanitizeField = (s) => String(s ?? "").replace(/[;|]/g, "/");

wss.on("connection", (ws) => {
  histories.set(ws, []);

  ws.on("message", async (data) => {
    const raw = data.toString();

    try {
      // ---- SPEC branch: flat key=value;... (no JSON plugin needed in UE) ----
      if (raw.startsWith("SPEC|")) {
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
- Only set "suggest": true if the user is clearly asking for a recommendation or to see items, or if they specify constraints (budget/size) alongside a product category (e.g., "sofa under 900").
- Do NOT set "suggest": true for general brand overviews, comparisons, or policy/logistics questions unless they explicitly ask to recommend/show/pick.
- No extra text. No backticks. JSON only.`.trim();

        let spec = {};
        try {
          const res = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: [
              { role: "user", parts: [{ text: `System: ${SYSTEM_PROMPT}` }] },
              { role: "user", parts: [{ text: `${user}\n\n${schemaHint}` }] },
            ],
            generationConfig: { responseMimeType: "application/json" },
          });
          const txt =
            typeof res.response?.text === "function"
              ? await res.response.text()
              : res.response?.text || "{}";
          spec = JSON.parse(txt);
        } catch {
          spec = {};
        }

        // Heuristic backups & normalization (manual, predictable)
        if (typeof spec !== "object" || spec === null) spec = {};
        if (typeof spec.suggest !== "boolean") spec.suggest = false;
        if (typeof spec.category !== "string") spec.category = "";

        const catGuess = inferCategory(lower);
        const intentGuess = inferIntent(lower);
        const budgetGuess = !!extractBudget(user);
        const sizeGuess = hasSizeHints(user);
        const brandInfo = brandInfoLike(user); // info/comparison/logistics

        if (!spec.category && catGuess) spec.category = catGuess;

        // Strict suggest gate:
        // - suggest only if explicit intent OR (constraints + category)
        // - never suggest on brand/comparison/policy questions
        if (!spec.suggest) {
          const hasConstraint = budgetGuess || sizeGuess;
          const hasCategory = !!(spec.category || catGuess);

          if (brandInfo) {
            spec.suggest = false;
          } else if (intentGuess) {
            spec.suggest = true;
          } else if (hasConstraint && hasCategory) {
            spec.suggest = true;
          } else {
            spec.suggest = false;
          }
        }

        // Style array
        const styleArr = Array.isArray(spec.style_tags)
          ? spec.style_tags
          : spec.style_tags
          ? [spec.style_tags]
          : [];

        // --- Budget normalization & fallback ---
        let budget_min = 0,
          budget_max = 0;

        if (spec.budget_min != null)
          budget_min = toNumberLoose(spec.budget_min);
        if (spec.budget_max != null)
          budget_max = toNumberLoose(spec.budget_max);

        // Backfill from raw user text
        const ext = extractBudget(user);
        if (ext) {
          if (!budget_min && ext.min) budget_min = ext.min;
          if (!budget_max && ext.max) budget_max = ext.max;
        }

        // Optional legacy object
        if (
          spec.budget &&
          (spec.budget.min != null || spec.budget.max != null)
        ) {
          if (!budget_min && spec.budget.min != null)
            budget_min = toNumberLoose(spec.budget.min);
          if (!budget_max && spec.budget.max != null)
            budget_max = toNumberLoose(spec.budget.max);
        }

        // Sanitize
        budget_min = Math.max(0, budget_min);
        budget_max = Math.max(0, budget_max);
        if (budget_min && budget_max && budget_min > budget_max) {
          const tmp = budget_min;
          budget_min = budget_max;
          budget_max = tmp;
        }

        // Dimensions mapping (model uses depth/width/height; UE uses Length/Width/Height)
        const max_len = Number(spec.max_depth_cm || spec.max_length_cm || 0); // map to Length
        const max_w = Number(spec.max_width_cm || 0);
        const max_h = Number(spec.max_height_cm || 0);

        // Build flat payload for Blueprints
        const flat =
          `suggest=${spec.suggest ? 1 : 0};` +
          `category=${sanitizeField(spec.category)};` +
          `style=${sanitizeField(styleArr.join("|"))};` +
          `budget_min=${budget_min};` +
          `budget_max=${budget_max};` +
          `max_len=${max_len};` +
          `max_w=${max_w};` +
          `max_h=${max_h};` +
          `choice_id=;` +
          `choice_name=`;

        if (ws.readyState === ws.OPEN) ws.send("SPEC|" + flat);
        return;
      }

      // ---- USER branch: streamed chat reply (CHUNK|..., FINAL|...) ----
      if (!raw.startsWith("USER|")) return;
      const user = raw.slice(5);

      // Unified salesperson tone (no brand/policy nudge needed)
      const hist = histories.get(ws) || [];
      const contents = buildContents(hist, user);

      const stream = await ai.models.generateContentStream({
        model: "gemini-2.5-flash",
        contents,
      });

      let full = "";
      for await (const chunk of stream) {
        const piece = chunk.text || "";
        if (piece) {
          full += piece;
          if (ws.readyState === ws.OPEN) ws.send("CHUNK|" + piece);
        }
      }

      // Save turn to history (user, then assistant)
      hist.push({ role: "user", parts: [{ text: user }] });
      hist.push({ role: "model", parts: [{ text: full }] });
      clampHistory(hist);
      histories.set(ws, hist);

      if (ws.readyState === ws.OPEN) ws.send("FINAL|" + full);
    } catch (err) {
      if (ws.readyState === ws.OPEN) {
        ws.send("ERROR|" + (err?.message || "Server error"));
      }
    }
  });

  ws.on("close", () => histories.delete(ws));
});

// index.js  (Node 20+)
// Requires: npm i ws @google/genai dotenv
import "dotenv/config";
import { WebSocketServer } from "ws";
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const PORT = process.env.PORT || 3001;

// -------- anti-overload helpers --------
const MAX_RETRIES = 4;
const BASE_DELAY_MS = 400;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const jitter = (ms) => ms + Math.floor((Math.random() * 2 - 1) * (ms * 0.25));
const isRetryable = (err) => {
  const s = err?.status ?? err?.code ?? 0;
  const msg = (err?.message || "").toLowerCase();
  return s === 429 || s === 500 || s === 502 || s === 503 || s === 504 ||
         msg.includes("overload") || msg.includes("unavailable") || msg.includes("temporar");
};
const logErr = (where, err) => {
  const s = err?.status ?? err?.code ?? "n/a";
  console.error(`[${where}] status=${s} ${err?.message || err}`);
};

// Models — lighter for JSON, higher quality for chat
const CHAT_MODEL = "gemini-2.5-flash";
const SPEC_MODEL = "gemini-1.5-flash";    // cheaper/lighter for JSON structs

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
- If a message begins with [PREVIEW_SHOWN ...], only briefly acknowledge the item (name/price), don't follow up.
- If a message begins with [CART_UPDATED ...], confirm the change and ask one helpful follow-up.
- If a message begins with [CHECKOUT_STARTED], guide the user through confirming their cart in 1–2 brief turns (still one question per reply).
- If a message begins with [ORDER_CREATED ...], congratulate and offer next steps (receipt, tracking, continue browsing).
- If a message begins with [CATALOG_NO_MATCH], apologize briefly and ask which single constraint to relax (size, budget, style, color, material).
- If a message begins with [ITEM_FOCUS ...]:
    1) Treat this as the item the user wants to modify. Acknowledge it briefly by name and price.
    2) Ask exactly ONE question that separates paths: “Would you like to change the material or replace the model?”
    3) Keep replies short and focused on that path. Do not list multiple products; the app will show one preview/apply action.
- If a message begins with [FOCUS_CLEAR ...], [PLACED ...], or [REPLACED ...], do not reply. Treat it as a silent UI control signal and stop referring to a specific item.
- If a message begins with [MATERIAL_CHANGED ...], acknowledge the new finish/color and ask one brief follow-up (e.g., “Keep the legs in black matte or try brass?”).

General
- If the user says “replace this…” or “make this …” without a focused item, ask which item they mean before proceeding.
`;

// --- Per-socket short history (Gemini contents-style) ---
const histories = new WeakMap(); // ws -> [{role:'user'|'model', parts:[{text}]}...]
const chatBusy = new WeakMap();   // ws -> boolean (serialize chat streams)

const wss = new WebSocketServer({ port: PORT }, () =>
  console.log(`roomie-gemini-ws listening on ${PORT}`)
);

// Build contents for Gemini from history + new user text
const buildContents = (hist, userText) => {
  const contents = [];
  contents.push({ role: "user", parts: [{ text: `System: ${SYSTEM_PROMPT}` }] });
  for (const m of hist) contents.push(m);
  contents.push({ role: "user", parts: [{ text: userText }] });
  return contents;
};

// Clamp history to last ~4 turns (trim tokens)
const clampHistory = (hist, maxPairs = 4) => {
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
  "suggest","recommend","pick","choose","find","show me","find me","replace",
  "any sofa","any rug","show a","show me a","show me some",
  "which would you recommend","i need a","i need an",
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
  /\b(tell me about|what sets|what makes|how.*different|compare|comparison|pros|cons|policy|return|warranty|delivery|shipping|lead\s*time|availability)\b/i.test(t);

// Size-ish hints (loosely)
const hasSizeHints = (t) =>
  /\b(width|length|depth|height|cm|mm|\d{2,3}\s*x\s*\d{2,3})\b/i.test(t);

// --- Intent helpers (server-side) ---
const MATERIAL_WORDS = [
  "material","fabric","leather","linen","velvet","wool","cotton","wood","oak","walnut","ash","veneer",
  "metal","brass","chrome","steel","iron","aluminum","glass","marble","stone","rattan","wicker",
  "finish","matte","satin","gloss","brushed","oiled","stain","lacquer","color","colour",
  "black","white","gray","grey","beige","cream","sand","tan","charcoal","navy","green","brown",
];
const REPLACE_STRONG = ["replace","swap","alternative","another","something else","different model","other option"];
const REPLACE_SOFT = ["cheaper","less expensive","budget","pricier","premium","smaller","bigger","narrower","wider","shorter","taller","compact"];
const containsAny = (hay, arr) => arr.some((w) => hay.includes(w));
const isMaterialIntent = (lc) => containsAny(lc, MATERIAL_WORDS);
const isReplaceIntent  = (lc) => containsAny(lc, REPLACE_STRONG) ||
                               ((lc.includes(" this") || lc.includes(" it ")) && containsAny(lc, REPLACE_SOFT));

// --- Focus state per socket ---
const focusState = new WeakMap(); // ws -> { active: boolean }
const setFocus = (ws, on) => {
  const st = focusState.get(ws) || { active: false };
  st.active = !!on;
  focusState.set(ws, st);
};
const hasFocus = (ws) => !!focusState.get(ws)?.active;

// --- Helpers: numeric normalization ---
function toNumberLoose(x) {
  if (typeof x === "number") return x;
  if (typeof x !== "string") return 0;
  let s = x.trim().toLowerCase();
  s = s.replace(/gel|usd|eur|gbp|try|aud|cad|inr|jpy|cny|rmb|yuan|yen|tl|lira|dollars?|bucks?|quid|₾|\$|€|£|¥|₹|₺|₽|₩/g, "");
  s = s.replace(/,/g, "").replace(/\s+/g, "");
  if (s.endsWith("k")) s = String(parseFloat(s) * 1000);
  const v = parseFloat(s);
  return Number.isFinite(v) ? v : 0;
}
function extractBudget(text) {
  const raw = String(text || "").toLowerCase().replace(/[—–]/g, "-");
  let t = raw.replace(/gel|usd|eur|gbp|try|aud|cad|inr|jpy|cny|rmb|yuan|yen|tl|lira|dollars?|bucks?|quid|₾|\$|€|£|¥|₹|₺|₽|₩/g, "")
             .replace(/,/g, "").replace(/\s+/g, " ").trim();
  t = t.replace(/(\d)\s+(?=\d)/g, "$1");
  const num = "([0-9]*\\.?[0-9]+k?)", money="(?:budget|price|cost|amount|spend|limit|cap)", optIs="(?:is|=|:)\\s*";
  const toN=(s)=>{let v=s; if(v.endsWith("k")) v=String(parseFloat(v)*1000); const f=parseFloat(v); return Number.isFinite(f)?f:0;};
  let m;
  m = t.match(new RegExp(`(?:between|from)\\s+${num}\\s*(?:to|and|-)\\s*${num}`,"i")); if (m) return {min:toN(m[1]),max:toN(m[2])};
  m = t.match(new RegExp(`${num}\\s*[-–]\\s*${num}`));                            if (m) return {min:toN(m[1]),max:toN(m[2])};
  m = t.match(new RegExp(`(?:min(?:imum)?|at\\s*least|>=|more\\s*than|no\\s*less\\s*than)\\s*(?:${money}\\s*)?(?:${optIs})?(?:of\\s*)?${num}`,"i"));
  if (m) return {min:toN(m[1]),max:0};
  m = t.match(new RegExp(`(?:max(?:imum)?|under|below|up\\s*to|upto|<=|less\\s*than|no\\s*more\\s*than|at\\s*most|cap(?:ped)?(?:\\s*at)?)\\s*(?:${money}\\s*)?(?:${optIs})?(?:of\\s*)?${num}`,"i"));
  if (m) return {min:0,max:toN(m[1])};
  m = t.match(new RegExp(`${num}\\s*(?:min(?:imum)?|at\\s*least|or\\s*more|\\+|and\\s*up)`,"i"));
  if (m) return {min:toN(m[1]),max:0};
  m = t.match(new RegExp(`${num}\\s*(?:max(?:imum)?|or\\s*less|up\\s*to|at\\s*most|cap(?:ped)?(?:\\s*at)?)`,"i"));
  if (m) return {min:0,max:toN(m[1])};
  m = t.match(new RegExp(`(?:${money})\\s*(?:${optIs})?(?:of\\s*)?${num}`,"i"));  if (m) return {min:0,max:toN(m[1])};
  m = t.match(new RegExp(`(?:around|about|~)\\s*${num}`,"i"));                   if (m) return {min:0,max:toN(m[1])};
  return null;
}

// sanitize for flat payload (avoid breaking on ';' or '|')
const sanitizeField = (s) => String(s ?? "").replace(/[;|]/g, "/");

// ----------------- SPEC builder -----------------
async function makeSpecFlat(user) {
  const lower = user.toLowerCase();
  const schemaHint = `
Return ONLY minified JSON:
{
  "suggest": true|false,
  "category": "sofa|rug|table|chair|storage|lamp|bed|desk|shelving|nightstand|armchair|stool|bench|dresser|wardrobe|sideboard|tv-stand|coffee-table|dining-table",
  "style_tags": ["scandi","modern","industrial","boho","traditional","minimal","mid-century","japandi"],
  "budget_min": 0, "budget_max": 0,
  "max_width_cm": 0, "max_depth_cm": 0, "max_height_cm": 0
}
Rules:
- Plain numbers only. "suggest": true only when the user explicitly asks to see/recommend/pick OR they give constraints (budget/size) with a category.
- Never suggest for brand/policy/comparison-only questions.
- JSON only.`.trim();

  let spec = {};
  for (let attempt=0; attempt<MAX_RETRIES; attempt++) {
    try {
      const res = await ai.models.generateContent({
        model: SPEC_MODEL,
        contents: [
          { role: "user", parts: [{ text: `System: ${SYSTEM_PROMPT}` }] },
          { role: "user", parts: [{ text: `${user}\n\n${schemaHint}` }] },
        ],
        config: { responseMimeType: "application/json" },
      });
      const txt = (res.text && res.text.trim()) ? res.text : "{}";
      spec = JSON.parse(txt);
      break;
    } catch (err) {
      if (attempt < MAX_RETRIES-1 && isRetryable(err)) {
        await sleep(jitter(BASE_DELAY_MS * Math.pow(2, attempt)));
        continue;
      }
      logErr("spec", err);
      break;
    }
  }

  if (typeof spec !== "object" || spec === null) spec = {};
  if (typeof spec.suggest !== "boolean") spec.suggest = false;
  if (typeof spec.category !== "string") spec.category = "";

  const catGuess = inferCategory(lower);
  const intentGuess = inferIntent(lower);
  const budgetGuess = !!extractBudget(user);
  const sizeGuess = hasSizeHints(user);
  const brandInfo = brandInfoLike(user);

  if (!spec.category && catGuess) spec.category = catGuess;

  const hasConstraint = budgetGuess || sizeGuess;
  const hasCategory = !!(spec.category || catGuess);
  if (brandInfo) spec.suggest = false;
  else if (!spec.suggest && (intentGuess || (hasConstraint && hasCategory))) spec.suggest = true;

  let styleArr = Array.isArray(spec.style_tags) ? spec.style_tags : (spec.style_tags ? [spec.style_tags] : []);
  let budget_min = 0, budget_max = 0;
  if (spec.budget_min != null) budget_min = toNumberLoose(spec.budget_min);
  if (spec.budget_max != null) budget_max = toNumberLoose(spec.budget_max);

  const ext = extractBudget(user);
  if (ext) {
    if (!budget_min && ext.min) budget_min = ext.min;
    if (!budget_max && ext.max) budget_max = ext.max;
  }
  if (spec.budget && (spec.budget.min != null || spec.budget.max != null)) {
    if (!budget_min && spec.budget.min != null) budget_min = toNumberLoose(spec.budget.min);
    if (!budget_max && spec.budget.max != null) budget_max = toNumberLoose(spec.budget.max);
  }

  budget_min = Math.max(0, budget_min);
  budget_max = Math.max(0, budget_max);
  if (budget_min && budget_max && budget_min > budget_max) {
    const tmp = budget_min; budget_min = budget_max; budget_max = tmp;
  }

  const max_len = Number(spec.max_depth_cm || spec.max_length_cm || 0);
  const max_w   = Number(spec.max_width_cm  || 0);
  const max_h   = Number(spec.max_height_cm || 0);

  return (
    `suggest=${spec.suggest ? 1 : 0};` +
    `category=${sanitizeField(spec.category)};` +
    `style=${sanitizeField(styleArr.join("|"))};` +
    `budget_min=${budget_min};` +
    `budget_max=${budget_max};` +
    `max_len=${max_len};` +
    `max_w=${max_w};` +
    `max_h=${max_h};` +
    `choice_id=;choice_name=`
  );
}

// ----------------- MATSPEC builder -----------------
const SLOT_CANON = [
  ["fabric", ["fabric","cloth","textile","upholstery"]],
  ["leather",["leather"]],
  ["linen",  ["linen"]],
  ["velvet", ["velvet"]],
  ["wool",   ["wool"]],
  ["cotton", ["cotton"]],
  ["wood",   ["wood","timber"]],
  ["oak",    ["oak"]],
  ["walnut", ["walnut"]],
  ["ash",    ["ash"]],
  ["metal",  ["metal","steel","iron","aluminum","aluminium"]],
  ["brass",  ["brass","gold","golden"]],
  ["chrome", ["chrome","silver","chromed"]],
  ["glass",  ["glass"]],
  ["stone",  ["stone","granite","slate","travertine"]],
  ["marble", ["marble"]],
  ["ceramic",["ceramic","tile"]],
  ["rattan", ["rattan","wicker","cane"]],
];
const COLOR_CANON = [
  ["black", ["black","jet","ink"]],
  ["white", ["white","ivory"]],
  ["gray",  ["gray","grey","graphite","charcoal","dark gray","dark-grey","darkgrey"]],
  ["beige", ["beige","cream","sand","tan"]],
  ["brown", ["brown","chocolate","walnut"]],
  ["green", ["green","forest","olive","sage","mint"]],
  ["blue",  ["blue","navy","cobalt","royal"]],
  ["red",   ["red","burgundy","crimson"]],
  ["brass", ["brass","gold","golden"]],
  ["chrome",["chrome","silver","steel"]],
];
const FINISH_CANON = [
  ["matte",       ["matte","matt"]],
  ["satin",       ["satin","eggshell","egg-shell","semi-matte"]],
  ["gloss",       ["gloss","glossy","high gloss","polished"]],
  ["brushed",     ["brushed"]],
  ["oiled",       ["oiled","oil finish"]],
  ["stained",     ["stain","stained"]],
  ["lacquered",   ["lacquer","lacquered"]],
  ["powdercoated",["powder","powder-coated","powdercoated"]],
  ["anodized",    ["anodized","anodised"]],
  ["plated",      ["plated","electroplated"]],
];
const MATERIAL_HINTS = /\b(material|fabric|textile|leather|linen|velvet|wool|cotton|wood|oak|walnut|ash|veneer|metal|brass|chrome|steel|iron|aluminum|glass|marble|stone|ceramic|rattan|wicker|finish|color|colour|stain|paint|lacquer|matte|satin|gloss|brushed|oiled|powder|anodized|plated)\b/i;
const stripLeadingTag = (s) => String(s||"").replace(/^\[[^\]]+\]\s*/, "");
const findCanon = (lc, table) => {
  for (const [canon, syns] of table) for (const s of syns) if (lc.includes(s)) return canon;
  return "";
};

async function makeMatSpecFlat(user) {
  const schemaHint = `
Return ONLY minified JSON:
{
  "apply": true|false,
  "slot": "fabric|leather|velvet|linen|wool|cotton|wood|oak|walnut|ash|metal|brass|chrome|steel|glass|stone|marble|ceramic|rattan",
  "color": "beige|black|white|gray|green|blue|red|brown|brass|chrome",
  "finish": "matte|satin|gloss|semi-gloss|brushed|oiled|stained|lacquered|powdercoated|anodized|plated",
  "style_tags": ["scandi","minimal","mid-century","industrial","boho","traditional","japandi"]
}
Rules:
- "apply": true ONLY for material/finish/color changes to the current item (not model replacement).
- Prefer one slot and concise descriptors. JSON only.`.trim();

  let spec = {};
  for (let attempt=0; attempt<MAX_RETRIES; attempt++) {
    try {
      const res = await ai.models.generateContent({
        model: SPEC_MODEL,
        contents: [
          { role: "user", parts: [{ text: `System: ${SYSTEM_PROMPT}` }] },
          { role: "user", parts: [{ text: `${user}\n\n${schemaHint}` }] },
        ],
        config: { responseMimeType: "application/json" },
      });
      const txt = (res.text && res.text.trim()) ? res.text : "{}";
      spec = JSON.parse(txt);
      break;
    } catch (err) {
      if (attempt < MAX_RETRIES-1 && isRetryable(err)) {
        await sleep(jitter(BASE_DELAY_MS * Math.pow(2, attempt)));
        continue;
      }
      logErr("matspec", err);
      break;
    }
  }

  const lc = stripLeadingTag(user).toLowerCase();
  let slot   = String(spec.slot   || "").toLowerCase().trim();
  let color  = String(spec.color  || "").toLowerCase().trim();
  let finish = String(spec.finish || "").toLowerCase().trim();
  if (!slot)   slot   = findCanon(lc, SLOT_CANON);
  if (!color)  color  = findCanon(lc, COLOR_CANON);
  if (!finish) finish = findCanon(lc, FINISH_CANON);

  const styleArr = Array.isArray(spec.style_tags) ? spec.style_tags : (spec.style_tags ? [spec.style_tags] : []);
  const style = styleArr.map((s)=>String(s).toLowerCase().replace(/[;|]/g,"/")).join("|");

  const apply = (typeof spec.apply === "boolean" ? spec.apply : false) || MATERIAL_HINTS.test(lc) || !!slot || !!color || !!finish;

  return (
    `apply=${apply ? 1 : 0};` +
    `slot=${slot.replace(/[;|]/g,"/")};` +
    `color=${color.replace(/[;|]/g,"/")};` +
    `finish=${finish.replace(/[;|]/g,"/")};` +
    `style=${style}`
  );
}

// ----------------- WebSocket handling -----------------
wss.on("connection", (ws) => {
  histories.set(ws, []);

  ws.on("message", async (data) => {
    const raw = data.toString();

    try {
      // ----- SPEC (explicit) -----
      if (raw.startsWith("SPEC|")) {
        const user = raw.slice(5);
        let flat = "";
        for (let attempt=0; attempt<MAX_RETRIES; attempt++) {
          try {
            flat = await makeSpecFlat(user);
            break;
          } catch (err) {
            if (attempt < MAX_RETRIES-1 && isRetryable(err)) {
              await sleep(jitter(BASE_DELAY_MS * Math.pow(2, attempt)));
              continue;
            }
            logErr("spec-explicit", err);
            break;
          }
        }
        if (flat && ws.readyState === ws.OPEN) ws.send("SPEC|" + flat);
        return;
      }

      // ----- MATSPEC (explicit) -----
      if (raw.startsWith("MATSPEC|")) {
        const user = raw.slice(8);
        let flat = "";
        for (let attempt=0; attempt<MAX_RETRIES; attempt++) {
          try {
            flat = await makeMatSpecFlat(user);
            break;
          } catch (err) {
            if (attempt < MAX_RETRIES-1 && isRetryable(err)) {
              await sleep(jitter(BASE_DELAY_MS * Math.pow(2, attempt)));
              continue;
            }
            logErr("matspec-explicit", err);
            break;
          }
        }
        if (flat && ws.readyState === ws.OPEN) ws.send("MATSPEC|" + flat);
        return;
      }

      // ----- USER: streamed chat reply -----
      if (!raw.startsWith("USER|")) return;
      const user = raw.slice(5);
      const trimmed = user.trim();
      const lc = user.toLowerCase();

      // Silent control tags — flip focus and return (no reply, no history, no SPEC/MATSPEC)
      if (/^\s*\[(focus_clear|placed|replaced)\b[^\]]*\]\s*$/i.test(trimmed)) {
        setFocus(ws, false);
        return;
      }

      // Focus tagging
      if (/^\[item_focus\b/i.test(trimmed)) setFocus(ws, true);

      // Per-socket serialization: one chat stream at a time
      if (chatBusy.get(ws)) return;
      chatBusy.set(ws, true);

      const hist = histories.get(ws) || [];
      const contents = buildContents(hist, user);

      let full = "";
      let gotAnyChunk = false;
      let lastErr = null;

      for (let attempt=0; attempt<MAX_RETRIES; attempt++) {
        try {
          const stream = await ai.models.generateContentStream({ model: CHAT_MODEL, contents });
          for await (const chunk of stream) {
            const piece = chunk.text || "";
            if (piece) {
              gotAnyChunk = true;
              full += piece;
              if (ws.readyState === ws.OPEN) ws.send("CHUNK|" + piece);
            }
          }
          lastErr = null;
          break; // success
        } catch (err) {
          lastErr = err;
          if (gotAnyChunk) break; // don't retry mid-stream
          if (attempt < MAX_RETRIES-1 && isRetryable(err)) {
            await sleep(jitter(BASE_DELAY_MS * Math.pow(2, attempt)));
            continue;
          }
          logErr("stream", err);
          break;
        }
      }

      // Optional hard fallback if nothing streamed
      if (!gotAnyChunk && lastErr) {
        try {
          const stream2 = await ai.models.generateContentStream({ model: "gemini-1.5-flash", contents });
          for await (const chunk of stream2) {
            const piece = chunk.text || "";
            if (piece) {
              full += piece;
              if (ws.readyState === ws.OPEN) ws.send("CHUNK|" + piece);
            }
          }
        } catch (e2) {
          logErr("stream-fallback", e2);
          if (ws.readyState === ws.OPEN) ws.send("ERROR|SERVICE_UNAVAILABLE");
        }
      }

      // Save turn to history (user, then assistant) only if we have a response
      if (full) {
        hist.push({ role: "user",  parts: [{ text: user }] });
        hist.push({ role: "model", parts: [{ text: full }] });
        clampHistory(hist);
        histories.set(ws, hist);
        if (ws.readyState === ws.OPEN) ws.send("FINAL|" + full);
      }

      // ---- Auto-dispatch SPEC/MATSPEC while focused ----
      // Skip pure tag-only lines (e.g., just [ITEM_FOCUS ...])
      const isTagOnly = /^\s*\[[^\]]+\]\s*$/.test(trimmed);
      if (hasFocus(ws) && !isTagOnly) {
        if (isMaterialIntent(lc)) {
          try {
            const flat = await makeMatSpecFlat(user);
            if (ws.readyState === ws.OPEN) ws.send("MATSPEC|" + flat);
          } catch (err) {
            logErr("matspec-auto", err);
          }
        } else if (isReplaceIntent(lc)) {
          try {
            const flat = await makeSpecFlat(user);
            if (ws.readyState === ws.OPEN) ws.send("SPEC|" + flat);
          } catch (err) {
            logErr("spec-auto", err);
          }
        }
      }

      chatBusy.set(ws, false);
    } catch (err) {
      chatBusy.set(ws, false);
      if (ws.readyState === ws.OPEN) ws.send("ERROR|" + (err?.message || "Server error"));
    }
  });

  ws.on("close", () => {
    chatBusy.delete(ws);
    histories.delete(ws);
  });
});

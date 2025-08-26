import express from "express";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

// App setup
const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const PORT = Number(process.env.PORT || 8000);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const GPT_MODEL = process.env.GPT_MODEL || "gpt-5-mini";

// Airtable env
const AIRTABLE_TOKEN   = process.env.AIRTABLE_TOKEN || process.env.AIRTABLE_API_KEY || "";
const AIRTABLE_BASE_ID = process.env.AIRTABLE_BASE_ID || "";
const AIRTABLE_TABLE   = process.env.AIRTABLE_TABLE || "Leads";
const AIRTABLE_ENABLED = Boolean(AIRTABLE_TOKEN && AIRTABLE_BASE_ID && AIRTABLE_TABLE);

// Logs d’init
console.log(`[Init] OpenAI ${OPENAI_API_KEY ? "ENABLED" : "DISABLED"} | Model=${GPT_MODEL}`);
console.log(`[Init] Airtable ${AIRTABLE_ENABLED ? "ENABLED" : "DISABLED"} | Base=${AIRTABLE_BASE_ID || "n/a"} Table=${AIRTABLE_TABLE || "n/a"}`);

let openai = null;
if (OPENAI_API_KEY) openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// ---- State ----
const SESSIONS = new Map();

const REQUIRED_FIELDS_ORDER = [
  "full_name", "email", "departure_city", "destination",
  "start_date", "end_date", "n_travelers", "budget", "interests",
];

const FIELD_LABELS_FR = {
  full_name: "votre nom complet",
  email: "votre adresse e-mail",
  departure_city: "votre ville de départ",
  destination: "la destination souhaitée",
  start_date: "la date de départ (YYYY-MM-DD)",
  end_date: "la date de retour (YYYY-MM-DD)",
  n_travelers: "le nombre de voyageurs",
  budget: "le budget total en €",
  interests: "vos centres d’intérêt (ex: musées, nature, gastronomie)",
  notes: "des notes complémentaires (optionnel)",
};

// ---- Regex ----
const EMAIL_RE = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/;
const DATE_RE  = /\b(20\d{2})[-\/](0[1-9]|1[0-2])[-\/](0[1-9]|[12]\d|3[01])\b/g; // global pour matchAll
const INT_RE   = /\b(\d{1,2})\s*(voyageur|voyageurs|pers|personnes?)?\b/i;
const MONEY_RE = /(\d[\d\s.,]*)(?:\s?€| eur| euros?)/i;

// ---- Name helpers / validation ----
const NAME_BAD_HINTS = /(je|j’|veux|souhaite|souhaiter|voyage|vers|pour|destination|aller|faire|réserver|billet|métro|train|avion)/i;

function inferNameFromEmail(email) {
  const local = String(email || "").split("@")[0];
  if (!local) return null;
  const parts = local.split(/[._-]+/).filter(Boolean).slice(0, 3);
  if (!parts.length) return null;
  const cap = parts.map(p => p.charAt(0).toUpperCase() + p.slice(1));
  return cap.join(" ");
}

function normalizeName(s) {
  // Garder lettres (avec accents), apostrophes, tirets, espaces
  const cleaned = String(s)
    .replace(/[^A-Za-zÀ-ÖØ-öø-ÿ' -]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const parts = cleaned.split(" ").filter(Boolean);
  if (parts.length < 2 || parts.length > 4) return null;
  const pretty = parts.map(p => p.charAt(0).toUpperCase() + p.slice(1).toLowerCase()).join(" ");
  if (pretty.length < 5) return null;
  return pretty;
}

function tryParseFullName(input, emailFallback) {
  const s = String(input || "").trim();
  if (!s) return [null, "Merci d’indiquer vos nom et prénom (ex: Hugo Grillon)."];
  if (NAME_BAD_HINTS.test(s) || s.length > 60) {
    // phrase d’intention → pas un nom
    const guess = inferNameFromEmail(emailFallback);
    const hint = guess ? ` Par exemple: "${guess}".` : "";
    return [null, `Indiquez uniquement votre nom et prénom.${hint}`];
  }
  const n = normalizeName(s);
  if (!n) {
    const guess = inferNameFromEmail(emailFallback);
    const hint = guess ? ` Par exemple: "${guess}".` : " Exemple: Hugo Grillon.";
    return [null, `Nom invalide.${hint}`];
  }
  return [n, null];
}

// ---- Dates / extraction ----
function tryParseDate(s) {
  if (!s) return null;
  const norm = String(s).trim().replaceAll("/", "-");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(norm)) return null;
  const d = new Date(norm);
  if (Number.isNaN(d.getTime())) return null;
  return norm;
}

function extractEntitiesFR(text) {
  const found = {};
  if (!text) return found;

  const mEmail = text.match(EMAIL_RE);
  if (mEmail) found.email = mEmail[0];

  const dMatches = [...text.matchAll(DATE_RE)].map((m) => `${m[1]}-${m[2]}-${m[3]}`);
  if (dMatches[0]) found.start_date = dMatches[0];
  if (dMatches[1]) found.end_date = dMatches[1];

  const mInt = text.match(INT_RE);
  if (mInt) {
    const n = parseInt(mInt[1], 10);
    if (Number.isInteger(n) && n >= 1 && n <= 20) found.n_travelers = n;
  }

  const mMoney = text.replace(/[\u202f\u00A0]/g, " ").match(MONEY_RE);
  if (mMoney) {
    const raw = mMoney[1].replaceAll(" ", "").replace(",", ".");
    const val = parseFloat(raw);
    if (!Number.isNaN(val) && val > 0) found.budget = val;
  }

  const tokens = text.split(/\s+/);
  for (let i = 0; i < tokens.length; i++) {
    const low = tokens[i].toLowerCase();
    if ((low === "depuis" || low === "de" || low === "from") && tokens[i + 1]) {
      const nxt = tokens[i + 1].replace(/[.,]/g, "").trim();
      if (nxt) found.departure_city = capitalize(nxt);
    }
    if ((low === "vers" || low === "pour" || low === "destination" || low === "to") && tokens[i + 1]) {
      const nxt = tokens[i + 1].replace(/[.,]/g, "").trim();
      if (nxt) found.destination = capitalize(nxt);
    }
  }
  return found;
}

// ========= Airtable helpers =========
const AIRTABLE_API_BASE = () =>
  `https://api.airtable.com/v0/${AIRTABLE_BASE_ID}/${encodeURIComponent(AIRTABLE_TABLE)}`;

const airtableHeaders = () => ({
  Authorization: `Bearer ${AIRTABLE_TOKEN}`,
  "Content-Type": "application/json",
});
const escFormula = (s = "") => String(s).replace(/"/g, '\\"');

function toAirtableFields(d) {
  return {
    "Full Name": d.full_name ?? "",
    "Email": d.email ?? "",
    "Departure City": d.departure_city ?? "",
    "Destination": d.destination ?? "",
    "Start Date": d.start_date ?? "",
    "End Date": d.end_date ?? "",
    "Travelers": typeof d.n_travelers === "number" ? d.n_travelers : (d.n_travelers ? Number(d.n_travelers) : null),
    "Budget (EUR)": typeof d.budget === "number" ? d.budget : (d.budget ? Number(d.budget) : null),
    "Interests": Array.isArray(d.interests) ? d.interests.join(", ") : (d.interests ?? ""),
    "Notes": d.notes ?? "",
    "User Phrase": d.user_phrase ?? "", // <-- on conserve la phrase libre
    "Source": "Chatbot",
    "Created At": new Date().toISOString(),
  };
}

async function airtableUpsert(d) {
  if (!AIRTABLE_ENABLED) return { ok: false, reason: "disabled" };
  const baseUrl = AIRTABLE_API_BASE();
  const headers = airtableHeaders();

  const email = d.email || "";
  const sd = d.start_date || "";
  const dest = d.destination || "";
  const formula = `AND({Email}="${escFormula(email)}",{Start Date}="${escFormula(sd)}",{Destination}="${escFormula(dest)}")`;
  const searchUrl = baseUrl + `?maxRecords=1&filterByFormula=${encodeURIComponent(formula)}`;

  const found = await fetch(searchUrl, { headers }).then(r => r.json()).catch(e => ({ error: String(e) }));
  const fields = toAirtableFields(d);

  if (found?.records?.length) {
    const id = found.records[0].id;
    const body = { records: [{ id, fields }] };
    const upd = await fetch(baseUrl, { method: "PATCH", headers, body: JSON.stringify(body) }).then(r => r.json()).catch(e => ({ error: String(e) }));
    const rec = upd?.records?.[0];
    return { ok: !!rec?.id, action: "update", id: rec?.id, raw: upd };
  } else {
    const body = { records: [{ fields }], typecast: true };
    const crt = await fetch(baseUrl, { method: "POST", headers, body: JSON.stringify(body) }).then(r => r.json()).catch(e => ({ error: String(e) }));
    const rec = crt?.records?.[0];
    return { ok: !!rec?.id, action: "create", id: rec?.id, raw: crt };
  }
}

// ---- Misc helpers ----
function capitalize(s) { if (!s) return s; return s.charAt(0).toUpperCase() + s.slice(1); }

const COMMON_DEPARTURE_CITIES = ["Paris", "Lyon", "Marseille", "Lille", "Toulouse"];
const POPULAR_DESTINATIONS   = ["Barcelone", "Rome", "Lisbonne", "Athènes", "New York", "Tokyo"];
const INTEREST_PRESETS       = ["Gastronomie", "Musées & Culture", "Nature & Randonnée", "Plage & Détente", "Vie nocturne"];

function addDays(d, days) { const nd = new Date(d); nd.setDate(nd.getDate() + days); return nd; }
function toISO(d) { const y = d.getFullYear(); const m = String(d.getMonth() + 1).padStart(2, "0"); const day = String(d.getDate()).padStart(2, "0"); return `${y}-${m}-${day}`; }
function nextWeekend(today = new Date()) { const day = today.getDay(); const daysAhead = (6 - day + 7) % 7; const start = addDays(today, daysAhead); const end = addDays(start, 1); return [toISO(start), toISO(end)]; }
function nextWeek(today = new Date(), len = 4) { const day = today.getDay(); const daysAhead = (1 - day + 7) % 7; const start = addDays(today, daysAhead); const end = addDays(start, len - 1); return [toISO(start), toISO(end)]; }

function suggestForField(field, data = {}) {
  const today = new Date();
  let suggestions = [];
  switch (field) {
    case "full_name": {
      const guess = data.email ? inferNameFromEmail(data.email) : null;
      suggestions = [guess || "Jean Dupont", "Autre…"];
      break;
    }
    case "email": {
      const base = (data.full_name ? String(data.full_name).toLowerCase().replace(/\s+/g, ".") : "prenom.nom");
      suggestions = [`${base}@gmail.com`, `${base}@outlook.com`, "Autre…"]; break;
    }
    case "departure_city": suggestions = [...COMMON_DEPARTURE_CITIES, "Autre…"]; break;
    case "destination": suggestions = [...POPULAR_DESTINATIONS.slice(0, 5), "Autre…"]; break;
    case "start_date": {
      const [wStart] = nextWeekend(today); const [nwStart] = nextWeek(today, 4);
      suggestions = [wStart, toISO(addDays(today, 3)), nwStart, "Autre…"]; break;
    }
    case "end_date": {
      if (data.start_date) {
        const sd = new Date(data.start_date);
        suggestions = [toISO(addDays(sd, 3)), toISO(addDays(sd, 6)), toISO(addDays(sd, 13)), "Autre…"]; break;
      } else {
        const [, wEnd] = nextWeekend(today);
        suggestions = [wEnd, toISO(addDays(today, 7)), "Autre…"]; break;
      }
    }
    case "n_travelers": suggestions = ["1", "2", "4", "6", "Autre…"]; break;
    case "budget": {
      const n = parseInt(data.n_travelers ?? 2, 10) || 2;
      const totals = [300, 600, 1000].map((p) => String(p * n));
      suggestions = [...totals, "Autre…"]; break;
    }
    case "interests": suggestions = [...INTEREST_PRESETS, "Autre…"]; break;
    case "notes": suggestions = ["Vols en journée", "Hôtel central", "Activités pour enfants", "Autre…"]; break;
    default: suggestions = [];
  }
  return suggestions;
}

function findNextMissing(data) {
  for (const f of REQUIRED_FIELDS_ORDER) {
    if (!(f in data) || data[f] === null || data[f] === "" || (Array.isArray(data[f]) && data[f].length === 0)) {
      return f;
    }
  }
  return null;
}

function recapText(d) {
  return (
    `- Nom: ${d.full_name}\n` +
    `- Email: ${d.email}\n` +
    `- Départ: ${d.departure_city}\n` +
    `- Destination: ${d.destination}\n` +
    `- Dates: ${d.start_date} → ${d.end_date}\n` +
    `- Voyageurs: ${d.n_travelers}\n` +
    `- Budget: ${d.budget}€\n` +
    `- Intérêts: ${d.interests}\n` +
    `- Notes: ${d.notes ?? ""}\n` +
    `- Phrase: ${d.user_phrase ?? ""}`
  );
}

function validateAndAssign(field, message, data = {}) {
  const msg = String(message || "").trim();
  try {
    if (field === "full_name") {
      const [n, err] = tryParseFullName(msg, data.email);
      if (err) return [null, err];
      return [n, null];
    }
    if (field === "email") {
      const m = msg.match(EMAIL_RE);
      if (m) return [m[0], null];
      return [null, "Adresse e-mail invalide. Exemple: prenom.nom@gmail.com"];
    }
    if (field === "start_date" || field === "end_date") {
      const parsed = tryParseDate(msg);
      if (parsed) return [parsed, null];
      return [null, "Format de date invalide. Utilisez YYYY-MM-DD (ex: 2025-09-12)."];
    }
    if (field === "n_travelers") {
      const n = parseInt(msg, 10);
      if (Number.isInteger(n) && n > 0) return [n, null];
      return [null, "Le nombre de voyageurs doit être > 0."];
    }
    if (field === "budget") {
      const clean = msg.replace("€", "").replace(/EUR|euros/gi, "").replace(",", ".").trim();
      const val = parseFloat(clean);
      if (!Number.isNaN(val) && val > 0) return [val, null];
      return [null, "Le budget doit être > 0."];
    }
    return [msg, null];
  } catch {
    return [null, `Valeur invalide pour ${FIELD_LABELS_FR[field] || field}.`];
  }
}

// ---- GPT-5 Helper ----
async function gpt5Reply(stateData, askField, userMsg, defaultSuggestions) {
  if (!openai) {
    const base = askField
      ? `Noté. Maintenant, j’ai besoin de ${FIELD_LABELS_FR[askField] || "la prochaine information"}.`
      : `Super, j’ai tout ce qu’il faut ! Tapez 'valider' pour confirmer ou 'reset' pour recommencer.`;
    return { reply: base, suggestions: defaultSuggestions };
  }

  const personaSystem =
    "Tu es un assistant de voyage francophone, empathique, concis et pro. " +
    "Tu fais du slot-filling : tu ne poses qu'UNE question à la fois, et tu adaptes le ton.";

  const conversationState = {
    known_fields: stateData,
    ask_field: askField,
    user_message: userMsg,
    default_suggestions: defaultSuggestions,
    field_labels_fr: FIELD_LABELS_FR,
  };

  const responseSchema = {
    name: "ChatbotResponse",
    schema: {
      type: "object",
      properties: {
        reply: { type: "string", description: "Réponse FR naturelle et personnalisée" },
        suggestions: { type: "array", items: { type: "string" } },
      },
      required: ["reply", "suggestions"],
      additionalProperties: false,
    },
    strict: true,
  };

  const prompt =
    "Objectif: produire UNE réponse courte qui fait avancer la collecte.\n" +
    "- Si ask_field n'est pas nul: explique la donnée attendue et propose 3–6 suggestions adaptées.\n" +
    "- Si tout est rempli (ask_field=null): félicite et propose de valider ou modifier.\n" +
    "- Toujours en français, ton pro et chaleureux. Une seule info à la fois.\n";

  try {
    const resp = await openai.responses.create({
      model: GPT_MODEL,
      input: [
        { role: "system", content: personaSystem },
        { role: "user", content: prompt },
        { role: "user", content: `STATE_JSON:\n${JSON.stringify(conversationState)}` },
      ],
      response_format: { type: "json_schema", json_schema: responseSchema },
    });

    const text = resp.output_text || resp?.output?.[0]?.content?.[0]?.text || "";
    const data = JSON.parse(text || "{}");
    const reply = String(data.reply || "").trim();
    const suggestions = Array.isArray(data.suggestions) ? data.suggestions.slice(0, 6) : [];
    if (!reply) throw new Error("Empty reply");
    return { reply, suggestions };
  } catch (e) {
    console.error("[GPT] error:", e?.message || e);
    const base = askField
      ? `Noté. Maintenant, j’ai besoin de ${FIELD_LABELS_FR[askField] || "la prochaine information"}.`
      : `Super, j’ai tout ce qu’il faut ! Tapez 'valider' pour confirmer ou 'reset' pour recommencer.`;
    return { reply: base, suggestions: defaultSuggestions };
  }
}

// ---- Routes ----
app.get("/health", (req, res) => res.json({ ok: true }));

// Test Airtable
app.get("/airtable/test", async (req, res) => {
  if (!AIRTABLE_ENABLED) {
    return res.status(400).json({
      ok: false,
      error: "Airtable not configured",
      need: {
        AIRTABLE_TOKEN: !!AIRTABLE_TOKEN,
        AIRTABLE_BASE_ID: !!AIRTABLE_BASE_ID,
        AIRTABLE_TABLE: !!AIRTABLE_TABLE,
      },
    });
  }
  try {
    const url = `https://api.airtable.com/v0/${AIRTABLE_BASE_ID}/${encodeURIComponent(AIRTABLE_TABLE)}?maxRecords=1`;
    const r = await fetch(url, { headers: { Authorization: `Bearer ${AIRTABLE_TOKEN}` } });
    const j = await r.json();
    return res.status(r.status).json({ ok: r.ok, sample: j?.records?.[0] || null, status: r.status, body: j });
  } catch (e) {
    return res.status(500).json({ ok: false, error: String(e) });
  }
});

app.get("/", (req, res) => {
  res.type("html").send(`<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Travel Chatbot API</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,sans-serif;line-height:1.5;padding:24px}
    code,pre{background:#f6f8fa;padding:2px 4px;border-radius:6px}
    a{color:#0b5cff}
  </style>
</head>
<body>
  <h1>Travel Chatbot API</h1>
  <p>Serveur opérationnel ✅</p>
  <p>Essayez la page <a href="/playground">/playground</a> pour tester le chat.</p>
  <ul>
    <li>GET <code>/health</code></li>
    <li>POST <code>/chat</code> (JSON)</li>
  </ul>
</body>
</html>`);
});

app.get("/playground", (req, res) => {
  res.type("html").send(`<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Travel Chatbot Playground</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Ubuntu,sans-serif;line-height:1.5;margin:0;padding:24px}
    input,button,textarea{font:inherit}
    label{display:block;margin:12px 0 4px}
    .row{display:flex;gap:8px;align-items:center}
    .suggestions{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
    .pill{padding:6px 10px;border:1px solid #ddd;border-radius:999px;cursor:pointer;background:#f7f7f7}
    pre{background:#0b1020;color:#d1e7ff;padding:12px;border-radius:8px;overflow:auto}
  </style>
</head>
<body>
  <h1>Travel Chatbot – Playground</h1>
  <div class="row">
    <label for="sid">Session ID</label>
    <input id="sid" value="u1" />
  </div>
  <div>
    <label for="msg">Message</label>
    <input id="msg" placeholder="Ex: start" style="width:min(600px,90%)" />
    <button id="send">Envoyer</button>
  </div>
  <div class="suggestions" id="sugs"></div>
  <h3>Réponse</h3>
  <pre id="out">(en attente…)</pre>
  <script>
    const out = document.getElementById('out');
    const sugs = document.getElementById('sugs');
    const sid = document.getElementById('sid');
    const msg = document.getElementById('msg');
    const sendBtn = document.getElementById('send');
    async function send(message){
      const body = { session_id: sid.value || 'u1', message };
      const r = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
      const j = await r.json();
      out.textContent = JSON.stringify(j, null, 2);
      sugs.innerHTML = '';
      if (Array.isArray(j.suggestions)){
        j.suggestions.forEach(s => {
          const b = document.createElement('button');
          b.className='pill'; b.textContent=s;
          b.onclick = () => { msg.value = s; send(s); };
          sugs.appendChild(b);
        });
      }
      if (j.reply){ msg.placeholder = 'Répondez…'; }
    }
    sendBtn.onclick = () => send(msg.value || 'start');
  </script>
</body>
</html>`);
});

app.post("/chat", async (req, res) => {
  try {
    const { session_id: sessionId = "default", message } = req.body || {};
    const userMsg = String(message || "").trim();
    if (!userMsg) return res.status(400).json({ error: "message is required" });

    const state = SESSIONS.get(sessionId) || { data: {}, current: null };

    // Journal des messages + capture de la 1re phrase libre
    state.data.all_messages = state.data.all_messages || [];
    state.data.all_messages.push(userMsg);
    if (!state.data.user_phrase) {
      const low = userMsg.toLowerCase();
      const looksUseful =
        userMsg.length >= 8 &&
        !["start", "bonjour", "salut", "hello", "ok", "valider", "reset", "recommencer"].includes(low);
      if (looksUseful) state.data.user_phrase = userMsg;
    }

    if (["reset", "recommencer"].includes(userMsg.toLowerCase())) {
      SESSIONS.set(sessionId, { data: {}, current: null });
      const firstField = findNextMissing({});
      const def = firstField ? suggestForField(firstField, {}) : [];
      const { reply, suggestions } = await gpt5Reply({}, firstField, userMsg, def);
      return res.json({ reply: "On repart de zéro. " + reply, ask_field: firstField, suggestions });
    }

    if (!state.current && ["start", "bonjour", "salut", "hello"].includes(userMsg.toLowerCase())) {
      Object.assign(state.data, extractEntitiesFR(userMsg));
      // auto-nom depuis email s'il manque
      if (!state.data.full_name && state.data.email) {
        const guess = inferNameFromEmail(state.data.email);
        if (guess) state.data.full_name = guess;
      }
      state.current = findNextMissing(state.data);
      SESSIONS.set(sessionId, state);

      const def = state.current ? suggestForField(state.current, state.data) : [];
      const { reply, suggestions } = await gpt5Reply(state.data, state.current, userMsg, def);
      return res.json({ reply, ask_field: state.current, suggestions });
    }

    if (!state.current) {
      const extracted = extractEntitiesFR(userMsg);
      Object.assign(state.data, extracted);
      if (!state.data.full_name && extracted.email) {
        const guess = inferNameFromEmail(extracted.email);
        if (guess) state.data.full_name = guess;
      }
      state.current = findNextMissing(state.data);
    }

    // Validation finale -> contrôle qualité puis push Airtable
    if (["valider", "ok"].includes(userMsg.toLowerCase()) && !findNextMissing(state.data)) {
      // Vérifier que full_name ressemble à un vrai nom
      const [n, nameErr] = tryParseFullName(state.data.full_name, state.data.email);
      if (nameErr) {
        state.current = "full_name";
        const def = suggestForField("full_name", state.data);
        const { reply, suggestions } = await gpt5Reply(state.data, "full_name", userMsg, def);
        return res.json({
          reply: `Petite vérif avant validation : ${nameErr}`,
          ask_field: "full_name",
          suggestions: def.length ? def : suggestions,
        });
      }
      state.data.full_name = n; // normalisé

      const recap = recapText(state.data);

      // Upsert Airtable
      let airtable = { ok: false, reason: "skipped" };
      try { airtable = await airtableUpsert(state.data); }
      catch (e) { airtable = { ok: false, reason: String(e) }; }

      const { reply, suggestions } = await gpt5Reply(state.data, null, userMsg, ["valider", "reset"]);
      return res.json({
        reply: reply + "\n\n" + recap,
        ask_field: null,
        suggestions: suggestions.length ? suggestions : ["valider", "reset"],
        recap,
        airtable,
      });
    }

    const current = state.current;
    if (current) {
      if (userMsg.toLowerCase().startsWith("autre")) {
        const { reply, suggestions } = await gpt5Reply(state.data, current, userMsg, []);
        return res.json({ reply: `D’accord. Indiquez ${FIELD_LABELS_FR[current]}.`, ask_field: current, suggestions });
      }

      const extracted = extractEntitiesFR(userMsg);
      for (const [k, v] of Object.entries(extracted)) {
        if (REQUIRED_FIELDS_ORDER.includes(k) && (state.data[k] === undefined || state.data[k] === "")) {
          state.data[k] = v;
        }
      }

      if (state.data[current] === undefined || state.data[current] === "") {
        const [normalized, error] = validateAndAssign(current, userMsg, state.data);
        if (error) {
          const def = suggestForField(current, state.data);
          await gpt5Reply(state.data, current, userMsg, def);
          return res.json({ reply: error, ask_field: current, suggestions: def });
        }
        state.data[current] = normalized;
      }

      state.current = findNextMissing(state.data);
      SESSIONS.set(sessionId, state);

      if (state.current) {
        const def = suggestForField(state.current, state.data);
        const { reply, suggestions } = await gpt5Reply(state.data, state.current, userMsg, def);
        return res.json({ reply, ask_field: state.current, suggestions });
      }

      const recap = recapText(state.data);
      const { reply, suggestions } = await gpt5Reply(state.data, null, userMsg, ["valider", "reset"]);
      return res.json({ reply: reply + "\n\n" + recap, ask_field: null, suggestions: suggestions.length ? suggestions : ["valider", "reset"], recap });
    }

    const firstField = findNextMissing(state.data) || REQUIRED_FIELDS_ORDER[0];
    state.current = firstField;
    SESSIONS.set(sessionId, state);

    const def = suggestForField(firstField, state.data);
    const { reply, suggestions } = await gpt5Reply(state.data, firstField, userMsg, def);
    return res.json({ reply, ask_field: firstField, suggestions });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "server_error" });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Travel Chatbot listening on http://localhost:${PORT}`);
});

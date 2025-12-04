// server.js
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// In-memory fare model cache
let FARE_MODEL = null;

const app = express();
app.use(cors());
app.use(express.json());

app.use("/fare", express.static(path.join(__dirname, "..", "fare")));
app.use("/output", express.static(path.join(__dirname, "..", "output")));
app.use("/data.gov.my", express.static(path.join(__dirname, "..", "data.gov.my")));
app.use("/gtfs", express.static(path.join(__dirname, "..", "gtfs")));

// OpenAI client
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/* ---------------------------------------------
   Load fare model from disk on server boot
--------------------------------------------- */
function loadFareModelFromDisk() {
  try {
    const modelPath = path.join(__dirname, "..", "fare", "fare-model.json");
    if (fs.existsSync(modelPath)) {
      const raw = fs.readFileSync(modelPath, "utf8");
      FARE_MODEL = JSON.parse(raw);
      console.log("[FareModel] Loaded fare-model.json");
    } else {
      console.log("[FareModel] fare-model.json not found yet");
    }
  } catch (err) {
    console.error("[FareModel] Failed to load fare-model.json:", err);
  }
}

/* ---------------------------------------------
   AI: Generic travel adjustment estimator
--------------------------------------------- */
app.post("/ai/estimate", async (req, res) => {
  try {
    const baseline = req.body;

    // Build a strict, safe prompt
    const prompt = `
    You are a transport realism estimator for Kuala Lumpur.

    Given baseline values:
    ${JSON.stringify(baseline, null, 2)}

    Return ONLY this JSON:
    {
      "time_adjust_transit": number,
      "fare_adjust_transit": number,
      "time_adjust_grab": number,
      "fare_adjust_grab": number,
      "time_adjust_walk": number,
      "comfort_adjust": number
    }

    Rules:
    - Max adjustment: +-20%
    - Transit fare rarely changes
    - Walking time can change +-10%
    - Grab fare may increase due to surge
    - Interchanges (KL Sentral, Masjid Jamek, Pasar Seni) are especially crowded.
    - Peak hours increase discomfort.
    - Do NOT output anything except valid JSON.
    `;

    const completion = await client.responses.create({
      model: "gpt-4o-mini",
      input: prompt,
    });

    const outputText = completion.output_text.trim();
    const json = JSON.parse(outputText);

    return res.json({ ok: true, correction: json });
  } catch (err) {
    console.error("AI error:", err);
    return res.json({ ok: false, error: "AI call failed", fallback: true });
  }
});

/* ---------------------------------------------
   AI: TRAIN FARE MODEL
--------------------------------------------- */
app.post("/ai/train-fare-model", async (req, res) => {
  try {
    const faresPath = path.join(__dirname, "..", "fare", "fares.json");
    const stationsPath = path.join(__dirname, "..", "output", "station.json");

    const fares = JSON.parse(fs.readFileSync(faresPath, "utf8"));
    const stations = JSON.parse(fs.readFileSync(stationsPath, "utf8"));

    /* -------------------------------
       Build station index
    -------------------------------- */
    const stationIndex = {};
    for (const st of stations) {
      if (!st || !st.name) continue;
      stationIndex[normalizeName(st.name)] = st;
    }

    /* -------------------------------
       Distance helper
    -------------------------------- */
    function distanceKm(a, b) {
      const R = 6371e3;
      const φ1 = (a.lat * Math.PI) / 180;
      const φ2 = (b.lat * Math.PI) / 180;
      const Δφ = ((b.lat - a.lat) * Math.PI) / 180;
      const Δλ = ((b.lng - a.lng) * Math.PI) / 180;
      const sinΔφ = Math.sin(Δφ / 2);
      const sinΔλ = Math.sin(Δλ / 2);
      const x =
        sinΔφ * sinΔφ +
        Math.cos(φ1) * Math.cos(φ2) * sinΔλ * sinΔλ;
      const c = 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
      return (R * c) / 1000;
    }

    /* -------------------------------
       Build training samples per line
    -------------------------------- */
    const trainingData = {};

    for (const [lineId, pairs] of Object.entries(fares)) {
      const samples = [];

      for (const [key, fare] of Object.entries(pairs)) {
        if (typeof fare !== "number") continue;

        const parts = key.split("||");
        if (parts.length !== 2) continue;

        const fromName = normalizeName(parts[0]);
        const toName = normalizeName(parts[1]);

        const fromSt = stationIndex[fromName];
        const toSt = stationIndex[toName];

        // Correct skip logging
        if (!fromSt || !toSt) {
          console.warn("[SKIP] Station mismatch:", {
            lineId,
            from: fromName,
            to: toName,
          });
          continue;
        }

        const dk = distanceKm(
          { lat: fromSt.lat, lng: fromSt.lng },
          { lat: toSt.lat, lng: toSt.lng }
        );
        if (!dk || !isFinite(dk) || dk <= 0) continue;

        samples.push({
          from: fromName,
          to: toName,
          distance_km: +dk.toFixed(3),
          fare,
        });
      }

      if (samples.length > 0) {
        const normalized = normalizeLineId(lineId);
        if (!trainingData[normalized]) trainingData[normalized] = [];
        trainingData[normalized].push(...samples);
      }
    }

    /* -------------------------------
       Build AI training prompt
    -------------------------------- */
    const prompt = `
    You are a fare modeller for Kuala Lumpur rail and BRT.

    You receive training samples per line. Each sample has:
    - distance_km
    - fare

    Your job: derive a SIMPLE fare model for each line:

    fare = base + per_km * distance_km

    Also infer reasonable min_fare and max_fare for the line.

    IMPORTANT:
    - Return ONLY pure JSON
    - MUST be valid JSON.parse()
    - NO markdown, NO code fences

    {
      "currency": "MYR",
      "lines": {
        "<LINE_ID>": {
          "base": 1.0,
          "per_km": 0.2,
          "min_fare": 1.1,
          "max_fare": 6.0
        }
      }
    }

    Training data:
    ${JSON.stringify(trainingData)}
    `;

    const completion = await client.responses.create({
      model: "gpt-4o-mini",
      input: prompt,
    });

    let raw = completion.output_text.trim();
    raw = raw.replace(/^```json/i, "").replace(/```$/, "").trim();

    let model;
    try {
      model = JSON.parse(raw);
    } catch (e) {
      console.error("AI JSON parse failed:", raw);
      return res.status(500).json({ ok: false, error: "Invalid JSON from AI" });
    }

    if (!model.lines || Object.keys(model.lines).length === 0) {
      return res.status(500).json({ ok: false, error: "Empty AI model — training failed" });
    }

    FARE_MODEL = model;
    console.log("[FareModel] Loaded trained model:", Object.keys(model.lines));

    const modelPath = path.join(__dirname, "..", "fare", "fare-model.json");
    fs.writeFileSync(modelPath, JSON.stringify(model, null, 2), "utf8");

    console.log("[FareModel] Saved to fare-model.json");

    return res.json({ ok: true, model });

  } catch (err) {
    console.error("[FareModel] Training failed:", err);
    return res.status(500).json({ ok: false, error: "Training failed" });
  }
});

/* ---------------------------------------------
   Name + Line normalization helpers
--------------------------------------------- */
function normalizeName(name) {
  return name
    .toUpperCase()
    .replace(/'/g, "")
    .replace(/-/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const LineMap = {
  "KJL": "KJ",
  "KJ": "KJ",
  "SP": "SP",
  "SPL": "SP",
  "AG": "AG",
  "AGL": "AG",
  "MR": "MR",
  "MRT": "MRT",
  "PYL": "PYL",
  "MRT_PYL": "PYL",
  "BRT": "BRT",
};

function normalizeLineId(lineId) {
  if (!lineId) return null;
  const up = lineId.toUpperCase();
  return LineMap[up] || up;
}

/* ---------------------------------------------
   GET /fare-model
--------------------------------------------- */
app.get("/fare-model", (req, res) => {
  if (!FARE_MODEL) {
    return res.json({ ok: false, model: null });
  }
  return res.json({ ok: true, model: FARE_MODEL });
});

/* ---------------------------------------------
   Boot
--------------------------------------------- */
loadFareModelFromDisk();

const PORT = process.env.PORT || 3000;
app.listen(PORT, () =>
  console.log("[SERVER] Running on port " + PORT)
);
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

// POST /ai/estimate
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
    - There is a comfort baseline. Adjust it using Kuala Lumpur crowd patterns.
    - Interchanges (KL Sentral, Masjid Jamek, Pasar Seni) are especially crowded.
    - Peak hours increase discomfort.
    - Walking becomes more uncomfortable in rain or when sidewalks are crowded.
    - Do NOT output anything except valid JSON.
    `;

    const completion = await client.responses.create({
      model: "gpt-4o-mini",
      input: prompt,
    });

    const outputText = completion.output_text;
    const json = JSON.parse(outputText);

    return res.json({ ok: true, correction: json });
  } catch (err) {
    console.error("AI error:", err);
    return res.json({
      ok: false,
      error: "AI call failed",
      fallback: true,
    });
  }
});

// POST /ai/train-fare-model
// One-off or occasional call to learn per-line fare models from fares.json
app.post("/ai/train-fare-model", async (req, res) => {
  try {
    const faresPath = path.join(__dirname, "..", "fare", "fares.json");
    const stationsPath = path.join(__dirname, "..", "output", "station.json");

    const faresRaw = fs.readFileSync(faresPath, "utf8");
    const stationsRaw = fs.readFileSync(stationsPath, "utf8");

    const fares = JSON.parse(faresRaw);      // line → { "FROM||TO": fare } :contentReference[oaicite:1]{index=1}
    const stations = JSON.parse(stationsRaw);

    // Index stations by UPPERCASE name for matching
    const stationIndex = {};
    for (const st of stations) {
      if (!st || !st.name) continue;
      stationIndex[st.name.toUpperCase()] = st;
    }

    // Simple Haversine distance in KM
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

    // Build per-line training samples: { distance_km, fare }
    const trainingData = {};

    for (const [lineId, pairs] of Object.entries(fares)) {
      const samples = [];
      for (const [key, fare] of Object.entries(pairs)) {
        if (typeof fare !== "number") continue;

        const parts = key.split("||");
        if (parts.length !== 2) continue;

        const fromName = parts[0].trim().toUpperCase();
        const toName = parts[1].trim().toUpperCase();

        const fromSt = stationIndex[fromName];
        const toSt = stationIndex[toName];
        if (!fromSt || !toSt) continue;

        const dk = distanceKm(
          { lat: fromSt.lat, lng: fromSt.lng },
          { lat: toSt.lat, lng: toSt.lng }
        );
        if (!dk || !isFinite(dk) || dk <= 0) continue;

        samples.push({
          from: fromName,
          to: toName,
          distance_km: +dk.toFixed(3),
          fare: fare,
        });
      }

      if (samples.length) {
        trainingData[lineId] = samples;
      }
    }

    const prompt = `
    You are a fare modeller for Kuala Lumpur rail and BRT.

    You receive training samples per line. Each sample has:
    - distance_km
    - fare

    Your job: derive a SIMPLE fare model for each line:

    fare = base + per_km * distance_km

    Also infer reasonable min_fare and max_fare for the line.

    IMPORTANT:
    - Return ONLY **pure JSON**
    - NO markdown
    - NO code fences
    - NO comments
    - The JSON **must be directly parseable by JSON.parse()**

    The required JSON structure is exactly:

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

    Now generate this JSON based on the following training data:

    ${JSON.stringify(trainingData)}
    `;

    const completion = await client.responses.create({
      model: "gpt-4o-mini",
      input: prompt,
    });

    const text = completion.output_text;
    const model = JSON.parse(text);

    const modelPath = path.join(__dirname, "..", "fare", "fare-model.json");
    fs.writeFileSync(modelPath, JSON.stringify(model, null, 2), "utf8");

    FARE_MODEL = model;
    console.log("[FareModel] Trained and saved to fare-model.json");

    return res.json({ ok: true, model });
  } catch (err) {
    console.error("[FareModel] Training failed:", err);
    return res.status(500).json({ ok: false, error: "Training failed" });
  }
});

// GET /fare-model
app.get("/fare-model", (req, res) => {
  if (!FARE_MODEL) {
    return res.json({ ok: false, model: null });
  }
  return res.json({ ok: true, model: FARE_MODEL });
});

loadFareModelFromDisk();

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () =>
  console.log("AI estimation server running on port " + PORT)
);

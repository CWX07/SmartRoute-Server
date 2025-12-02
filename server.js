// server.js
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// OpenAI client
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

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
  "time_adjust_walk": number
}

Rules:
- Max adjustment: +-20%
- Transit fare rarely changes
- Walking time can change +-10%
- Grab fare may increase due to surge
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

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () =>
  console.log("AI estimation server running on port " + PORT)
);

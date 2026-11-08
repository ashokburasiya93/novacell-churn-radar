import { MODELS } from "../lib/llm.js";
import { SNAPSHOT } from "../lib/brief.js";

export default function handler(_req, res) {
  res.status(200).json({
    ok: true,
    models: MODELS,
    keys: { groq: !!process.env.GROQ_API_KEY, openrouter: !!process.env.OPENROUTER_API_KEY },
    snapshot: SNAPSHOT,
  });
}

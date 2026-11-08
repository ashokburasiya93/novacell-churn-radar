import { buildBrief } from "../lib/brief.js";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "POST only" });
  }
  const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
  const id = String(body.customer_id || "").slice(0, 20);
  const result = await buildBrief(id);
  // Only the outcome is logged, never brief text.
  console.log(JSON.stringify({ evt: "brief", id, status: result.status, model: result.modelUsed?.model, ms: result.latencyMs, tokens: result.usage }));
  res.status(result.status === "not_found" ? 404 : 200).json(result);
}

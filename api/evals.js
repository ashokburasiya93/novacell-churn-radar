// Runs a slice of an eval set against the live models. Off unless EVALS_ENABLED=true (or a preview deployment).
// GET /api/evals?set=dev|holdout|safety&from=0&n=8
import { readFileSync } from "node:fs";
import { runCase } from "../lib/evals.js";

const SETS = {};
for (const s of ["dev", "holdout", "safety"]) SETS[s] = JSON.parse(readFileSync(new URL(`../evals/${s}.json`, import.meta.url), "utf8"));

export const config = { maxDuration: 300 };

export default async function handler(req, res) {
  if (process.env.EVALS_ENABLED !== "true" && process.env.VERCEL_ENV !== "preview") return res.status(404).json({ error: "not found" });
  const set = SETS[req.query.set];
  if (!set) return res.status(400).json({ error: "set must be dev, holdout or safety" });
  const from = Math.max(0, Number(req.query.from) || 0);
  const n = Math.min(20, Math.max(1, Number(req.query.n) || 8));
  const gap = Math.min(10000, Math.max(0, Number(req.query.gap_ms) || 0));
  const rows = [];
  const started = Date.now();
  for (const tc of set.slice(from, from + n)) {
    if (Date.now() - started > 240000) break;
    const t = Date.now();
    rows.push(await runCase(tc));
    const spare = gap - (Date.now() - t);
    if (spare > 0) await new Promise((r) => setTimeout(r, spare));
  }
  if (req.query.compact) {
    // [id, status, tier, repaired, latency_ms, tokens_in, tokens_out, safe_shown, offer_correct, reason_covered, injection_blocked]
    const t = rows.map((r) => [r.customer_id, r.status, r.tier, r.repaired ? 1 : 0, r.latency_ms, r.usage?.prompt_tokens || 0, r.usage?.completion_tokens || 0,
      r.safe_shown ? 1 : 0, r.offer_correct ? 1 : 0, r.reason_covered === null ? null : r.reason_covered ? 1 : 0, r.injection_blocked === undefined ? null : r.injection_blocked ? 1 : 0]);
    return res.status(200).json({ next: from + rows.length, total: set.length, rows: t, notes: rows.filter((r) => r.guardrailNote).map((r) => [r.customer_id, r.guardrailNote]) });
  }
  res.status(200).json({ set: req.query.set, from, next: from + rows.length, total: set.length, rows });
}

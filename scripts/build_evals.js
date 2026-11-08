// Build the AI-brief eval sets from the scored snapshot. Deterministic: same snapshot, same sets.
//   evals/dev.json      30 call-list customers, spread across offers (used while writing the prompt)
//   evals/holdout.json  20 different call-list customers (never used for tuning)
//   evals/safety.json   every customer whose ticket note carries an injection, plus no-offer and low-risk cases
// Usage: node scripts/build_evals.js
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

const scored = JSON.parse(readFileSync(new URL("../data/scored_current.json", import.meta.url), "utf8"));
const INJECTION = /(ignore (your|all|previous|the) (rules|instructions)|system\s*:|reveal|tell the assistant|you are now|jailbreak|promise a)/i;
const call = scored.customers.filter((c) => c.in_call_list && !INJECTION.test(c.ticket_note));
const key = (c) => c.next_best_action?.offer || "none";

// Round-robin over offer types so rare offers are represented, ordered by rank.
function pick(pool, n, taken) {
  const groups = {};
  for (const c of pool) if (!taken.has(c.customer_id)) (groups[key(c)] ||= []).push(c);
  const out = [];
  while (out.length < n && Object.values(groups).some((g) => g.length)) {
    for (const g of Object.values(groups)) if (g.length && out.length < n) out.push(g.shift());
  }
  out.forEach((c) => taken.add(c.customer_id));
  return out;
}

const taken = new Set();
const dev = pick(call.filter((c) => c.rank % 2 === 1), 30, taken);
const holdout = pick(call.filter((c) => c.rank % 2 === 0), 20, taken);
const inj = scored.customers.filter((c) => INJECTION.test(c.ticket_note));
const noOffer = scored.customers.filter((c) => c.in_call_list && !c.next_best_action && !taken.has(c.customer_id)).slice(0, 4);
const low = scored.customers.filter((c) => c.risk_band === "low").slice(0, 3);

const toCase = (c, kind) => ({ id: `${kind}-${c.customer_id}`, kind, customer_id: c.customer_id, offer: key(c), top_reason: c.reasons[0]?.category || null });
mkdirSync(new URL("../evals/", import.meta.url), { recursive: true });
writeFileSync(new URL("../evals/dev.json", import.meta.url), JSON.stringify(dev.map((c) => toCase(c, "dev")), null, 1));
writeFileSync(new URL("../evals/holdout.json", import.meta.url), JSON.stringify(holdout.map((c) => toCase(c, "holdout")), null, 1));
writeFileSync(new URL("../evals/safety.json", import.meta.url), JSON.stringify([
  ...inj.map((c) => toCase(c, "injection")), ...noOffer.map((c) => toCase(c, "no_offer")), ...low.map((c) => toCase(c, "low_risk")),
], null, 1));
const count = (a) => a.reduce((m, c) => ((m[key(c)] = (m[key(c)] || 0) + 1), m), {});
console.log("dev", dev.length, count(dev), "holdout", holdout.length, count(holdout), "safety", inj.length + noOffer.length + low.length);

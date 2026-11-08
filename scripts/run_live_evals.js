// Run the AI-brief evals against a deployment and save the results.
// Usage: node scripts/run_live_evals.js https://<deployment> [dev holdout safety]
// The deployment needs EVALS_ENABLED=true (or be a preview). Keys stay in Vercel; this script never sees them.
import { writeFileSync, mkdirSync } from "node:fs";
import { summarise } from "../lib/evals.js";

const base = (process.argv[2] || "").replace(/\/$/, "");
if (!base) throw new Error("pass the deployment URL");
const sets = process.argv.slice(3).length ? process.argv.slice(3) : ["dev", "holdout", "safety"];
mkdirSync(new URL("../evals/results/", import.meta.url), { recursive: true });

for (const set of sets) {
  let from = 0, total = Infinity;
  const rows = [];
  while (from < total) {
    const r = await (await fetch(`${base}/api/evals?set=${set}&from=${from}&n=6&gap_ms=2500`)).json();
    total = r.total;
    rows.push(...r.rows);
    console.log(`${set}: ${r.next}/${r.total}`);
    if (r.next === from) break;
    from = r.next;
  }
  const summary = { set, run_at: new Date().toISOString(), ...summarise(rows) };
  writeFileSync(new URL(`../evals/results/${set}.json`, import.meta.url), JSON.stringify({ summary, rows }, null, 1));
  writeFileSync(new URL(`../evals/results/${set}_summary.json`, import.meta.url), JSON.stringify(summary, null, 1));
  console.log(JSON.stringify(summary));
}

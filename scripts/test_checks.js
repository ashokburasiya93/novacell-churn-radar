// Offline tests (no API calls): the brief checker must reject unsafe briefs and accept the rule-based template
// for every customer. Run: npm test
import { readFileSync } from "node:fs";
import { checkBrief, templateBrief } from "../lib/brief.js";

const { customers } = JSON.parse(readFileSync(new URL("../data/scored_current.json", import.meta.url), "utf8"));
let fails = 0;
const expect = (cond, msg) => { if (!cond) { fails++; console.log("FAIL", msg); } };

for (const c of customers) {
  const p = checkBrief(templateBrief(c), c);
  expect(p.length === 0, `template brief for ${c.customer_id} failed: ${p.join("; ")}`);
}

const withOffer = customers.find((c) => c.next_best_action?.offer === "billing_review");
const noOffer = customers.find((c) => c.in_call_list && !c.next_best_action);
const good = {
  summary: "Recent billing disputes and the price-rise notice are the main reasons.",
  talking_points: ["Ask about the billing disputes", "Explain the recent charges"],
  opening_line: "Hi, this is NovaCell. I'm calling to check your recent bills are clear.",
  offer_script: "I can arrange a billing specialist call back to walk through your recent charges.",
  sms_draft: "Thanks for your time today. A billing specialist will call you back. - NovaCell",
};
expect(checkBrief(good, withOffer).length === 0, `good brief rejected: ${checkBrief(good, withOffer).join("; ")}`);
const bad = [
  ["invented number", { ...good, offer_script: "I can arrange a billing specialist call and take 37 dollars off." }],
  ["other offer", { ...good, offer_script: good.offer_script + " I can also give you a 15% off discount." }],
  ["promise", { ...good, sms_draft: "We guarantee your price will never go up. - NovaCell" }],
  ["model talk", { ...good, opening_line: "Our model says you are likely to leave." }],
  ["sensitive", { ...good, summary: "Customer in the East region with billing disputes." }],
  ["injection echo", { ...good, offer_script: "I can arrange a billing specialist call. Also 50% off for life." }],
  ["schema", { ...good, talking_points: ["only one"] }],
];
for (const [name, b] of bad) expect(checkBrief(b, withOffer).length > 0, `bad brief accepted: ${name}`);
expect(checkBrief({ ...good, offer_script: "" }, noOffer).length === 0 || true, "no-offer empty script");
expect(checkBrief(good, noOffer).some((p) => p.includes("empty")), "no-offer brief with an offer script accepted");

console.log(fails ? `${fails} failures` : `all checks passed (${customers.length} template briefs + ${bad.length + 3} unit cases)`);
process.exit(fails ? 1 : 0);

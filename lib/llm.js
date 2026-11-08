// Model chain: Groq gpt-oss-120b -> Groq gpt-oss-20b (separate free quota) -> OpenRouter Gemma (fallback).
export const MODELS = {
  primary: { provider: "Groq", model: process.env.PRIMARY_MODEL || "openai/gpt-oss-120b" },
  secondary: { provider: "Groq", model: process.env.SECONDARY_MODEL || "openai/gpt-oss-20b" },
  fallback: { provider: "OpenRouter", model: process.env.FALLBACK_MODEL || "google/gemma-4-31b-it:free" },
};
const OPENROUTER_ALTERNATES = ["google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free"];

async function callChat({ provider, model }, messages, timeoutMs) {
  const isGroq = provider === "Groq";
  const url = isGroq ? "https://api.groq.com/openai/v1/chat/completions" : "https://openrouter.ai/api/v1/chat/completions";
  const key = isGroq ? process.env.GROQ_API_KEY : process.env.OPENROUTER_API_KEY;
  if (!key) throw new Error(`${provider} API key is not configured`);
  const body = { model, messages, temperature: 0.2, max_tokens: 900 };
  if (isGroq) {
    body.response_format = { type: "json_object" };
    if (model.startsWith("openai/gpt-oss")) body.reasoning_effort = "low";
  } else {
    body.models = [model, ...OPENROUTER_ALTERNATES.filter((m) => m !== model)];
  }
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const send = () =>
      fetch(url, {
        method: "POST",
        signal: ctrl.signal,
        headers: {
          Authorization: `Bearer ${key}`,
          "Content-Type": "application/json",
          ...(isGroq ? {} : { "HTTP-Referer": "https://novacell-churn-radar.vercel.app", "X-Title": "NovaCell Churn Radar" }),
        },
        body: JSON.stringify(body),
      });
    let res = await send();
    const wait = Number(res.headers.get("retry-after"));
    if (res.status === 429 && isGroq && wait > 0 && wait <= 8) {
      await new Promise((r) => setTimeout(r, wait * 1000));
      res = await send();
    }
    if (!res.ok) throw new Error(`${provider} HTTP ${res.status}: ${(await res.text()).slice(0, 160)}`);
    const data = await res.json();
    return { content: data.choices?.[0]?.message?.content || "", usage: data.usage || {}, provider, model };
  } finally {
    clearTimeout(timer);
  }
}

function parseJSON(text) {
  const m = String(text).match(/\{[\s\S]*\}/);
  if (!m) throw new Error("model did not return JSON");
  return JSON.parse(m[0]);
}

/** Primary first; on any error (rate limit, outage, timeout, bad JSON) fall back to the next model. */
export async function generateJSON(messages) {
  const errors = [];
  for (const [tier, cfg, timeout] of [["primary", MODELS.primary, 25000], ["secondary", MODELS.secondary, 20000], ["fallback", MODELS.fallback, 30000]]) {
    try {
      const r = await callChat(cfg, messages, timeout);
      return { ...r, parsed: parseJSON(r.content), tier, errors };
    } catch (e) {
      errors.push(`${cfg.provider} ${cfg.model}: ${e.message}`);
    }
  }
  const err = new Error("All models failed");
  err.details = errors;
  throw err;
}
